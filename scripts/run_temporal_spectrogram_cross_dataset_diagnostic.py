#!/usr/bin/env python3
"""Resumable development-only MATLAB/VSB temporal-vs-STFT diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from partial_discharge_adaptive_fusion.config import raw_data_root, require_cuda, runtime_info  # noqa: E402
from partial_discharge_adaptive_fusion.dataset import (  # noqa: E402
    iter_vsb_signal_batches, load_mat_partition, load_vsb_metadata, resolve_dataset,
)
from partial_discharge_adaptive_fusion.evaluation import (  # noqa: E402
    binary_metrics, grouped_paired_bootstrap_delta, hierarchical_grouped_delta_ci,
    hierarchical_paired_delta_ci, paired_bootstrap_delta,
)
from partial_discharge_adaptive_fusion.fusion import select_threshold  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.data import NumpyDataset  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.train import predict_logits  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.pulse_data import PulseBagDataset  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.pulse_train import (  # noqa: E402
    _loader as pulse_loader, cross_fitted_pulse_logits, predict_pulse_outputs, train_pulse_expert,
)
from partial_discharge_adaptive_fusion.modeling.train import (  # noqa: E402
    cross_fitted_logits, fold_assignments, positive_class_weight, train_expert,
)
from partial_discharge_adaptive_fusion.pulse_impl import PulsePolicy  # noqa: E402
from partial_discharge_adaptive_fusion.protocol import MATLAB_DATASET_ID, VSB_DATASET_ID  # noqa: E402
from partial_discharge_adaptive_fusion.representations import fit_standardizer, temporal_input  # noqa: E402
from partial_discharge_adaptive_fusion.spectrogram import (  # noqa: E402
    IndexedArrayView, STFTSpec, build_matlab_spectrogram_dataset, build_vsb_event_spectrogram_dataset,
    compute_stft_log_power, fit_spectrogram_standardizer, write_atomic_array_cache,
)
from partial_discharge_adaptive_fusion.temporal_spectrogram import (  # noqa: E402
    classify_cross_dataset_verdict, prediction_overlap_oracle, probability_summary,
    select_fixed_weight_oof, threshold_curve,
)
from partial_discharge_adaptive_fusion.vsb_forensics import (  # noqa: E402
    evaluate_feature_baseline_with_predictions, prepare_forensic_baseline_frame,
)
from partial_discharge_adaptive_fusion.vsb_modality import (  # noqa: E402
    build_modality_feature_inventory, feature_set_columns, validate_modality_source_artifacts,
)

START = time.perf_counter()


def _standardizer_fingerprint(standardizer: Any) -> str:
    return hashlib.sha256(standardizer.mean.tobytes() + standardizer.std.tobytes()).hexdigest()


class ElapsedFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        return f"[{stamp}] [elapsed={time.perf_counter() - START:010.1f}s] {record.getMessage()}"


def make_logger(path: Path | None) -> logging.Logger:
    logger = logging.getLogger("temporal_spectrogram")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = ElapsedFormatter()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete(root: Path, stage: str) -> bool:
    return (root / f"{stage}.complete").is_file() and (root / f"{stage}.json").is_file()


def finish(root: Path, stage: str, payload: dict[str, Any]) -> None:
    atomic_json(root / f"{stage}.json", payload)
    (root / f"{stage}.complete").write_text("complete\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("protocol_status") != "development_only":
        raise RuntimeError("The diagnostic configuration must be development_only")
    return config


def dataset_handles(raw_root: Path):
    matlab = resolve_dataset(MATLAB_DATASET_ID, "v1", raw_root=raw_root)
    vsb = resolve_dataset(VSB_DATASET_ID, "2018-kaggle-snapshot", raw_root=raw_root)
    return matlab, vsb


def development_metadata(audit_root: Path, raw_vsb: Any) -> pd.DataFrame:
    source = pd.read_csv(audit_root / "development_metadata.csv", dtype={"sample_id": str, "id_measurement": str, "phase": str})
    raw = load_vsb_metadata(raw_vsb)
    source["signal_id"] = source["signal_id"].astype(str)
    raw["signal_id"] = raw["signal_id"].astype(str)
    result = source.merge(raw[["signal_id"]], on="signal_id", how="left", validate="one_to_one")
    if result["signal_id"].isna().any() or len(result) != 6972:
        raise RuntimeError("The forensic development metadata cannot be mapped to local VSB signals")
    return result


def stage_preflight(config: dict[str, Any], raw_root: Path, audit_root: Path, output: Path, logger: logging.Logger) -> None:
    if complete(output, "preflight"):
        logger.info("preflight already complete; reusing verified result")
        return
    started = time.perf_counter()
    if config["training"].get("epoch_selection") != "fixed_registered_epochs":
        raise RuntimeError("Repaired diagnostic requires fixed registered epoch selection")
    if config["normalization"].get("oof_standardizer") != "fit_on_outer_train_fold_only":
        raise RuntimeError("Repaired diagnostic requires fold-local OOF standardization")
    info = runtime_info()
    require_cuda()
    if not info.cuda_available:
        raise RuntimeError("CUDA is required before any diagnostic workload")
    matlab, vsb = dataset_handles(raw_root)
    train = load_mat_partition(matlab, "Tr1.mat")
    validation = load_mat_partition(matlab, "Va1.mat")
    if train.signal.shape[1] != 400 or validation.signal.shape[1] != 400:
        raise RuntimeError("MATLAB development partitions are not 400-sample signals")
    provenance = validate_modality_source_artifacts(audit_root)
    metadata = development_metadata(audit_root, vsb)
    if set(metadata["split"].astype(str)) != {"train", "validation"}:
        raise RuntimeError("VSB preflight includes a forbidden holdout split")
    if metadata.groupby("id_measurement")["phase"].nunique().ne(3).any():
        raise RuntimeError("VSB development metadata does not preserve three phases")
    if set(metadata.loc[metadata["split"] == "train", "id_measurement"]) & set(metadata.loc[metadata["split"] == "validation", "id_measurement"]):
        raise RuntimeError("VSB measurement groups cross train/validation")
    payload = {
        "status": "PASS", "elapsed_seconds": time.perf_counter() - started,
        "runtime": info.as_dict(), "raw_root_contract": "PD_RAW_DATA_ROOT/data/raw",
        "matlab": {"train_signals": len(train.signal), "validation_signals": len(validation.signal), "partitions_opened": ["Tr1.mat", "Va1.mat"]},
        "vsb": {"development_signals": len(metadata), "development_measurements": int(metadata["id_measurement"].nunique()), "holdout_rows": 0},
        "source_provenance": provenance, "holdouts": {"vsb_grouped_test": "LOCKED", "official_test": "LOCKED", "matlab_te1": "LOCKED", "matlab_te2": "LOCKED"},
    }
    atomic_json(output / "preflight.json", payload)
    finish(output, "preflight", payload)
    logger.info("preflight PASS: CUDA=%s MATLAB=%d/%d VSB development=%d", info.gpu_name, len(train.signal), len(validation.signal), len(metadata))


def stage_cache(config: dict[str, Any], raw_root: Path, audit_root: Path, output: Path, cache: Path, logger: logging.Logger) -> None:
    if complete(output, "cache"):
        logger.info("cache already complete; reusing verified arrays")
        return
    if not complete(output, "preflight"):
        raise RuntimeError("Run preflight before cache")
    required = [cache / "matlab_log_power.npy", cache / "vsb_event_log_power.npy", cache / "vsb_event_mask.npy", cache / "vsb_event_indices.npy", cache / "vsb_development_metadata.csv", cache / "cache_metadata.json"]
    if all(path.is_file() for path in required):
        metadata = json.loads((cache / "cache_metadata.json").read_text(encoding="utf-8"))
        matlab_shape = tuple(np.load(cache / "matlab_log_power.npy", mmap_mode="r").shape)
        vsb_shape = tuple(np.load(cache / "vsb_event_log_power.npy", mmap_mode="r").shape)
        mask_shape = tuple(np.load(cache / "vsb_event_mask.npy", mmap_mode="r").shape)
        if tuple(metadata.get("matlab_shape", ())) != matlab_shape or tuple(metadata.get("vsb_shape", ())) != vsb_shape or tuple(metadata.get("mask_shape", ())) != mask_shape:
            raise RuntimeError("Existing raw cache metadata does not match its arrays")
        finish(output, "cache", {"status": "PASS", "reused": True, "cache_root": str(cache), "matlab_shape": list(matlab_shape), "vsb_shape": list(vsb_shape), "mask_shape": list(mask_shape)})
        logger.info("cache PASS: reused immutable raw arrays MATLAB=%s VSB=%s", matlab_shape, vsb_shape)
        return
    started = time.perf_counter()
    matlab, vsb = dataset_handles(raw_root)
    train = load_mat_partition(matlab, "Tr1.mat")
    validation = load_mat_partition(matlab, "Va1.mat")
    matlab_values = np.concatenate([train.signal, validation.signal], axis=0)
    matlab_spec = STFTSpec(n_fft=128, win_length=64, hop_length=16)
    matlab_stft = build_matlab_spectrogram_dataset(matlab_values, matlab_spec).astype(np.float16)
    write_atomic_array_cache(cache / "matlab_log_power.npy", matlab_stft, {"dataset": "matlab", "spec": matlab_spec.as_dict(), "partitions": ["Tr1.mat", "Va1.mat"], "preprocessing": "stft-v1"})
    metadata = development_metadata(audit_root, vsb)
    frame = metadata[["sample_id", "signal_id", "id_measurement", "phase", "target", "split", "oof_fold"]].copy()
    frame.to_csv(cache / "vsb_development_metadata.csv", index=False)
    policy = PulsePolicy()
    vsb_spec = STFTSpec(n_fft=128, win_length=128, hop_length=32)
    shape = (len(frame), 2, 86, 1, vsb_spec.frequency_count(), vsb_spec.frame_count(512))
    destination = cache / "vsb_event_log_power.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    masks = np.zeros((len(frame), 2, 86), dtype=bool)
    indices = np.full((len(frame), 2, 86), -1, dtype=np.int64)
    processed = 0
    logger.info("cache VSB STFT started: signals=%d estimated_bytes=%.2f GiB", len(frame), np.prod(shape) * 2 / 1024**3)
    for batch in iter_vsb_signal_batches(vsb, frame, batch_size=4):
        bags, batch_masks, batch_indices = build_vsb_event_spectrogram_dataset(batch.signal, policy, vsb_spec, segment_length=512, capacity=86)
        values[processed:processed + len(batch.signal)] = bags.astype(np.float16)
        masks[processed:processed + len(batch.signal)] = batch_masks
        indices[processed:processed + len(batch.signal)] = batch_indices
        processed += len(batch.signal)
        if processed % 100 == 0 or processed == len(frame):
            logger.info("cache VSB STFT progress: %d/%d signals", processed, len(frame))
    values.flush()
    del values
    temporary.replace(destination)
    np.save(cache / "vsb_event_mask.npy", masks)
    np.save(cache / "vsb_event_indices.npy", indices)
    for name in ("matlab_log_power.npy", "vsb_event_log_power.npy", "vsb_event_mask.npy", "vsb_event_indices.npy"):
        (cache / f"{name}.complete").write_text("complete\n", encoding="utf-8")
    atomic_json(cache / "cache_metadata.json", {"status": "PASS", "matlab_shape": list(matlab_stft.shape), "vsb_shape": list(shape), "mask_shape": list(masks.shape), "policy": policy.as_dict(), "vsb_spec": vsb_spec.as_dict()})
    finish(output, "cache", {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "matlab_shape": list(matlab_stft.shape), "vsb_shape": list(shape), "signals": processed})
    logger.info("cache PASS: MATLAB=%s VSB=%s", matlab_stft.shape, shape)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=float), -40, 40)))


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate package."""

    if frame.empty:
        return "(no rows)"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _final_neural_predictions(train_x: Any, train_y: np.ndarray, validation_x: Any, validation_y: np.ndarray, *, kind: str, seed: int, epochs: int, batch_size: int = 4) -> np.ndarray:
    result = train_expert(
        train_x, train_y, validation_x, validation_y, kind=kind, seed=seed, epochs=epochs,
        batch_size=batch_size, pos_weight=positive_class_weight(train_y),
        train_indices=np.arange(len(train_y)), validation_indices=np.arange(len(validation_y)),
        validation_batch_size=32, num_workers=0, persistent_workers=False,
        mixed_precision=False, epoch_selection="fixed",
    )
    loader = __import__("torch").utils.data.DataLoader(NumpyDataset(validation_x, validation_y), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    return _sigmoid(predict_logits(result.model, loader))


def _pulse_validation_predictions(
    values: Any,
    mask: np.ndarray,
    labels: np.ndarray,
    seed: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    epochs: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], Any]:
    standardizer = fit_spectrogram_standardizer(values, train_indices, valid_mask=mask)
    standardized = standardizer.view(values, valid_mask=mask)
    result = train_pulse_expert(
        standardized, mask, labels, representation="spectrogram", seed=seed, epochs=epochs,
        train_indices=train_indices, validation_indices=np.empty(0, dtype=np.int64), batch_size=8,
        inference_batch_size=32, pos_weight=positive_class_weight(labels[train_indices]),
        num_workers=0, persistent_workers=False, mixed_precision=False, epoch_selection="fixed",
    )
    loader = pulse_loader(
        standardized, mask, labels, validation_indices, batch_size=32, shuffle=False,
        num_workers=0, persistent_workers=False, prefetch_factor=2,
    )
    outputs = predict_pulse_outputs(result.model, loader, mixed_precision=False)
    return np.asarray(outputs["signal_probability"], dtype=np.float64), outputs, standardizer


def stage_experts(config: dict[str, Any], raw_root: Path, audit_root: Path, output: Path, cache: Path, logger: logging.Logger) -> None:
    if complete(output, "experts"):
        logger.info("experts already complete; reusing predictions")
        return
    if not complete(output, "cache"):
        raise RuntimeError("Run cache before experts")
    started = time.perf_counter()
    matlab, vsb = dataset_handles(raw_root)
    train_mat = load_mat_partition(matlab, "Tr1.mat")
    val_mat = load_mat_partition(matlab, "Va1.mat")
    mat_folds = fold_assignments(train_mat.label, n_splits=5, seed=42)
    temporal_epochs = int(config["training"].get("matlab_temporal_epochs", 5))
    matlab_spec_epochs = int(config["training"]["matlab_spectrogram"].get("epochs", 5))
    vsb_spec_epochs = int(config["training"]["vsb_spectrogram"].get("epochs", 7))
    mat_raw_stft = np.load(cache / "matlab_log_power.npy", mmap_mode="r")
    mat_raw_train = mat_raw_stft[:len(train_mat.signal)]
    mat_raw_val = mat_raw_stft[len(train_mat.signal):]
    standardizer_records: list[dict[str, Any]] = []
    current_seed: int | None = None

    def mat_temporal_fold(raw: Any, fit: np.ndarray, holdout: np.ndarray, fold: int) -> np.ndarray:
        standardizer = fit_standardizer(np.asarray(raw)[fit], axis=0)
        standardizer_records.append({"dataset": "matlab", "representation": "temporal", "scope": "oof_outer_train", "seed": current_seed, "fold": fold, "fit_count": int(len(fit)), "fingerprint": _standardizer_fingerprint(standardizer)})
        return temporal_input(np.asarray(raw), standardizer)

    def mat_spectrogram_fold(raw: Any, fit: np.ndarray, holdout: np.ndarray, fold: int) -> Any:
        standardizer = fit_spectrogram_standardizer(raw, fit)
        standardizer_records.append({"dataset": "matlab", "representation": "spectrogram", "scope": "oof_outer_train", "seed": current_seed, "fold": fold, "fit_count": standardizer.fit_count, "fingerprint": standardizer.fingerprint})
        return standardizer.view(raw)

    final_temporal_standardizer = fit_standardizer(train_mat.signal, axis=0)
    mat_temporal_train = temporal_input(train_mat.signal, final_temporal_standardizer)
    mat_temporal_val = temporal_input(val_mat.signal, final_temporal_standardizer)
    final_matlab_spec_standardizer = fit_spectrogram_standardizer(mat_raw_stft, np.arange(len(train_mat.signal)))
    mat_spec_train = final_matlab_spec_standardizer.transform(mat_raw_train)
    mat_spec_val = final_matlab_spec_standardizer.transform(mat_raw_val)
    standardizer_records.extend([
        {"dataset": "matlab", "representation": "temporal", "scope": "final_train", "seed": None, "fit_count": int(len(train_mat.signal)), "fingerprint": _standardizer_fingerprint(final_temporal_standardizer)},
        {"dataset": "matlab", "representation": "spectrogram", "scope": "final_train", "seed": None, "fit_count": final_matlab_spec_standardizer.fit_count, "fingerprint": final_matlab_spec_standardizer.fingerprint},
    ])
    rows: list[pd.DataFrame] = []
    aggregation_rows: list[pd.DataFrame] = []
    for seed in config["scope"]["seeds"]:
        current_seed = int(seed)
        logger.info("MATLAB seed %d temporal OOF", seed)
        temporal_oof = cross_fitted_logits(
            train_mat.signal, train_mat.label, kind="temporal", folds=mat_folds, seed=int(seed),
            epochs=temporal_epochs, batch_size=4, validation_batch_size=32, num_workers=0,
            persistent_workers=False, mixed_precision=False, fold_transform=mat_temporal_fold,
            epoch_selection="fixed",
        ).logits
        spec_oof = cross_fitted_logits(
            mat_raw_train, train_mat.label, kind="spectrogram", folds=mat_folds, seed=int(seed),
            epochs=matlab_spec_epochs, batch_size=4, validation_batch_size=32, num_workers=0,
            persistent_workers=False, mixed_precision=False, fold_transform=mat_spectrogram_fold,
            epoch_selection="fixed",
        ).logits
        temporal_threshold = select_threshold(train_mat.label, _sigmoid(temporal_oof))
        spec_threshold = select_threshold(train_mat.label, _sigmoid(spec_oof))
        temporal_val = _final_neural_predictions(mat_temporal_train, train_mat.label, mat_temporal_val, val_mat.label, kind="temporal", seed=int(seed), epochs=temporal_epochs)
        spec_val = _final_neural_predictions(mat_spec_train, train_mat.label, mat_spec_val, val_mat.label, kind="spectrogram", seed=int(seed), epochs=matlab_spec_epochs)
        rows.extend([
            pd.DataFrame({"dataset": "matlab", "split": "train_oof", "sample_id": train_mat.sample_id, "id_measurement": train_mat.sample_id, "phase": "NA", "target": train_mat.label, "method": "temporal", "seed": seed, "probability": _sigmoid(temporal_oof), "prediction": (_sigmoid(temporal_oof) >= temporal_threshold).astype(int), "threshold": temporal_threshold}),
            pd.DataFrame({"dataset": "matlab", "split": "validation", "sample_id": val_mat.sample_id, "id_measurement": val_mat.sample_id, "phase": "NA", "target": val_mat.label, "method": "temporal", "seed": seed, "probability": temporal_val, "prediction": (temporal_val >= temporal_threshold).astype(int), "threshold": temporal_threshold}),
            pd.DataFrame({"dataset": "matlab", "split": "train_oof", "sample_id": train_mat.sample_id, "id_measurement": train_mat.sample_id, "phase": "NA", "target": train_mat.label, "method": "spectrogram", "seed": seed, "probability": _sigmoid(spec_oof), "prediction": (_sigmoid(spec_oof) >= spec_threshold).astype(int), "threshold": spec_threshold}),
            pd.DataFrame({"dataset": "matlab", "split": "validation", "sample_id": val_mat.sample_id, "id_measurement": val_mat.sample_id, "phase": "NA", "target": val_mat.label, "method": "spectrogram", "seed": seed, "probability": spec_val, "prediction": (spec_val >= spec_threshold).astype(int), "threshold": spec_threshold}),
        ])
        logger.info("MATLAB seed %d experts ready", seed)
    frame, canonical = prepare_forensic_baseline_frame(audit_root)
    inventory = build_modality_feature_inventory(canonical)
    temporal_columns = feature_set_columns(inventory, "S+T", canonical)
    temporal_rows: list[pd.DataFrame] = []
    metadata = pd.read_csv(cache / "vsb_development_metadata.csv", dtype={"sample_id": str, "id_measurement": str, "phase": str})
    for seed in config["scope"]["seeds"]:
        result = evaluate_feature_baseline_with_predictions(frame, temporal_columns, classifier="hist_gradient_boosting", seed=int(seed), grouped=True, mode="phase_independent")
        oof = result["train_oof_predictions"].copy(); val = result["validation_predictions"].copy()
        for current, split in ((oof, "train_oof"), (val, "validation")):
            current["dataset"] = "vsb"; current["split"] = split; current["method"] = "temporal"; current["seed"] = seed; current["threshold"] = result["threshold_from_train_oof"]
            temporal_rows.append(current)
        logger.info("VSB seed %d temporal HGB MCC=%.4f threshold=%.3f", seed, result["metrics"]["mcc"], result["threshold_from_train_oof"])
    # The neural VSB spectrogram stage uses the same parent-signal loss and mask
    # as the registered pulse trainer. It is intentionally separate from HGB.
    vsb_values = np.load(cache / "vsb_event_log_power.npy", mmap_mode="r")
    vsb_mask = np.load(cache / "vsb_event_mask.npy", mmap_mode="r")
    train_index = np.flatnonzero(metadata["split"].to_numpy() == "train")
    val_index = np.flatnonzero(metadata["split"].to_numpy() == "validation")
    groups = metadata["id_measurement"].astype(str).to_numpy()
    labels = metadata["target"].to_numpy(np.int64)
    folds = metadata["oof_fold"].to_numpy(np.int64)
    vsb_train_values = IndexedArrayView(vsb_values, train_index)
    vsb_train_mask = IndexedArrayView(vsb_mask, train_index)
    cached_indices = np.asarray(np.load(cache / "vsb_event_indices.npy", mmap_mode="r"))
    localization = metadata[["sample_id", "id_measurement", "phase", "target", "split"]].copy()
    localization["valid_event_count_half_0"] = np.asarray(vsb_mask[:, 0].sum(axis=1), dtype=np.int64)
    localization["valid_event_count_half_1"] = np.asarray(vsb_mask[:, 1].sum(axis=1), dtype=np.int64)
    localization["event_index_min"] = np.where(vsb_mask, cached_indices, np.iinfo(np.int64).max).min(axis=(1, 2))
    localization["event_index_max"] = np.where(vsb_mask, cached_indices, -1).max(axis=(1, 2))
    localization["boundary_rejections"] = np.nan
    localization.to_csv(output / "vsb_event_localization.csv", index=False)

    def vsb_spectrogram_fold(raw: Any, fit: np.ndarray, holdout: np.ndarray, fold: int) -> tuple[Any, Any]:
        standardizer = fit_spectrogram_standardizer(raw, fit, valid_mask=vsb_train_mask)
        standardizer_records.append({"dataset": "vsb", "representation": "spectrogram", "scope": "oof_outer_train", "seed": current_seed, "fold": fold, "fit_count": standardizer.fit_count, "fingerprint": standardizer.fingerprint})
        return standardizer.view(raw, valid_mask=vsb_train_mask), vsb_train_mask

    for seed in config["scope"]["seeds"]:
        current_seed = int(seed)
        logger.info("VSB seed %d spectrogram OOF", seed)
        train_labels = labels[train_index]
        train_groups = groups[train_index]
        train_folds = folds[train_index]
        oof_result = cross_fitted_pulse_logits(
            vsb_train_values, vsb_train_mask, train_labels, representation="spectrogram",
            folds=train_folds, seed=int(seed), groups=train_groups, epochs=vsb_spec_epochs,
            batch_size=8, inference_batch_size=32, num_workers=0, persistent_workers=False,
            mixed_precision=False, fold_transform=vsb_spectrogram_fold, epoch_selection="fixed",
        )
        oof = oof_result.logits
        probability_oof = _sigmoid(oof)
        threshold = select_threshold(train_labels, probability_oof)
        validation_probability, validation_aggregation, final_vsb_standardizer = _pulse_validation_predictions(
            vsb_values, vsb_mask, labels, int(seed), train_index, val_index, epochs=vsb_spec_epochs,
        )
        standardizer_records.append({"dataset": "vsb", "representation": "spectrogram", "scope": "final_train", "seed": int(seed), "fit_count": final_vsb_standardizer.fit_count, "fingerprint": final_vsb_standardizer.fingerprint})
        current_oof = metadata.iloc[train_index][["sample_id", "id_measurement", "phase", "target"]].copy()
        current_oof["probability"] = probability_oof; current_oof["prediction"] = (current_oof["probability"] >= threshold).astype(int); current_oof["dataset"] = "vsb"; current_oof["split"] = "train_oof"; current_oof["method"] = "spectrogram"; current_oof["seed"] = seed; current_oof["threshold"] = threshold
        current_val = metadata.iloc[val_index][["sample_id", "id_measurement", "phase", "target"]].copy()
        current_val["probability"] = validation_probability; current_val["prediction"] = (validation_probability >= threshold).astype(int); current_val["dataset"] = "vsb"; current_val["split"] = "validation"; current_val["method"] = "spectrogram"; current_val["seed"] = seed; current_val["threshold"] = threshold
        rows.extend([current_oof, current_val])
        if oof_result.aggregation_summaries:
            oof_aggregation = pd.DataFrame({"sample_id": metadata.iloc[train_index]["sample_id"].to_numpy(), **{name: values for name, values in oof_result.aggregation_summaries.items()}})
            oof_aggregation["dataset"], oof_aggregation["split"], oof_aggregation["seed"] = "vsb", "train_oof", seed
            aggregation_rows.append(oof_aggregation)
        val_aggregation = pd.DataFrame({"sample_id": metadata.iloc[val_index]["sample_id"].to_numpy(), **validation_aggregation})
        val_aggregation["dataset"], val_aggregation["split"], val_aggregation["seed"] = "vsb", "validation", seed
        aggregation_rows.append(val_aggregation)
        logger.info("VSB seed %d spectrogram ready", seed)
    rows.extend(temporal_rows)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    pd.DataFrame(standardizer_records).to_json(output / "standardizer_fingerprints.json", orient="records", indent=2)
    if aggregation_rows:
        pd.concat(aggregation_rows, ignore_index=True).to_parquet(output / "vsb_aggregation_diagnostics.parquet", index=False)
    finish(output, "experts", {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "prediction_rows": len(predictions), "datasets": ["matlab", "vsb"], "seeds": config["scope"]["seeds"]})


def stage_evaluation(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "evaluation"):
        prediction_path = output / "validation_predictions.parquet"
        if prediction_path.is_file():
            existing = pd.read_parquet(prediction_path)
            if "prediction" in existing.columns and not existing["prediction"].isna().any():
                logger.info("evaluation already complete; reusing report tables")
                return
        logger.info("existing evaluation marker is stale; repairing predictions and recomputing evaluation")
    if not complete(output, "experts"):
        raise RuntimeError("Run experts before evaluation")
    started = time.perf_counter()
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    if "prediction" not in predictions.columns:
        predictions["prediction"] = np.nan
    missing_prediction = predictions["prediction"].isna()
    if missing_prediction.any():
        predictions.loc[missing_prediction, "prediction"] = (
            predictions.loc[missing_prediction, "probability"]
            >= predictions.loc[missing_prediction, "threshold"]
        ).astype(np.int64)
        predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    if not np.isfinite(predictions["probability"].to_numpy(dtype=float)).all():
        raise RuntimeError("Validation predictions contain non-finite probabilities")
    if not predictions["prediction"].isin([0, 1]).all():
        raise RuntimeError("Validation predictions must be binary after thresholding")
    metric_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    fixed_metric_rows: list[dict[str, Any]] = []
    fixed_prediction_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    hierarchical_records: dict[tuple[str, str], list[dict[str, np.ndarray]]] = {}
    weights = tuple(float(value) for value in config["diagnostics"]["fixed_weights"])
    for (dataset, seed), group in predictions.groupby(["dataset", "seed"], sort=True):
        seed = int(seed)
        validation = group[group["split"] == "validation"]
        oof = group[group["split"] == "train_oof"]
        for split_name, current in (("validation", validation), ("train_oof", oof)):
            for method, method_rows in current.groupby("method", sort=True):
                probability = method_rows["probability"].to_numpy(dtype=float)
                threshold = float(method_rows["threshold"].iloc[0])
                summary = probability_summary(probability)
                probability_rows.append({"dataset": dataset, "seed": seed, "method": method, "split": split_name, **summary})
                curve_rows.extend({"dataset": dataset, "seed": seed, "method": method, "split": split_name, **row} for row in threshold_curve(method_rows["target"].to_numpy(), probability))
        temporal = validation[validation["method"] == "temporal"].sort_values("sample_id")
        spectrogram = validation[validation["method"] == "spectrogram"].sort_values("sample_id")
        temporal_oof = oof[oof["method"] == "temporal"].sort_values("sample_id")
        spectrogram_oof = oof[oof["method"] == "spectrogram"].sort_values("sample_id")
        if len(temporal) == 0 or temporal["sample_id"].tolist() != spectrogram["sample_id"].tolist() or temporal_oof["sample_id"].tolist() != spectrogram_oof["sample_id"].tolist():
            raise RuntimeError(f"Prediction alignment failed for {dataset} seed {seed}")
        for method, current in (("temporal", temporal), ("spectrogram", spectrogram)):
            metric_rows.append({"dataset": dataset, "seed": seed, "method": method, "split": "validation", **binary_metrics(current["target"].to_numpy(), current["probability"].to_numpy(), float(current["threshold"].iloc[0]))})
        oracle = prediction_overlap_oracle(temporal["target"].to_numpy(), temporal["prediction"].to_numpy(), spectrogram["prediction"].to_numpy(), temporal["id_measurement"].to_numpy())
        overlap_rows.append({"dataset": dataset, "seed": seed, **{key: value for key, value in oracle.items() if key != "rows"}})
        for row in oracle["rows"]:
            overlap_rows.append({"dataset": dataset, "seed": seed, **row})
        mixture_rows, best_fixed = select_fixed_weight_oof(
            temporal["target"].to_numpy(), temporal["probability"].to_numpy(), spectrogram["probability"].to_numpy(),
            temporal_oof["target"].to_numpy(), temporal_oof["probability"].to_numpy(), spectrogram_oof["probability"].to_numpy(), weights,
        )
        mixture_frame = pd.DataFrame(mixture_rows)
        mixture_frame["dataset"], mixture_frame["seed"] = dataset, seed
        mixture_frame.to_csv(output / f"fixed_mixture_{dataset}_{seed}.csv", index=False)
        for label, selected in (("50_50", next(row for row in mixture_rows if abs(row["temporal_weight"] - 0.5) < 1e-12)), ("best_fixed", best_fixed)):
            probability = selected["temporal_weight"] * temporal["probability"].to_numpy() + (1.0 - selected["temporal_weight"]) * spectrogram["probability"].to_numpy()
            hard = (probability >= selected["threshold"]).astype(np.int64)
            metrics = binary_metrics(temporal["target"].to_numpy(), probability, selected["threshold"])
            fixed_metric_rows.append({"dataset": dataset, "seed": seed, "method": label, "split": "validation", "temporal_weight": selected["temporal_weight"], **metrics})
            fixed_prediction_rows.append(pd.DataFrame({"dataset": dataset, "seed": seed, "method": label, "split": "validation", "sample_id": temporal["sample_id"].to_numpy(), "id_measurement": temporal["id_measurement"].to_numpy(), "phase": temporal["phase"].to_numpy(), "target": temporal["target"].to_numpy(), "probability": probability, "prediction": hard, "threshold": selected["threshold"]}))
        oof_mcc = {method: float(binary_metrics(current["target"].to_numpy(), current["probability"].to_numpy(), float(current["threshold"].iloc[0]))["mcc"]) for method, current in (("temporal", temporal_oof), ("spectrogram", spectrogram_oof))}
        best_individual = "temporal" if (oof_mcc["temporal"], 1) >= (oof_mcc["spectrogram"], 0) else "spectrogram"
        best_individual_prediction = temporal["prediction"].to_numpy() if best_individual == "temporal" else spectrogram["prediction"].to_numpy()
        fixed_by_method = {frame["method"].iloc[0]: frame for frame in fixed_prediction_rows if frame["dataset"].iloc[0] == dataset and int(frame["seed"].iloc[0]) == seed}
        comparisons = {
            "spectrogram_minus_temporal": (spectrogram["prediction"].to_numpy(), temporal["prediction"].to_numpy()),
            "50_50_minus_best_individual": (fixed_by_method["50_50"]["prediction"].to_numpy(), best_individual_prediction),
            "best_fixed_minus_best_individual": (fixed_by_method["best_fixed"]["prediction"].to_numpy(), best_individual_prediction),
        }
        for comparison, (prediction_a, prediction_b) in comparisons.items():
            unit = temporal["id_measurement"].to_numpy() if dataset == "vsb" else temporal["sample_id"].to_numpy()
            result = grouped_paired_bootstrap_delta(temporal["target"].to_numpy(), prediction_a, prediction_b, unit, iterations=int(config["statistics"]["paired_bootstrap_iterations"]), seed=42042 + seed) if dataset == "vsb" else paired_bootstrap_delta(temporal["target"].to_numpy(), prediction_a, prediction_b, iterations=int(config["statistics"]["paired_bootstrap_iterations"]), seed=42042 + seed)
            bootstrap_rows.append({"dataset": dataset, "seed": seed, "comparison": comparison, "bootstrap_unit": "id_measurement" if dataset == "vsb" else "signal", **result})
            hierarchical_records.setdefault((dataset, comparison), []).append({"labels": temporal["target"].to_numpy(), "prediction_a": prediction_a, "prediction_b": prediction_b, "group_ids": unit})
    metrics = pd.concat([pd.DataFrame(metric_rows), pd.DataFrame(fixed_metric_rows)], ignore_index=True)
    overlaps = pd.DataFrame(overlap_rows)
    pd.DataFrame(curve_rows).to_csv(output / "threshold_curves.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(output / "probability_diagnostics.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    overlaps.to_csv(output / "prediction_overlap.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "bootstrap_deltas.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_json(output / "bootstrap_deltas.json", orient="records", indent=2)
    pd.concat(fixed_prediction_rows, ignore_index=True).to_parquet(output / "fixed_mixture_predictions.parquet", index=False)
    hierarchical_rows = []
    for (dataset, comparison), records in hierarchical_records.items():
        if dataset == "vsb":
            result = hierarchical_grouped_delta_ci(records, comparison, iterations=int(config["statistics"]["paired_bootstrap_iterations"]), seed=int(config["statistics"].get("bootstrap_seed", 42042)))
        else:
            result = hierarchical_paired_delta_ci(records, comparison, iterations=int(config["statistics"]["paired_bootstrap_iterations"]), seed=int(config["statistics"].get("bootstrap_seed", 42042)))
        hierarchical_rows.append({"dataset": dataset, **result})
    atomic_json(output / "hierarchical_bootstrap.json", hierarchical_rows)
    pd.DataFrame(hierarchical_rows).to_csv(output / "hierarchical_bootstrap.csv", index=False)
    regression_rows: list[dict[str, Any]] = []
    for dataset, key in (("vsb", "vsb_temporal"), ("matlab", "matlab_temporal")):
        expected = config["regression"][key]["expected"]
        tolerance_mcc = float(config["regression"][key]["tolerance_mcc"])
        tolerance_threshold = float(config["regression"][key]["tolerance_threshold"])
        observed = metrics[(metrics["dataset"] == dataset) & (metrics["method"] == "temporal")].sort_values("seed")
        for seed, expected_value in expected.items():
            seed = int(seed)
            row = observed[observed["seed"] == seed]
            observed_mcc = float(row["mcc"].iloc[0]) if len(row) else float("nan")
            if isinstance(expected_value, dict):
                target_mcc, target_threshold = float(expected_value["mcc"]), float(expected_value["threshold"])
            else:
                target_mcc, target_threshold = float(expected_value), None
            observed_threshold = float(row["threshold"].iloc[0]) if len(row) else float("nan")
            status = abs(observed_mcc - target_mcc) <= tolerance_mcc and (target_threshold is None or abs(observed_threshold - target_threshold) <= tolerance_threshold)
            regression_rows.append({"dataset": dataset, "seed": seed, "observed_mcc": observed_mcc, "expected_mcc": target_mcc, "observed_threshold": observed_threshold, "expected_threshold": target_threshold, "tolerance_mcc": tolerance_mcc, "tolerance_threshold": tolerance_threshold, "status": "PASS" if status else "FAIL"})
    regression_ok = all(row["status"] == "PASS" for row in regression_rows)
    atomic_json(output / "regression_checks.json", {"status": "PASS" if regression_ok else "FAIL", "checks": regression_rows})
    summary_payload = {"status": "PASS", "regression_ok": regression_ok, "metrics": metrics.to_dict(orient="records"), "overlap": overlap_rows, "bootstrap": bootstrap_rows, "hierarchical_bootstrap": hierarchical_rows, "regression": regression_rows}
    atomic_json(output / "evaluation_summary.json", summary_payload)
    finish(output, "evaluation", {"status": "PASS", "regression_ok": regression_ok, "elapsed_seconds": time.perf_counter() - started, "metric_rows": len(metrics), "overlap_rows": len(overlaps)})
    logger.info("evaluation PASS: metrics=%d overlap=%d regression=%s", len(metrics), len(overlaps), regression_ok)


def stage_report(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "report"):
        logger.info("report already complete; reusing curated report")
        return
    if not complete(output, "evaluation"):
        raise RuntimeError("Run evaluation before report")
    metrics = pd.read_csv(output / "metrics.csv")
    overlaps = pd.read_csv(output / "prediction_overlap.csv")
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    regression = json.loads((output / "regression_checks.json").read_text(encoding="utf-8"))
    hierarchical = json.loads((output / "hierarchical_bootstrap.json").read_text(encoding="utf-8"))
    summary = {
        "pipeline_valid": preflight.get("status") == "PASS" and regression.get("status") == "PASS",
        "integrity_ok": preflight.get("status") == "PASS",
        "regression_ok": regression.get("status") == "PASS",
        "spectrogram_evidence": bool(len(metrics)),
        "vsb_spectrogram_strong": False,
        "matlab_spectrogram_strong": False,
        "complementarity_supported": False,
    }
    for dataset in ("matlab", "vsb"):
        values = metrics[(metrics.dataset == dataset) & (metrics.method == "spectrogram")]["mcc"]
        if len(values) == 3:
            if dataset == "vsb": summary["vsb_spectrogram_strong"] = float(values.mean()) >= 0.60 and float(values.min()) > 0.55
            else: summary["matlab_spectrogram_strong"] = float(values.mean()) >= 0.95 and float(values.min()) >= 0.93
    if "oracle_headroom" in overlaps:
        top_level = overlaps[overlaps.get("stratum", pd.Series(index=overlaps.index, dtype=object)).isna()]
        headroom = top_level["oracle_headroom"].to_numpy(dtype=float)
        ci_rows = [row for row in hierarchical if row.get("comparison") == "spectrogram_minus_temporal"]
        summary["complementarity_supported"] = bool(
            len(headroom) == 6 and float(headroom.mean()) >= float(config["verdict"]["meaningful_oracle_headroom"])
            and int(np.sum(headroom >= float(config["verdict"]["minimum_headroom_per_seed"]))) >= int(config["verdict"]["minimum_headroom_seeds"])
            and all(row.get("ci_95", [float("nan")])[0] > float(config["verdict"]["aggregate_ci_lower_bound"]) for row in ci_rows)
        )
    summary["verdict"] = classify_cross_dataset_verdict(summary)
    report = [
        "# Cross-Dataset Temporal vs Spectrogram Diagnostic (Repaired Run)", "",
        f"**Pipeline valid:** `{summary['pipeline_valid']}`  ", f"**Scientific verdict:** `{summary['verdict']}`", "",
        "This repaired run is separate from the prior provisional report. Raw caches are reused without modification; repaired predictions and reports use a new versioned output root.", "",
        "## OBSERVED", "",
        "This development-only report uses MATLAB Tr1/Va1 and VSB grouped development identities. VSB grouped test, official test, MATLAB Te1/Te2, CWT reruns, adaptive fusion, reliability models, and external weights remain locked.", "",
        "### Temporal regression gates", "", _markdown_table(pd.DataFrame(regression.get("checks", []))), "",
        "### Validation metrics", "", _markdown_table(metrics), "",
        "### Prediction overlap and oracle", "", _markdown_table(overlaps), "",
        "### Hierarchical paired bootstrap", "", _markdown_table(pd.DataFrame(hierarchical)), "",
        "### Additional diagnostics", "",
        "- `threshold_curves.csv`: registered threshold grid for OOF and validation; selection is OOF-only.",
        "- `probability_diagnostics.csv`: probability ranges and quantiles.",
        "- `fixed_mixture_predictions.parquet`: diagnostic-only 50/50 and OOF-selected best-fixed mixtures.",
        "- `vsb_aggregation_diagnostics.parquet`: event counts, top-k counts, event probabilities, half-cycle probabilities, and parent probabilities.", "",
        "## INTERPRETATION", "",
        "Pipeline validity and scientific verdict are reported separately. A failed temporal regression gate forces `INCONCLUSIVE`. Fixed mixtures are diagnostic-only and do not create an adaptive-fusion pipeline.", "",
        "## UNRESOLVED", "",
        "Dual-CyCon frequency concepts are verified only at high level. Erişti and exact reported literature values remain PROJECT LITERATURE ANCHOR — NOT REVERIFIED. Localization summaries are limited to information present in the immutable event cache.",
    ]
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(ROOT / config["outputs"]["summary"], {"diagnostic": "Cross-Dataset Temporal vs Spectrogram", **summary, "regression": regression, "metrics": metrics.to_dict(orient="records"), "overlap": overlaps.to_dict(orient="records"), "hierarchical_bootstrap": hierarchical})
    manifest = ROOT / config["outputs"]["manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True); manifest.write_text(f"# Diagnostic Manifest\n\n- Pipeline valid: `{summary['pipeline_valid']}`\n- Scientific verdict: `{summary['verdict']}`\n- Generated rows: {len(metrics)} metrics, {len(overlaps)} overlap\n- Temporal regression: `{regression.get('status')}`\n- Holdouts: locked\n- Data contract: `PD_RAW_DATA_ROOT/data/raw`\n- Raw cache: reused and unnormalized\n", encoding="utf-8")
    finish(output, "report", {"status": "PASS", "verdict": summary["verdict"], "metrics": len(metrics), "overlap": len(overlaps)})
    logger.info("report PASS: verdict=%s", summary["verdict"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=Path("results/audits/vsb-literature-forensic"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--stage", choices=("preflight", "cache", "experts", "evaluation", "report", "all"), default="preflight")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    raw_root = raw_data_root(args.raw_root)
    output = (args.output_root or ROOT / config["outputs"]["audit_root"]).resolve()
    audit_root = args.audit_root.resolve()
    cache = (args.cache_root or ROOT / config["outputs"]["cache_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    logger = make_logger(args.log_file)
    stages = [args.stage] if args.stage != "all" else ["preflight", "cache", "experts", "evaluation", "report"]
    for stage in stages:
        logger.info("stage %s started", stage)
        if stage == "preflight": stage_preflight(config, raw_root, audit_root, output, logger)
        elif stage == "cache": stage_cache(config, raw_root, audit_root, output, cache, logger)
        elif stage == "experts": stage_experts(config, raw_root, audit_root, output, cache, logger)
        elif stage == "evaluation": stage_evaluation(config, output, logger)
        elif stage == "report": stage_report(config, output, logger)
        logger.info("stage %s finished", stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
