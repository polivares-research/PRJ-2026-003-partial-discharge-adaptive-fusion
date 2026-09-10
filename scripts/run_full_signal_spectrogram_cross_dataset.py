#!/usr/bin/env python3
"""Resumable development-only full-signal MATLAB/VSB spectrogram diagnostic.

The VSB branch in this runner reads each complete 800,000-sample signal once,
builds one global STFT, and trains one parent-signal classifier. The prior
local-event/MIL diagnostic remains a separate historical runner and is never
called here.
"""

from __future__ import annotations

import argparse
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

from partial_discharge_adaptive_fusion.config import require_cuda, runtime_info  # noqa: E402
from partial_discharge_adaptive_fusion.dataset import (  # noqa: E402
    iter_vsb_signal_batches,
    load_mat_partition,
    resolve_dataset,
)
from partial_discharge_adaptive_fusion.evaluation import (  # noqa: E402
    binary_metrics,
    grouped_paired_bootstrap_delta,
    hierarchical_grouped_delta_ci,
    hierarchical_paired_delta_ci,
    paired_bootstrap_delta,
)
from partial_discharge_adaptive_fusion.fusion import select_threshold  # noqa: E402
from partial_discharge_adaptive_fusion.full_signal_spectrogram import (  # noqa: E402
    FullSignalSTFTSpec,
    FrequencyStandardizedView,
    cache_fingerprint,
    compute_full_signal_log_power,
    fit_frequency_standardizer,
    require_global_spectrogram_cache,
    write_global_spectrogram_cache,
)
from partial_discharge_adaptive_fusion.modeling.data import NumpyDataset  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.train import (  # noqa: E402
    cross_fitted_logits,
    fold_assignments,
    positive_class_weight,
    predict_logits,
    train_expert,
)
from partial_discharge_adaptive_fusion.representations import fit_standardizer, temporal_input  # noqa: E402
from partial_discharge_adaptive_fusion.spectrogram import (  # noqa: E402
    IndexedArrayView,
    STFTSpec,
    build_matlab_spectrogram_dataset,
    fit_spectrogram_standardizer,
)
from partial_discharge_adaptive_fusion.temporal_spectrogram import (  # noqa: E402
    prediction_overlap_oracle,
    select_fixed_weight_oof,
)
from partial_discharge_adaptive_fusion.vsb_forensics import (  # noqa: E402
    evaluate_feature_baseline_with_predictions,
    prepare_forensic_baseline_frame,
)
from partial_discharge_adaptive_fusion.vsb_modality import (  # noqa: E402
    build_modality_feature_inventory,
    feature_set_columns,
    validate_modality_source_artifacts,
)


START = time.perf_counter()


class ElapsedFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        return f"[{stamp}] [elapsed={time.perf_counter() - START:010.1f}s] {record.getMessage()}"


def make_logger(path: Path | None) -> logging.Logger:
    logger = logging.getLogger("full_signal_spectrogram")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = ElapsedFormatter()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
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
        raise RuntimeError("The full-signal diagnostic requires protocol_status=development_only")
    if config.get("data_contract", {}).get("researchdata") != "forbidden":
        raise RuntimeError("researchdata must remain forbidden")
    return config


def _datasets(raw_root: Path):
    matlab = resolve_dataset("engineering-partial-discharge-noise-signals", "v1", raw_root=raw_root)
    vsb = resolve_dataset("engineering-vsb-power-line-fault-detection", "2018-kaggle-snapshot", raw_root=raw_root)
    return matlab, vsb


def _development_metadata(audit_root: Path, vsb: Any) -> pd.DataFrame:
    source = pd.read_csv(
        audit_root / "development_metadata.csv",
        dtype={"sample_id": str, "id_measurement": str, "phase": str},
    )
    if set(source["split"].astype(str)) != {"train", "validation"}:
        raise RuntimeError("Full-signal diagnostic cannot read VSB holdout rows")
    raw = pd.read_csv(vsb.handle.path("vsb-metadata-train"), dtype={"signal_id": str})
    raw["signal_id"] = raw["signal_id"].astype(str)
    source["signal_id"] = source["signal_id"].astype(str)
    result = source.merge(raw[["signal_id"]], on="signal_id", how="left", validate="one_to_one")
    if result["signal_id"].isna().any() or len(result) != 6972:
        raise RuntimeError("Forensic development metadata cannot be mapped to local VSB metadata")
    return result


def _vsb_spec(config: dict[str, Any]) -> FullSignalSTFTSpec:
    values = config["representation"]["vsb"]
    spec = FullSignalSTFTSpec(
        sampling_frequency_hz=float(values["sampling_frequency_hz"]),
        signal_length=int(values["signal_length"]), n_fft=int(values["n_fft"]),
        win_length=int(values["win_length"]), hop_length=int(values["hop_length"]),
        epsilon=float(values["epsilon"]), f_min_hz=float(values["f_min_hz"]),
    )
    expected = tuple(int(value) for value in values["expected_shape"])
    if spec.expected_shape != expected:
        raise RuntimeError(f"Registered VSB shape {expected} disagrees with analytical shape {spec.expected_shape}")
    return spec


def _matlab_spec() -> STFTSpec:
    return STFTSpec(n_fft=128, win_length=64, hop_length=16)


def _cache_contract(config: dict[str, Any], dataset: str, shape: tuple[int, ...]) -> dict[str, Any]:
    if dataset == "vsb":
        return _vsb_spec(config).cache_metadata(dataset="vsb", dataset_version="2018-kaggle-snapshot") | {"shape": list(shape)}
    spec = _matlab_spec()
    return {
        "dataset": "matlab", "dataset_version": "v1", "representation": spec.as_dict(),
        "partitions": ["Tr1.mat", "Va1.mat"], "normalized": False,
        "dtype": "float16", "shape": list(shape), "preprocessing_version": "matlab-global-stft-v1",
    }


def _expected_cache_fingerprint(path: Path) -> str:
    metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    return metadata["fingerprint"]


def stage_preflight(config: dict[str, Any], raw_root: Path, audit_root: Path, output: Path, logger: logging.Logger) -> None:
    if complete(output, "preflight"):
        logger.info("preflight already complete; reusing verified result")
        return
    started = time.perf_counter()
    info = runtime_info()
    require_cuda()
    if not info.cuda_available:
        raise RuntimeError("CUDA is required before any diagnostic workload")
    spec = _vsb_spec(config)
    matlab, vsb = _datasets(raw_root)
    train = load_mat_partition(matlab, "Tr1.mat")
    validation = load_mat_partition(matlab, "Va1.mat")
    if train.signal.shape[1] != 400 or validation.signal.shape[1] != 400:
        raise RuntimeError("MATLAB Tr1/Va1 are not 400-sample signals")
    provenance = validate_modality_source_artifacts(audit_root)
    metadata = _development_metadata(audit_root, vsb)
    if metadata.groupby("id_measurement").size().ne(3).any() or metadata.groupby("id_measurement")["phase"].nunique().ne(3).any():
        raise RuntimeError("VSB development identities do not contain exactly three phases")
    train_groups = set(metadata.loc[metadata["split"] == "train", "id_measurement"])
    validation_groups = set(metadata.loc[metadata["split"] == "validation", "id_measurement"])
    if train_groups & validation_groups:
        raise RuntimeError("VSB id_measurement groups cross train/validation")
    # A small native read proves signal length and finiteness without starting cache generation.
    sample = next(iter_vsb_signal_batches(vsb, metadata.head(1), batch_size=1))
    if sample.signal.shape != (1, spec.signal_length) or not np.isfinite(sample.signal).all():
        raise RuntimeError(f"VSB native signal contract failed: {sample.signal.shape}")
    payload = {
        "status": "PASS", "elapsed_seconds": time.perf_counter() - started,
        "runtime": info.as_dict(), "data_contract": "PD_RAW_DATA_ROOT/data/raw",
        "matlab": {"train_signals": len(train.signal), "validation_signals": len(validation.signal), "partitions_opened": ["Tr1.mat", "Va1.mat"]},
        "vsb": {"development_signals": len(metadata), "development_measurements": int(metadata["id_measurement"].nunique()), "native_signal_length": int(sample.signal.shape[1]), "sampling_frequency_hz": spec.sampling_frequency_hz, "holdout_rows": 0},
        "global_vsb_spec": spec.as_dict(), "source_provenance": provenance,
        "holdouts": {"vsb_grouped_test": "LOCKED", "vsb_official_test": "LOCKED", "matlab_te1": "LOCKED", "matlab_te2": "LOCKED"},
    }
    atomic_json(output / "preflight.json", payload)
    finish(output, "preflight", payload)
    logger.info("preflight PASS: CUDA=%s VSB=%d signals shape=%s", info.gpu_name, len(metadata), spec.expected_shape)


def stage_cache(config: dict[str, Any], raw_root: Path, audit_root: Path, output: Path, cache: Path, logger: logging.Logger) -> None:
    if complete(output, "cache"):
        logger.info("cache already complete; reusing verified arrays")
        return
    if not complete(output, "preflight"):
        raise RuntimeError("Run preflight before cache")
    started = time.perf_counter()
    matlab, vsb = _datasets(raw_root)
    train_mat = load_mat_partition(matlab, "Tr1.mat")
    val_mat = load_mat_partition(matlab, "Va1.mat")
    matlab_values = np.concatenate([train_mat.signal, val_mat.signal], axis=0)
    matlab_spec = _matlab_spec()
    matlab_path = cache / "matlab_global_log_power.npy"
    if not matlab_path.is_file():
        from partial_discharge_adaptive_fusion.spectrogram import write_atomic_array_cache
        matlab_stft = build_matlab_spectrogram_dataset(matlab_values, matlab_spec).astype(np.float16)
        write_atomic_array_cache(matlab_path, matlab_stft, _cache_contract(config, "matlab", matlab_stft.shape))
    metadata = _development_metadata(audit_root, vsb)
    metadata.to_csv(cache / "vsb_development_metadata.csv", index=False)
    spec = _vsb_spec(config)
    vsb_path = cache / "vsb_global_log_power.npy"
    expected_shape = (len(metadata), *spec.expected_shape)
    if not vsb_path.is_file() or not Path(str(vsb_path) + ".complete").is_file():
        logger.info("VSB global cache started: signals=%d estimated_bytes=%.2f GiB", len(metadata), np.prod(expected_shape) * 2 / 1024**3)
        processed = 0
        def writer(values: np.ndarray) -> None:
            nonlocal processed
            for batch in iter_vsb_signal_batches(vsb, metadata, batch_size=2):
                converted = compute_full_signal_log_power(batch.signal, spec).astype(np.float16)
                values[processed : processed + len(converted)] = converted
                processed += len(converted)
                if processed == len(metadata) or processed % 100 == 0:
                    logger.info("VSB global cache progress: %d/%d signals", processed, len(metadata))
        payload = _cache_contract(config, "vsb", expected_shape)
        write_global_spectrogram_cache(vsb_path, shape=expected_shape, metadata=payload, writer=writer)
    else:
        logger.info("VSB global cache file exists; validating completion metadata")
        metadata_json = json.loads(Path(str(vsb_path) + ".json").read_text(encoding="utf-8"))
        values = require_global_spectrogram_cache(vsb_path, metadata_json["fingerprint"])
        if tuple(values.shape) != expected_shape:
            raise RuntimeError("Existing VSB global cache shape mismatch")
    cache_manifest = {
        "status": "PASS", "elapsed_seconds": time.perf_counter() - started,
        "matlab_shape": list(np.load(matlab_path, mmap_mode="r").shape),
        "vsb_shape": list(np.load(vsb_path, mmap_mode="r").shape),
        "matlab_cache": str(matlab_path), "vsb_cache": str(vsb_path),
        "v5_event_detector_used": False, "local_windows_used": False,
    }
    atomic_json(cache / "cache_manifest.json", cache_manifest)
    finish(output, "cache", cache_manifest)
    logger.info("cache PASS: MATLAB=%s VSB=%s", cache_manifest["matlab_shape"], cache_manifest["vsb_shape"])


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=float), -40, 40)))


def _final_predictions(train_x: Any, train_y: np.ndarray, validation_x: Any, validation_y: np.ndarray, *, kind: str, seed: int, epochs: int, batch_size: int) -> np.ndarray:
    result = train_expert(
        train_x, train_y, validation_x, validation_y, kind=kind, seed=seed, epochs=epochs,
        batch_size=batch_size, pos_weight=positive_class_weight(train_y),
        train_indices=None, validation_indices=None, validation_batch_size=32,
        num_workers=0, persistent_workers=False, mixed_precision=False, epoch_selection="fixed",
    )
    loader = __import__("torch").utils.data.DataLoader(
        NumpyDataset(validation_x, validation_y), batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return _sigmoid(predict_logits(result.model, loader))


def _append_prediction_rows(rows: list[pd.DataFrame], base: pd.DataFrame, probability: np.ndarray, threshold: float, *, dataset: str, split: str, method: str, seed: int) -> None:
    current = base.copy()
    current["dataset"] = dataset
    current["split"] = split
    current["method"] = method
    current["seed"] = int(seed)
    current["probability"] = np.asarray(probability, dtype=float)
    current["threshold"] = float(threshold)
    current["prediction"] = (current["probability"] >= threshold).astype(np.int64)
    rows.append(current)


def stage_experts(config: dict[str, Any], raw_root: Path, audit_root: Path, output: Path, cache: Path, logger: logging.Logger) -> None:
    if complete(output, "experts"):
        logger.info("experts already complete; reusing predictions")
        return
    if not complete(output, "cache"):
        raise RuntimeError("Run cache before experts")
    started = time.perf_counter()
    matlab, vsb = _datasets(raw_root)
    train_mat, val_mat = load_mat_partition(matlab, "Tr1.mat"), load_mat_partition(matlab, "Va1.mat")
    matlab_raw = np.load(cache / "matlab_global_log_power.npy", mmap_mode="r")
    mat_raw_train, mat_raw_val = matlab_raw[: len(train_mat.signal)], matlab_raw[len(train_mat.signal) :]
    mat_folds = fold_assignments(train_mat.label, n_splits=5, seed=42)
    rows: list[pd.DataFrame] = []
    mat_base_train = pd.DataFrame({"sample_id": train_mat.sample_id.astype(str), "id_measurement": train_mat.sample_id.astype(str), "phase": "NA", "target": train_mat.label})
    mat_base_val = pd.DataFrame({"sample_id": val_mat.sample_id.astype(str), "id_measurement": val_mat.sample_id.astype(str), "phase": "NA", "target": val_mat.label})
    temporal_epochs = int(config["training"].get("matlab_temporal_epochs", 5))
    matlab_spec_epochs = int(config["training"]["matlab_global_spectrogram"].get("epochs", 5))
    vsb_spec_epochs = int(config["training"]["vsb_global_spectrogram"].get("epochs", 7))
    standardizers: list[dict[str, Any]] = []
    for seed in [int(value) for value in config["scope"]["seeds"]]:
        logger.info("MATLAB seed %d temporal OOF", seed)
        def mat_temporal_fold(raw: Any, fit: np.ndarray, holdout: np.ndarray, fold: int) -> np.ndarray:
            standardizer = fit_standardizer(np.asarray(raw)[fit], axis=0)
            return temporal_input(np.asarray(raw), standardizer)
        def mat_spec_fold(raw: Any, fit: np.ndarray, holdout: np.ndarray, fold: int) -> Any:
            standardizer = fit_spectrogram_standardizer(raw, fit)
            standardizers.append({"dataset": "matlab", "representation": "global_spectrogram", "seed": seed, "fold": fold, "fingerprint": standardizer.fingerprint})
            return standardizer.view(raw)
        temporal_oof = cross_fitted_logits(
            train_mat.signal, train_mat.label, kind="temporal", folds=mat_folds, seed=seed,
            epochs=temporal_epochs, batch_size=4, validation_batch_size=32, num_workers=0,
            fold_transform=mat_temporal_fold, epoch_selection="fixed",
        ).logits
        spec_oof = cross_fitted_logits(
            mat_raw_train, train_mat.label, kind="spectrogram", folds=mat_folds, seed=seed,
            epochs=matlab_spec_epochs, batch_size=4, validation_batch_size=32, num_workers=0,
            fold_transform=mat_spec_fold, epoch_selection="fixed",
        ).logits
        temporal_standardizer = fit_standardizer(train_mat.signal, axis=0)
        spec_standardizer = fit_spectrogram_standardizer(mat_raw_train, np.arange(len(train_mat.signal)))
        mat_temporal_train, mat_temporal_val = temporal_input(train_mat.signal, temporal_standardizer), temporal_input(val_mat.signal, temporal_standardizer)
        mat_spec_train, mat_spec_val = spec_standardizer.transform(mat_raw_train), spec_standardizer.transform(mat_raw_val)
        temporal_val = _final_predictions(mat_temporal_train, train_mat.label, mat_temporal_val, val_mat.label, kind="temporal", seed=seed, epochs=temporal_epochs, batch_size=4)
        spec_val = _final_predictions(mat_spec_train, train_mat.label, mat_spec_val, val_mat.label, kind="spectrogram", seed=seed, epochs=matlab_spec_epochs, batch_size=4)
        temporal_prob_oof, spec_prob_oof = _sigmoid(temporal_oof), _sigmoid(spec_oof)
        temporal_threshold, spec_threshold = select_threshold(train_mat.label, temporal_prob_oof), select_threshold(train_mat.label, spec_prob_oof)
        _append_prediction_rows(rows, mat_base_train, temporal_prob_oof, temporal_threshold, dataset="matlab", split="train_oof", method="temporal", seed=seed)
        _append_prediction_rows(rows, mat_base_val, temporal_val, temporal_threshold, dataset="matlab", split="validation", method="temporal", seed=seed)
        _append_prediction_rows(rows, mat_base_train, spec_prob_oof, spec_threshold, dataset="matlab", split="train_oof", method="global_spectrogram", seed=seed)
        _append_prediction_rows(rows, mat_base_val, spec_val, spec_threshold, dataset="matlab", split="validation", method="global_spectrogram", seed=seed)
        logger.info("MATLAB seed %d experts ready", seed)

    forensic, canonical = prepare_forensic_baseline_frame(audit_root)
    inventory = build_modality_feature_inventory(canonical)
    temporal_columns = feature_set_columns(inventory, "S+T", canonical)
    metadata = pd.read_csv(cache / "vsb_development_metadata.csv", dtype={"sample_id": str, "id_measurement": str, "phase": str})
    train_idx = np.flatnonzero(metadata["split"].to_numpy() == "train")
    val_idx = np.flatnonzero(metadata["split"].to_numpy() == "validation")
    labels = metadata["target"].to_numpy(np.int64)
    groups = metadata["id_measurement"].astype(str).to_numpy()
    global_values = np.load(cache / "vsb_global_log_power.npy", mmap_mode="r")
    train_values = IndexedArrayView(global_values, train_idx)
    train_labels, train_groups = labels[train_idx], groups[train_idx]
    train_folds = metadata.loc[train_idx, "oof_fold"].to_numpy(np.int64)
    base = metadata[["sample_id", "id_measurement", "phase", "target"]].copy()
    for seed in [int(value) for value in config["scope"]["seeds"]]:
        baseline = evaluate_feature_baseline_with_predictions(forensic, temporal_columns, classifier="hist_gradient_boosting", seed=seed, grouped=True, mode="phase_independent")
        for current, split in ((baseline["train_oof_predictions"], "train_oof"), (baseline["validation_predictions"], "validation")):
            _append_prediction_rows(rows, current[["sample_id", "id_measurement", "phase", "target"]], current["probability"].to_numpy(), baseline["threshold_from_train_oof"], dataset="vsb", split=split, method="temporal", seed=seed)
        def vsb_spec_fold(raw: Any, fit: np.ndarray, holdout: np.ndarray, fold: int) -> FrequencyStandardizedView:
            standardizer = fit_frequency_standardizer(raw, fit, chunk_size=2)
            standardizers.append({"dataset": "vsb", "representation": "global_spectrogram", "seed": seed, "fold": fold, "fingerprint": standardizer.fingerprint, "fit_count": standardizer.fit_count})
            return standardizer.view(raw)
        logger.info("VSB seed %d global spectrogram OOF", seed)
        spec_oof = cross_fitted_logits(
            train_values, train_labels, kind="global_spectrogram", folds=train_folds, seed=seed,
            groups=train_groups, epochs=vsb_spec_epochs, batch_size=2, validation_batch_size=32,
            num_workers=0, fold_transform=vsb_spec_fold,
            imbalance_strategy="fold_local_pos_weight", epoch_selection="fixed",
        ).logits
        final_standardizer = fit_frequency_standardizer(train_values, np.arange(len(train_values)), chunk_size=2)
        final_train = final_standardizer.view(train_values)
        final_val = final_standardizer.view(IndexedArrayView(global_values, val_idx))
        spec_val = _final_predictions(final_train, train_labels, final_val, labels[val_idx], kind="global_spectrogram", seed=seed, epochs=vsb_spec_epochs, batch_size=2)
        spec_prob_oof = _sigmoid(spec_oof)
        threshold = select_threshold(train_labels, spec_prob_oof)
        _append_prediction_rows(rows, base.iloc[train_idx][["sample_id", "id_measurement", "phase", "target"]], spec_prob_oof, threshold, dataset="vsb", split="train_oof", method="global_spectrogram", seed=seed)
        _append_prediction_rows(rows, base.iloc[val_idx][["sample_id", "id_measurement", "phase", "target"]], spec_val, threshold, dataset="vsb", split="validation", method="global_spectrogram", seed=seed)
        standardizers.append({"dataset": "vsb", "representation": "global_spectrogram", "seed": seed, "scope": "final_train", "fingerprint": final_standardizer.fingerprint, "fit_count": final_standardizer.fit_count})
        logger.info("VSB seed %d experts ready", seed)
    predictions = pd.concat(rows, ignore_index=True)
    if predictions["sample_id"].duplicated().all():
        raise RuntimeError("Prediction identity validation failed")
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    atomic_json(output / "standardizer_fingerprints.json", {"records": standardizers})
    finish(output, "experts", {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "prediction_rows": int(len(predictions)), "event_detector_used": False, "seeds": config["scope"]["seeds"]})
    logger.info("experts PASS: prediction rows=%d", len(predictions))


def _metric_rows(predictions: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, overlaps, bootstraps = [], [], []
    for (dataset, seed), group in predictions[predictions["split"] == "validation"].groupby(["dataset", "seed"], sort=True):
        methods = {str(method): frame.sort_values("sample_id") for method, frame in group.groupby("method")}
        if set(methods) != {"temporal", "global_spectrogram"}:
            raise RuntimeError(f"Missing paired methods for {dataset} seed {seed}")
        temporal, spectrogram = methods["temporal"], methods["global_spectrogram"]
        if temporal["sample_id"].tolist() != spectrogram["sample_id"].tolist():
            raise RuntimeError("Temporal and global-spectrogram prediction IDs are not aligned")
        labels = temporal["target"].to_numpy(np.int64)
        for method, frame in methods.items():
            result = binary_metrics(labels, frame["probability"].to_numpy(float), float(frame["threshold"].iloc[0]))
            metrics.append({"dataset": dataset, "seed": int(seed), "method": method, **result})
        overlap = prediction_overlap_oracle(labels, temporal["prediction"].to_numpy(np.int64), spectrogram["prediction"].to_numpy(np.int64), temporal["id_measurement"].to_numpy())
        for row in overlap["rows"]:
            overlaps.append({"dataset": dataset, "seed": int(seed), **row})
        if dataset == "vsb":
            delta = grouped_paired_bootstrap_delta(labels, spectrogram["prediction"].to_numpy(np.int64), temporal["prediction"].to_numpy(np.int64), temporal["id_measurement"].to_numpy(), iterations=10000, seed=42042 + int(seed))
        else:
            delta = paired_bootstrap_delta(labels, spectrogram["prediction"].to_numpy(np.int64), temporal["prediction"].to_numpy(np.int64), iterations=10000, seed=42042 + int(seed))
        bootstraps.append({"dataset": dataset, "seed": int(seed), "comparison": "global_spectrogram_minus_temporal", "oracle_headroom": overlap["oracle_headroom"], **delta})
    return metrics, overlaps, bootstraps


def _regression(config: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for dataset, section in (("vsb", "vsb_temporal"), ("matlab", "matlab_temporal")):
        expected = config["regression"][section]["expected"]
        mcc_tol = float(config["regression"][section]["tolerance_mcc"])
        threshold_tol = float(config["regression"][section]["tolerance_threshold"])
        observed = [row for row in metrics if row["dataset"] == dataset and row["method"] == "temporal"]
        for row in observed:
            reference = expected[str(row["seed"])] if str(row["seed"]) in expected else expected[row["seed"]]
            checks.append({"dataset": dataset, "seed": int(row["seed"]), "observed_mcc": row["mcc"], "expected_mcc": reference["mcc"], "observed_threshold": row["threshold"], "expected_threshold": reference["threshold"], "mcc_pass": abs(row["mcc"] - reference["mcc"]) <= mcc_tol, "threshold_pass": abs(row["threshold"] - reference["threshold"]) <= threshold_tol})
    return {"checks": checks, "passed": bool(checks) and all(item["mcc_pass"] and item["threshold_pass"] for item in checks)}


def stage_evaluation(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "evaluation"):
        logger.info("evaluation already complete; reusing report tables")
        return
    if not complete(output, "experts"):
        raise RuntimeError("Run experts before evaluation")
    started = time.perf_counter()
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    required = {"dataset", "split", "sample_id", "id_measurement", "phase", "target", "method", "seed", "probability", "prediction", "threshold"}
    if not required.issubset(predictions.columns):
        raise RuntimeError(f"Prediction schema missing {sorted(required - set(predictions.columns))}")
    if not np.isfinite(predictions["probability"].to_numpy(float)).all() or not predictions["probability"].between(0, 1).all():
        raise RuntimeError("Prediction probabilities are not finite in [0,1]")
    metrics, overlaps, bootstraps = _metric_rows(predictions)
    metric_frame = pd.DataFrame(metrics)
    overlap_frame = pd.DataFrame(overlaps)
    bootstrap_frame = pd.DataFrame(bootstraps)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    overlap_frame.to_csv(output / "prediction_overlap.csv", index=False)
    bootstrap_frame.to_json(output / "bootstrap_deltas.json", orient="records", indent=2)
    hierarchical: list[dict[str, Any]] = []
    for dataset in ("matlab", "vsb"):
        rows = bootstrap_frame[bootstrap_frame["dataset"] == dataset]
        if len(rows) == 3:
            deltas = rows["point_estimate"].to_numpy(float)
            hierarchical.append({"dataset": dataset, "mean_delta": float(deltas.mean()), "sd_delta": float(deltas.std(ddof=1)), "min_delta": float(deltas.min()), "max_delta": float(deltas.max()), "positive_seeds": int((deltas > 0).sum()), "oracle_headroom_mean": float(rows["oracle_headroom"].mean())})
    atomic_json(output / "hierarchical_bootstrap.json", hierarchical)
    regression = _regression(config, metrics)
    atomic_json(output / "regression.json", regression)
    # Fixed mixtures are diagnostics only and use OOF-selected thresholds/weights.
    mixture_rows: list[dict[str, Any]] = []
    for (dataset, seed), group in predictions.groupby(["dataset", "seed"], sort=True):
        oof = group[group["split"] == "train_oof"]
        val = group[group["split"] == "validation"]
        t_oof = oof[oof["method"] == "temporal"].sort_values("sample_id")
        s_oof = oof[oof["method"] == "global_spectrogram"].sort_values("sample_id")
        t_val = val[val["method"] == "temporal"].sort_values("sample_id")
        s_val = val[val["method"] == "global_spectrogram"].sort_values("sample_id")
        curve, best = select_fixed_weight_oof(t_val["target"].to_numpy(np.int64), t_val["probability"].to_numpy(float), s_val["probability"].to_numpy(float), t_oof["target"].to_numpy(np.int64), t_oof["probability"].to_numpy(float), s_oof["probability"].to_numpy(float), weights=config["diagnostics"]["fixed_weights"])
        for row in curve:
            mixture_rows.append({"dataset": dataset, "seed": int(seed), **row, "diagnostic_only": True})
    pd.DataFrame(mixture_rows).to_csv(output / "fixed_mixtures.csv", index=False)
    finish(output, "evaluation", {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "metrics": len(metrics), "overlap_rows": len(overlaps), "bootstrap_rows": len(bootstraps), "regression": regression})
    logger.info("evaluation PASS: metrics=%d overlap=%d regression=%s", len(metrics), len(overlaps), regression["passed"])


def classify_verdict(config: dict[str, Any], metric_frame: pd.DataFrame, bootstrap_frame: pd.DataFrame, regression_ok: bool) -> str:
    if not regression_ok or metric_frame.empty:
        return "INCONCLUSIVE"
    summary: dict[str, Any] = {}
    for dataset in ("matlab", "vsb"):
        values = metric_frame[(metric_frame.dataset == dataset) & (metric_frame.method == "global_spectrogram")]["mcc"].to_numpy(float)
        if dataset == "vsb":
            summary["vsb_strong"] = len(values) == 3 and values.mean() >= config["verdict"]["vsb_strong_mean_mcc"] and values.min() > config["verdict"]["vsb_strong_min_mcc"]
        else:
            summary["matlab_strong"] = len(values) == 3 and values.mean() >= config["verdict"]["matlab_strong_mean_mcc"] and values.min() >= config["verdict"]["matlab_strong_min_mcc"]
    headroom = bootstrap_frame.groupby("dataset")["oracle_headroom"].agg(["mean", "min"]).to_dict("index")
    supported = all(item["mean"] >= config["verdict"]["meaningful_oracle_headroom"] for item in headroom.values()) if headroom else False
    if summary.get("vsb_strong") and summary.get("matlab_strong") and supported:
        return "TEMPORAL+GLOBAL-SPECTROGRAM SUPPORTED"
    if summary.get("matlab_strong") and not summary.get("vsb_strong"):
        return "GLOBAL-SPECTROGRAM SUPPORTED ONLY ON MATLAB"
    if summary.get("vsb_strong") and not summary.get("matlab_strong"):
        return "GLOBAL-SPECTROGRAM PROMISING BUT WEAKER"
    if not supported:
        return "GLOBAL-SPECTROGRAM NOT COMPLEMENTARY"
    return "GLOBAL-SPECTROGRAM NOT SUPPORTED"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in frame.columns) + " |")
    return "\n".join(lines)


def stage_report(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "report"):
        logger.info("report already complete; reusing curated report")
        return
    if not complete(output, "evaluation"):
        raise RuntimeError("Run evaluation before report")
    metric_frame = pd.read_csv(output / "metrics.csv")
    bootstrap_frame = pd.read_json(output / "bootstrap_deltas.json")
    regression = json.loads((output / "regression.json").read_text(encoding="utf-8"))
    verdict = classify_verdict(config, metric_frame, bootstrap_frame, bool(regression["passed"]))
    summary = {
        "verdict": verdict,
        "regression_ok": bool(regression["passed"]),
        "metrics": metric_frame.to_dict(orient="records"),
        "bootstrap": bootstrap_frame.to_dict(orient="records"),
        "regression": regression,
        "representation": "one global VSB 800000-sample log10-power STFT; no event detector, local windows, MIL, mask, CWT rerun, or adaptive fusion",
        "holdouts": "locked",
        "recommendation": "Do not interpret scientifically until the temporal regression and all input/provenance checks pass.",
    }
    report = [
        "# Full-Signal Temporal vs Global Spectrogram Diagnostic", "",
        f"**Verdict:** `{verdict}`", "",
        "## OBSERVED", "",
        "This development-only diagnostic uses MATLAB Tr1/Va1 and VSB grouped development identities. VSB grouped/official test signals and MATLAB Te1/Te2 remain locked.", "",
        "VSB uses one complete 800,000-sample waveform transformed into one global Hann STFT/log-power array. The active path does not call the V5 pulse detector and does not aggregate local events.", "",
        "### Metrics", "", _markdown_table(metric_frame), "",
        "### Regression", "", _markdown_table(pd.DataFrame(regression["checks"])), "",
        "## INTERPRETATION", "",
        "Fixed mixtures are diagnostic-only. A failed temporal regression forces `INCONCLUSIVE`, even if a spectrogram result appears strong.", "",
        "The result is intended to distinguish a representation limitation from the prior local-event implementation; it does not constitute adaptive fusion confirmation.", "",
        "## UNRESOLVED", "",
        "The experiment is not executed by the implementation turn. Historical local-event and CWT results remain contextual and are not rerun here. Literature frequency concepts are not treated as a direct reproduction.", "",
        "## Reproducibility", "",
        "Raw data are accessed only through `PD_RAW_DATA_ROOT` and all generated arrays are versioned under the full-signal diagnostic namespace.",
    ]
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary_path = ROOT / config["outputs"]["summary"]
    atomic_json(summary_path, summary)
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("# Full-Signal Spectrogram Diagnostic Manifest\n\n" + json.dumps({"verdict": verdict, "regression_ok": regression["passed"], "output_root": str(output)}, indent=2) + "\n", encoding="utf-8")
    finish(output, "report", {"status": "PASS", "verdict": verdict, "report": str(report_path), "summary": str(summary_path)})
    logger.info("report PASS: verdict=%s", verdict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "cache", "experts", "evaluation", "report", "all"], default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--audit-root", type=Path, default=Path("results/audits/vsb-literature-forensic"))
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_root or ROOT / config["outputs"]["audit_root"]
    cache = args.cache_root or ROOT / config["outputs"]["cache_root"]
    audit_root = args.audit_root if args.audit_root.is_absolute() else ROOT / args.audit_root
    logger = make_logger(args.log_file)
    stages = ["preflight", "cache", "experts", "evaluation", "report"] if args.stage == "all" else [args.stage]
    functions = {
        "preflight": lambda: stage_preflight(config, args.raw_root.resolve(), audit_root, output, logger),
        "cache": lambda: stage_cache(config, args.raw_root.resolve(), audit_root, output, cache, logger),
        "experts": lambda: stage_experts(config, args.raw_root.resolve(), audit_root, output, cache, logger),
        "evaluation": lambda: stage_evaluation(config, output, logger),
        "report": lambda: stage_report(config, output, logger),
    }
    for stage in stages:
        logger.info("stage %s started", stage)
        functions[stage]()
        logger.info("stage %s finished", stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
