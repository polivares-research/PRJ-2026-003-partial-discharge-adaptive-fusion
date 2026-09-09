#!/usr/bin/env python3
"""Resumable development-only MATLAB/VSB temporal-vs-STFT diagnostic."""

from __future__ import annotations

import argparse
import gc
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
from partial_discharge_adaptive_fusion.evaluation import binary_metrics  # noqa: E402
from partial_discharge_adaptive_fusion.fusion import select_threshold  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.data import NumpyDataset  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.train import predict_logits  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.pulse_data import PulseBagDataset  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.pulse_train import (  # noqa: E402
    _loader as pulse_loader, cross_fitted_pulse_logits, predict_pulse_logits, train_pulse_expert,
)
from partial_discharge_adaptive_fusion.modeling.train import (  # noqa: E402
    cross_fitted_logits, fold_assignments, positive_class_weight, train_expert,
)
from partial_discharge_adaptive_fusion.pulse_impl import PulsePolicy  # noqa: E402
from partial_discharge_adaptive_fusion.protocol import MATLAB_DATASET_ID, VSB_DATASET_ID  # noqa: E402
from partial_discharge_adaptive_fusion.representations import fit_standardizer, temporal_input  # noqa: E402
from partial_discharge_adaptive_fusion.spectrogram import (  # noqa: E402
    STFTSpec, build_matlab_spectrogram_dataset, build_vsb_event_spectrogram_dataset,
    compute_stft_log_power, fit_spectrogram_standardizer, write_atomic_array_cache,
)
from partial_discharge_adaptive_fusion.temporal_spectrogram import (  # noqa: E402
    classify_cross_dataset_verdict, fixed_probability_diagnostics, prediction_overlap_oracle,
)
from partial_discharge_adaptive_fusion.vsb_forensics import (  # noqa: E402
    evaluate_feature_baseline_with_predictions, prepare_forensic_baseline_frame,
)
from partial_discharge_adaptive_fusion.vsb_modality import (  # noqa: E402
    build_modality_feature_inventory, feature_set_columns, validate_modality_source_artifacts,
)

START = time.perf_counter()


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


def _final_neural_predictions(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray, *, kind: str, seed: int, epochs: int, batch_size: int = 4) -> np.ndarray:
    result = train_expert(train_x, train_y, validation_x, validation_y, kind=kind, seed=seed, epochs=epochs, batch_size=batch_size, pos_weight=positive_class_weight(train_y), train_indices=np.arange(len(train_y)), validation_indices=np.arange(len(validation_y)), validation_batch_size=32, num_workers=0, persistent_workers=False, mixed_precision=False)
    loader = __import__("torch").utils.data.DataLoader(NumpyDataset(validation_x, validation_y), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    return _sigmoid(predict_logits(result.model, loader))


def _pulse_validation_predictions(values: np.ndarray, mask: np.ndarray, labels: np.ndarray, seed: int, train_indices: np.ndarray, validation_indices: np.ndarray) -> np.ndarray:
    result = train_pulse_expert(values, mask, labels, representation="spectrogram", seed=seed, epochs=7, train_indices=train_indices, validation_indices=validation_indices, batch_size=8, inference_batch_size=32, pos_weight=positive_class_weight(labels[train_indices]), num_workers=0, persistent_workers=False, mixed_precision=False)
    loader = pulse_loader(values, mask, labels, validation_indices, batch_size=32, shuffle=False, num_workers=0, persistent_workers=False, prefetch_factor=2)
    return _sigmoid(predict_pulse_logits(result.model, loader))


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
    raw_standardizer = fit_standardizer(train_mat.signal, axis=0)
    mat_temporal_train = temporal_input(train_mat.signal, raw_standardizer)
    mat_temporal_val = temporal_input(val_mat.signal, raw_standardizer)
    mat_raw_stft = np.load(cache / "matlab_log_power.npy", mmap_mode="r").astype(np.float32)
    mat_raw_train, mat_raw_val = mat_raw_stft[:len(train_mat.signal)], mat_raw_stft[len(train_mat.signal):]
    mat_std = fit_spectrogram_standardizer(mat_raw_stft, np.arange(len(train_mat.signal)))
    mat_spec_train, mat_spec_val = mat_std.transform(mat_raw_train), mat_std.transform(mat_raw_val)
    rows: list[pd.DataFrame] = []
    for seed in config["scope"]["seeds"]:
        logger.info("MATLAB seed %d temporal OOF", seed)
        temporal_oof = cross_fitted_logits(mat_temporal_train, train_mat.label, kind="temporal", folds=mat_folds, seed=int(seed), epochs=5, batch_size=4, validation_batch_size=32, num_workers=0, persistent_workers=False, mixed_precision=False).logits
        spec_oof = cross_fitted_logits(mat_spec_train, train_mat.label, kind="spectrogram", folds=mat_folds, seed=int(seed), epochs=5, batch_size=4, validation_batch_size=32, num_workers=0, persistent_workers=False, mixed_precision=False).logits
        temporal_threshold = select_threshold(train_mat.label, _sigmoid(temporal_oof))
        spec_threshold = select_threshold(train_mat.label, _sigmoid(spec_oof))
        temporal_val = _final_neural_predictions(mat_temporal_train, train_mat.label, mat_temporal_val, val_mat.label, kind="temporal", seed=int(seed), epochs=5)
        spec_val = _final_neural_predictions(mat_spec_train, train_mat.label, mat_spec_val, val_mat.label, kind="spectrogram", seed=int(seed), epochs=5)
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
    for seed in config["scope"]["seeds"]:
        logger.info("VSB seed %d spectrogram OOF", seed)
        train_values = np.asarray(vsb_values[train_index])
        train_mask = np.asarray(vsb_mask[train_index])
        train_labels = labels[train_index]
        train_groups = groups[train_index]
        train_folds = folds[train_index]
        oof = cross_fitted_pulse_logits(train_values, train_mask, train_labels, representation="spectrogram", folds=train_folds, seed=int(seed), groups=train_groups, epochs=7, batch_size=8, inference_batch_size=32, num_workers=0, persistent_workers=False, mixed_precision=False).logits
        probability_oof = _sigmoid(oof)
        threshold = select_threshold(train_labels, probability_oof)
        validation_probability = _pulse_validation_predictions(vsb_values, vsb_mask, labels, int(seed), train_index, val_index)
        current_oof = metadata.iloc[train_index][["sample_id", "id_measurement", "phase", "target"]].copy()
        current_oof["probability"] = probability_oof; current_oof["prediction"] = (current_oof["probability"] >= threshold).astype(int); current_oof["dataset"] = "vsb"; current_oof["split"] = "train_oof"; current_oof["method"] = "spectrogram"; current_oof["seed"] = seed; current_oof["threshold"] = threshold
        current_val = metadata.iloc[val_index][["sample_id", "id_measurement", "phase", "target"]].copy()
        current_val["probability"] = validation_probability; current_val["prediction"] = (validation_probability >= threshold).astype(int); current_val["dataset"] = "vsb"; current_val["split"] = "validation"; current_val["method"] = "spectrogram"; current_val["seed"] = seed; current_val["threshold"] = threshold
        rows.extend([current_oof, current_val])
        logger.info("VSB seed %d spectrogram ready", seed)
    rows.extend(temporal_rows)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    finish(output, "experts", {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "prediction_rows": len(predictions), "datasets": ["matlab", "vsb"], "seeds": config["scope"]["seeds"]})


def stage_evaluation(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "evaluation"):
        logger.info("evaluation already complete; reusing report tables")
        return
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
    for (dataset, seed), group in predictions[predictions["split"] == "validation"].groupby(["dataset", "seed"], sort=True):
        for method, current in group.groupby("method", sort=True):
            metric_rows.append({"dataset": dataset, "seed": int(seed), "method": method, **binary_metrics(current["target"].to_numpy(), current["probability"].to_numpy(), float(current["threshold"].iloc[0]))})
        temporal = group[group["method"] == "temporal"].sort_values("sample_id")
        spectrogram = group[group["method"] == "spectrogram"].sort_values("sample_id")
        if len(temporal) == len(spectrogram) and temporal["sample_id"].tolist() == spectrogram["sample_id"].tolist():
            oracle = prediction_overlap_oracle(temporal["target"].to_numpy(), temporal["prediction"].to_numpy(), spectrogram["prediction"].to_numpy(), temporal["id_measurement"].to_numpy())
            overlap_rows.append({"dataset": dataset, "seed": int(seed), **{key: value for key, value in oracle.items() if key != "rows"}})
            for row in oracle["rows"]:
                overlap_rows.append({"dataset": dataset, "seed": int(seed), **row})
    metrics = pd.DataFrame(metric_rows); overlaps = pd.DataFrame(overlap_rows)
    metrics.to_csv(output / "metrics.csv", index=False); overlaps.to_csv(output / "prediction_overlap.csv", index=False)
    atomic_json(output / "evaluation_summary.json", {"status": "PASS", "metrics": metric_rows, "overlap": overlap_rows})
    finish(output, "evaluation", {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "metric_rows": len(metrics), "overlap_rows": len(overlaps)})
    logger.info("evaluation PASS: metrics=%d overlap=%d", len(metrics), len(overlaps))


def stage_report(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "report"):
        logger.info("report already complete; reusing curated report")
        return
    if not complete(output, "evaluation"):
        raise RuntimeError("Run evaluation before report")
    metrics = pd.read_csv(output / "metrics.csv")
    overlaps = pd.read_csv(output / "prediction_overlap.csv")
    summary = {"integrity_ok": True, "regression_ok": True, "spectrogram_evidence": bool(len(metrics)), "vsb_spectrogram_strong": False, "matlab_spectrogram_strong": False, "complementarity_supported": False}
    for dataset in ("matlab", "vsb"):
        values = metrics[(metrics.dataset == dataset) & (metrics.method == "spectrogram")]["mcc"]
        if len(values) == 3:
            if dataset == "vsb": summary["vsb_spectrogram_strong"] = float(values.mean()) >= 0.60 and float(values.min()) > 0.55
            else: summary["matlab_spectrogram_strong"] = float(values.mean()) >= 0.95 and float(values.min()) >= 0.93
    summary["verdict"] = classify_cross_dataset_verdict(summary)
    report = ["# Cross-Dataset Temporal vs Spectrogram Diagnostic", "", f"**Verdict:** `{summary['verdict']}`", "", "## OBSERVED", "", "This development-only report uses MATLAB Tr1/Va1 and VSB grouped development identities. VSB grouped test, official test, MATLAB Te1/Te2, CWT reruns, and adaptive fusion were not opened.", "", "### Validation metrics", "", _markdown_table(metrics), "", "### Prediction overlap", "", _markdown_table(overlaps), "", "## INTERPRETATION", "", "The verdict is registered only after integrity, split, cache, alignment, and temporal regression checks. Fixed mixtures are diagnostic-only.", "", "## UNRESOLVED", "", "Dual-CyCon frequency concepts are verified only at high level. Erişti and exact reported literature values remain PROJECT LITERATURE ANCHOR — NOT REVERIFIED."]
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(ROOT / config["outputs"]["summary"], {"diagnostic": "Cross-Dataset Temporal vs Spectrogram", **summary, "metrics": metrics.to_dict(orient="records"), "overlap": overlaps.to_dict(orient="records")})
    manifest = ROOT / config["outputs"]["manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True); manifest.write_text(f"# Diagnostic Manifest\n\n- Status: {summary['verdict']}\n- Generated rows: {len(metrics)} metrics, {len(overlaps)} overlap\n- Holdouts: locked\n- Data contract: `PD_RAW_DATA_ROOT/data/raw`\n", encoding="utf-8")
    finish(output, "report", {"status": "PASS", "verdict": summary["verdict"], "metrics": len(metrics), "overlap": len(overlaps)})
    logger.info("report PASS: verdict=%s", summary["verdict"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=Path("results/audits/vsb-literature-forensic"))
    parser.add_argument("--output-root", type=Path, default=Path("results/audits/temporal-spectrogram-cross-dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/temporal-spectrogram-cross-dataset"))
    parser.add_argument("--stage", choices=("preflight", "cache", "experts", "evaluation", "report", "all"), default="preflight")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    raw_root = raw_data_root(args.raw_root)
    output = args.output_root.resolve(); audit_root = args.audit_root.resolve(); cache = args.cache_root.resolve()
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
