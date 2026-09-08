"""Train the V5 VSB development candidates and write gate seed records.

This runner consumes only the verified V5 development caches.  It never opens
the VSB grouped holdout, MATLAB Te2, or the official unlabeled VSB test.  The
three pulse candidates are derived lazily from the immutable Np=257/CWT=64
cache; the V4 uniform control is reported as unavailable when no compatible
cache has been prepared rather than being silently replaced.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

from partial_discharge_adaptive_fusion.config import require_cuda, runtime_info
from partial_discharge_adaptive_fusion.fusion import select_threshold, sigmoid
from partial_discharge_adaptive_fusion.logging_utils import configure_progress_logging
from partial_discharge_adaptive_fusion.modeling.pulse_preprocessing import (
    IndexedPulseView,
    PulseMaskView,
    PulseRepresentationView,
    StandardizedPulseView,
    fit_pulse_standardizer,
)
from partial_discharge_adaptive_fusion.modeling.pulse_train import (
    _folds,
    _loader,
    predict_pulse_logits,
    shutdown_pulse_loader,
    train_pulse_expert,
)
from partial_discharge_adaptive_fusion.pulse_cache import (
    CacheMetadata,
    open_verified_memmap,
)
from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.reporting import prediction_frame, write_json, write_predictions
from partial_discharge_adaptive_fusion.v5_protocol import (
    V5_CANDIDATE_IDS,
    V5_DEVELOPMENT_SEEDS,
    assert_v5_holdouts_closed,
    validate_v5_config,
)


PULSE_CANDIDATES = ("pulse_np86_cwt32", "pulse_np86_cwt64", "pulse_np257_cwt64")
V4_CONTROL = "v4_uniform_control"
CACHE_FILES = {
    "temporal": "pulse_temporal_np257.dat",
    "cwt": "pulse_cwt_np257_scales64.dat",
    "valid_mask": "pulse_valid_np257.dat",
    "peak_indices": "pulse_peak_indices_np257.dat",
}


class CacheUnavailable(RuntimeError):
    """Raised when a predeclared candidate has no compatible prepared cache."""


def _cache_expected(payload: dict[str, Any]) -> CacheMetadata:
    return CacheMetadata(
        dataset_id=str(payload["dataset_id"]),
        dataset_version=str(payload["dataset_version"]),
        source_fingerprint=str(payload["source_fingerprint"]),
        split_manifest_hash=str(payload["split_manifest_hash"]),
        pulse_policy=dict(payload["pulse_policy"]),
        transform=dict(payload["transform"]),
        dtype=str(payload["dtype"]),
        preprocessing_version=str(payload["preprocessing_version"]),
        library_versions=dict(payload["library_versions"]),
        shape=tuple(int(value) for value in payload["shape"]),
        contract_version=str(payload["contract_version"]),
    )


def _open_verified_cache(cache_root: Path, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = cache_root / CACHE_FILES[name]
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not metadata_path.is_file():
        raise CacheUnavailable(f"Missing cache metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    values, observed = open_verified_memmap(path, _cache_expected(payload))
    return values, observed


def _load_caches(cache_root: Path, logger) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    logger.info(f"opening verified V5 caches: root={cache_root}")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for name in CACHE_FILES:
        arrays[name], metadata[name] = _open_verified_cache(cache_root, name)
        logger.info(f"cache verified: {name}, shape={tuple(arrays[name].shape)}, dtype={arrays[name].dtype}")
    common_keys = ("source_fingerprint", "split_manifest_hash", "contract_version", "preprocessing_version")
    for key in common_keys:
        observed = {metadata[name][key] for name in metadata}
        if len(observed) != 1:
            raise CacheUnavailable(f"V5 cache metadata disagree on {key}: {sorted(observed)}")
    return arrays, metadata


def _load_manifest(cache_root: Path, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    path = cache_root / "v5_development_parent_manifest.csv"
    if not path.is_file():
        raise CacheUnavailable(f"Missing V5 development manifest: {path}")
    frame = pd.read_csv(path)
    required = {"signal_id", "id_measurement", "phase", "target", "split", "oof_fold"}
    missing = required - set(frame.columns)
    if missing:
        raise CacheUnavailable(f"V5 development manifest is missing columns: {sorted(missing)}")
    if len(frame) != arrays["temporal"].shape[0]:
        raise CacheUnavailable("V5 cache arrays and parent manifest have different row counts")
    if set(frame["split"].astype(str)) - {"train", "validation"}:
        raise CacheUnavailable("Development cache contains a blocked or unknown split")
    train = frame["split"].astype(str).eq("train")
    if train.sum() == 0 or frame.loc[train, "oof_fold"].astype(int).min() < 0:
        raise CacheUnavailable("V5 development manifest has incomplete training OOF folds")
    train_groups = set(frame.loc[train, "id_measurement"].astype(str))
    validation_groups = set(frame.loc[~train, "id_measurement"].astype(str))
    overlap = train_groups & validation_groups
    if overlap:
        raise CacheUnavailable(f"VSB id_measurement groups overlap train/validation: {sorted(overlap)[:5]}")
    if frame["signal_id"].astype(str).duplicated().any():
        raise CacheUnavailable("V5 development manifest contains duplicate signal IDs")
    return frame.reset_index(drop=True)


def _scale_indices(n_scales: int, available: int = 64) -> np.ndarray:
    if n_scales == available:
        return np.arange(available, dtype=np.int64)
    # The prepared cache spans the registered 1.5--64 scale range.  Select
    # nearest bins spanning that same range for the bounded 32-scale candidate.
    return np.unique(np.rint(np.linspace(0, available - 1, n_scales)).astype(np.int64))


def _candidate_views(
    candidate_id: str,
    candidate: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if candidate_id == V4_CONTROL:
        raise CacheUnavailable(
            "v4_uniform_control has no compatible V5 uniform-window cache in this checkout; "
            "prepare that cache before including the control in the gate"
        )
    if candidate_id not in PULSE_CANDIDATES:
        raise ValueError(f"Unsupported V5 candidate: {candidate_id}")
    n_pulses = int(candidate.get("n_pulses_per_half", 0))
    n_scales = int(candidate.get("cwt_scales", 64))
    if n_pulses not in {86, 257} or n_scales not in {32, 64}:
        raise CacheUnavailable(f"Candidate {candidate_id} is not compatible with the prepared cache")
    scale_indices = _scale_indices(n_scales)
    temporal = PulseRepresentationView(
        arrays["temporal"], representation="temporal", n_pulses=n_pulses,
    )
    cwt = PulseRepresentationView(
        arrays["cwt"], representation="cwt", n_pulses=n_pulses, scale_indices=scale_indices,
    )
    mask = PulseMaskView(arrays["valid_mask"], n_pulses)
    return temporal, cwt, mask, mask, {
        "n_pulses_per_half": n_pulses,
        "cwt_scales": n_scales,
        "cwt_scale_indices": scale_indices.tolist(),
        "cwt_scale_policy": "nearest_bins_spanning_registered_64_scale_range",
    }


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _completed_seed_record(
    output: Path,
    candidate_id: str,
    seed: int,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    labels_train: np.ndarray,
    labels_validation: np.ndarray,
) -> dict[str, Any] | None:
    """Rebuild one completed seed record from atomically written seed artifacts."""

    record_path = output / "training_record.json"
    oof_path = output / "oof_predictions.parquet"
    validation_path = output / "validation_predictions.parquet"
    if not all(path.is_file() for path in (record_path, oof_path, validation_path)):
        return None
    try:
        training = json.loads(record_path.read_text(encoding="utf-8"))
        oof = pd.read_parquet(oof_path)
        validation = pd.read_parquet(validation_path)
        required = {"sample_id", "label", "seed", "probability_temporal", "probability_cwt"}
        if required - set(oof.columns) or required - set(validation.columns):
            return None
        if training.get("candidate_id") != candidate_id or int(training.get("seed", -1)) != int(seed):
            return None
        if not np.array_equal(oof["sample_id"].astype(str).to_numpy(), train_ids.astype(str)):
            return None
        if not np.array_equal(validation["sample_id"].astype(str).to_numpy(), validation_ids.astype(str)):
            return None
        if not np.array_equal(oof["label"].to_numpy(np.int64), labels_train):
            return None
        if not np.array_equal(validation["label"].to_numpy(np.int64), labels_validation):
            return None
        temporal_threshold = float(training["temporal_threshold_from_oof"])
        cwt_threshold = float(training["cwt_threshold_from_oof"])
        temporal_oof = oof["probability_temporal"].to_numpy(np.float64)
        cwt_oof = oof["probability_cwt"].to_numpy(np.float64)
        temporal_validation = validation["probability_temporal"].to_numpy(np.float64)
        cwt_validation = validation["probability_cwt"].to_numpy(np.float64)
        probabilities = (temporal_oof, cwt_oof, temporal_validation, cwt_validation)
        if not all(np.isfinite(values).all() and np.all((values >= 0.0) & (values <= 1.0)) for values in probabilities):
            return None
    except (KeyError, TypeError, ValueError, OSError):
        return None
    temporal_validation_mcc = float(matthews_corrcoef(labels_validation, temporal_validation >= temporal_threshold))
    cwt_validation_mcc = float(matthews_corrcoef(labels_validation, cwt_validation >= cwt_threshold))
    return {
        "seed": int(seed),
        "temporal_oof_mcc": float(matthews_corrcoef(labels_train, temporal_oof >= temporal_threshold)),
        "cwt_oof_mcc": float(matthews_corrcoef(labels_train, cwt_oof >= cwt_threshold)),
        "temporal_validation_mcc": temporal_validation_mcc,
        "cwt_validation_mcc": cwt_validation_mcc,
        "best_individual_mcc": max(temporal_validation_mcc, cwt_validation_mcc),
        "minimum_expert_mcc": min(temporal_validation_mcc, cwt_validation_mcc),
        "temporal_threshold_from_oof": temporal_threshold,
        "cwt_threshold_from_oof": cwt_threshold,
        "elapsed_seconds": 0.0,
        "resumed_from_verified_artifacts": True,
    }


def _train_oof(
    values: Any,
    valid_mask: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    *,
    representation: str,
    seed: int,
    epochs: int,
    batch_size: int,
    inference_batch_size: int,
    cycle_consistency: bool,
    runtime: dict[str, Any],
    logger,
) -> dict[str, Any]:
    """Train fold-local models and return parent-signal OOF logits."""

    logits = np.full(len(labels), np.nan, dtype=np.float64)
    histories: dict[str, list[dict[str, float]]] = {}
    standardizers: dict[str, dict[str, float | int]] = {}
    for fold in np.unique(folds):
        holdout = np.flatnonzero(folds == fold)
        fit = np.flatnonzero(folds != fold)
        fit_groups = groups[fit]
        inner_count = min(5, len(np.unique(fit_groups)))
        if inner_count < 2:
            raise RuntimeError(f"{representation} fold {fold}: insufficient groups for inner validation")
        inner = _folds(labels[fit], fit_groups, inner_count, seed + 1009 * int(fold))
        inner_train = fit[inner != 0]
        inner_validation = fit[inner == 0]
        standardizer = fit_pulse_standardizer(
            values, valid_mask, inner_train, logger=logger,
            stage=f"{representation} OOF fold {fold} standardizer", chunk_size=256,
        )
        normalized = StandardizedPulseView(values, valid_mask, standardizer)
        positives = int(np.sum(labels[inner_train] == 1))
        negatives = int(np.sum(labels[inner_train] == 0))
        pos_weight = negatives / positives if positives else None
        logger.info(
            f"{representation} OOF fold {fold}: fit={len(inner_train)}, "
            f"inner_validation={len(inner_validation)}, holdout={len(holdout)}"
        )
        result = train_pulse_expert(
            normalized, valid_mask, labels, representation=representation,
            seed=seed + int(fold), epochs=epochs, train_indices=inner_train,
            validation_indices=inner_validation, batch_size=batch_size,
            inference_batch_size=inference_batch_size, pos_weight=pos_weight,
            cycle_consistency=cycle_consistency,
            num_workers=int(runtime.get("num_workers", 0)),
            persistent_workers=bool(runtime.get("persistent_workers", False)),
            prefetch_factor=int(runtime.get("prefetch_factor", 2)),
            mixed_precision=bool(runtime.get("mixed_precision", False)),
        )
        loader = _loader(
            normalized, valid_mask, labels, holdout, batch_size=inference_batch_size,
            shuffle=False, num_workers=int(runtime.get("num_workers", 0)),
            persistent_workers=bool(runtime.get("persistent_workers", False)),
            prefetch_factor=int(runtime.get("prefetch_factor", 2)),
        )
        logits[holdout] = predict_pulse_logits(
            result.model, loader, mixed_precision=bool(runtime.get("mixed_precision", False)),
        )
        histories[str(int(fold))] = result.history
        standardizers[str(int(fold))] = {
            "mean": standardizer.mean, "std": standardizer.std,
            "n_values": standardizer.n_values, "best_epoch": result.best_epoch,
        }
        shutdown_pulse_loader(loader)
        del result, normalized, loader
        _cleanup_cuda()
        logger.info(f"{representation} OOF fold {fold}: completed")
    if not np.isfinite(logits).all():
        raise RuntimeError(f"{representation} OOF coverage is incomplete")
    return {"logits": logits, "histories": histories, "standardizers": standardizers}


def _train_validation_model(
    values: Any,
    valid_mask: Any,
    labels_all: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    folds: np.ndarray,
    *,
    representation: str,
    seed: int,
    epochs: int,
    batch_size: int,
    inference_batch_size: int,
    cycle_consistency: bool,
    runtime: dict[str, Any],
    logger,
) -> dict[str, Any]:
    """Fit on an OOF-derived training subset and predict the fixed validation split."""

    inner_train = train_indices[folds != np.min(folds)]
    inner_validation = train_indices[folds == np.min(folds)]
    standardizer = fit_pulse_standardizer(
        values, valid_mask, inner_train, logger=logger,
        stage=f"{representation} validation standardizer", chunk_size=256,
    )
    normalized = StandardizedPulseView(values, valid_mask, standardizer)
    positives = int(np.sum(labels_all[inner_train] == 1))
    negatives = int(np.sum(labels_all[inner_train] == 0))
    result = train_pulse_expert(
        normalized, valid_mask, labels_all, representation=representation,
        seed=seed, epochs=epochs, train_indices=inner_train,
        validation_indices=inner_validation, batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        pos_weight=(negatives / positives if positives else None),
        cycle_consistency=cycle_consistency,
        num_workers=int(runtime.get("num_workers", 0)),
        persistent_workers=bool(runtime.get("persistent_workers", False)),
        prefetch_factor=int(runtime.get("prefetch_factor", 2)),
        mixed_precision=bool(runtime.get("mixed_precision", False)),
    )
    loader = _loader(
        normalized, valid_mask, labels_all, validation_indices,
        batch_size=inference_batch_size, shuffle=False,
        num_workers=int(runtime.get("num_workers", 0)),
        persistent_workers=bool(runtime.get("persistent_workers", False)),
        prefetch_factor=int(runtime.get("prefetch_factor", 2)),
    )
    logits = predict_pulse_logits(
        result.model, loader, mixed_precision=bool(runtime.get("mixed_precision", False)),
    )
    payload = {
        "logits": logits, "history": result.history,
        "standardizer": {
            "mean": standardizer.mean, "std": standardizer.std,
            "n_values": standardizer.n_values, "best_epoch": result.best_epoch,
        },
    }
    shutdown_pulse_loader(loader)
    del result, normalized, loader
    _cleanup_cuda()
    return payload


def _run_candidate(
    config: dict[str, Any], candidate_id: str, candidate: dict[str, Any],
    arrays: dict[str, np.ndarray], manifest: pd.DataFrame, results_root: Path, logger,
) -> dict[str, Any]:
    started = time.perf_counter()
    temporal, cwt, temporal_mask, cwt_mask, representation_spec = _candidate_views(
        candidate_id, candidate, arrays,
    )
    labels_all = manifest["target"].astype(np.int64).to_numpy()
    split = manifest["split"].astype(str).to_numpy()
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    labels_train = labels_all[train_indices]
    labels_validation = labels_all[validation_indices]
    train_ids = manifest.loc[train_indices, "signal_id"].astype(str).to_numpy()
    validation_ids = manifest.loc[validation_indices, "signal_id"].astype(str).to_numpy()
    groups_train = manifest.loc[train_indices, "id_measurement"].astype(str).to_numpy()
    folds = manifest.loc[train_indices, "oof_fold"].astype(np.int64).to_numpy()
    temporal_train = IndexedPulseView(temporal, train_indices)
    cwt_train = IndexedPulseView(cwt, train_indices)
    mask_train = IndexedPulseView(temporal_mask, train_indices)
    runtime = config.get("runtime", {})
    expert_config = config["experts"]
    cycle_consistency = False
    logger.info(
        "candidate started: %s; pulses=%s, cwt_scales=%s, "
        "cycle_consistency=disabled (phase-pair diagnostic not proven)"
        % (candidate_id, representation_spec["n_pulses_per_half"], representation_spec["cwt_scales"])
    )
    records: list[dict[str, Any]] = []
    for seed in V5_DEVELOPMENT_SEEDS:
        seed_started = time.perf_counter()
        output = results_root / "development" / candidate_id / f"seed-{seed}"
        completed = _completed_seed_record(
            output, candidate_id, int(seed), train_ids, validation_ids,
            labels_train, labels_validation,
        )
        if completed is not None:
            records.append(completed)
            logger.info(f"candidate={candidate_id} seed={seed}: resumed from verified artifacts")
            continue
        logger.info(f"candidate={candidate_id} seed={seed}: temporal OOF started")
        temporal_oof = _train_oof(
            temporal_train, mask_train, labels_train, groups_train, folds,
            representation="temporal", seed=int(seed), epochs=int(expert_config["epochs"]["temporal"]),
            batch_size=int(expert_config["batch_size"]),
            inference_batch_size=int(expert_config["inference_batch_size"]),
            cycle_consistency=cycle_consistency, runtime=runtime, logger=logger,
        )
        logger.info(f"candidate={candidate_id} seed={seed}: CWT OOF started")
        cwt_oof = _train_oof(
            cwt_train, mask_train, labels_train, groups_train, folds,
            representation="cwt", seed=int(seed) + 100,
            epochs=int(expert_config["epochs"]["cwt"]),
            batch_size=int(expert_config["batch_size"]),
            inference_batch_size=int(expert_config["inference_batch_size"]),
            cycle_consistency=False, runtime=runtime, logger=logger,
        )
        logger.info(f"candidate={candidate_id} seed={seed}: validation models started")
        temporal_validation = _train_validation_model(
            temporal, temporal_mask, labels_all, train_indices, validation_indices, folds,
            representation="temporal", seed=int(seed),
            epochs=int(expert_config["epochs"]["temporal"]),
            batch_size=int(expert_config["batch_size"]),
            inference_batch_size=int(expert_config["inference_batch_size"]),
            cycle_consistency=cycle_consistency, runtime=runtime, logger=logger,
        )
        cwt_validation = _train_validation_model(
            cwt, cwt_mask, labels_all, train_indices, validation_indices, folds,
            representation="cwt", seed=int(seed) + 100,
            epochs=int(expert_config["epochs"]["cwt"]),
            batch_size=int(expert_config["batch_size"]),
            inference_batch_size=int(expert_config["inference_batch_size"]),
            cycle_consistency=False, runtime=runtime, logger=logger,
        )
        temporal_oof_probability = sigmoid(temporal_oof["logits"])
        cwt_oof_probability = sigmoid(cwt_oof["logits"])
        temporal_threshold = select_threshold(labels_train, temporal_oof_probability)
        cwt_threshold = select_threshold(labels_train, cwt_oof_probability)
        temporal_validation_probability = sigmoid(temporal_validation["logits"])
        cwt_validation_probability = sigmoid(cwt_validation["logits"])
        temporal_validation_mcc = float(
            matthews_corrcoef(labels_validation, temporal_validation_probability >= temporal_threshold)
        )
        cwt_validation_mcc = float(
            matthews_corrcoef(labels_validation, cwt_validation_probability >= cwt_threshold)
        )
        output.mkdir(parents=True, exist_ok=True)
        oof_rows = prediction_frame(
            train_ids, labels_train,
            dataset_id="engineering-vsb-power-line-fault-detection", split="development_oof",
            seed=int(seed), config_version=config["config_version"],
            temporal=temporal_oof_probability, cwt=cwt_oof_probability,
        )
        validation_rows = prediction_frame(
            validation_ids, labels_validation,
            dataset_id="engineering-vsb-power-line-fault-detection", split="development_validation",
            seed=int(seed), config_version=config["config_version"],
            temporal=temporal_validation_probability, cwt=cwt_validation_probability,
        )
        write_predictions(oof_rows, output / "oof_predictions.parquet")
        write_predictions(validation_rows, output / "validation_predictions.parquet")
        write_json(
            {
                "candidate_id": candidate_id, "seed": int(seed),
                "representation": representation_spec,
                "cycle_consistency_enabled": False,
                "temporal": {"oof": temporal_oof["standardizers"], "validation": temporal_validation["standardizer"]},
                "cwt": {"oof": cwt_oof["standardizers"], "validation": cwt_validation["standardizer"]},
                "temporal_threshold_from_oof": temporal_threshold,
                "cwt_threshold_from_oof": cwt_threshold,
                "temporal_history": temporal_validation["history"],
                "cwt_history": cwt_validation["history"],
            },
            output / "training_record.json",
        )
        record = {
            "seed": int(seed),
            "temporal_oof_mcc": float(matthews_corrcoef(labels_train, temporal_oof_probability >= temporal_threshold)),
            "cwt_oof_mcc": float(matthews_corrcoef(labels_train, cwt_oof_probability >= cwt_threshold)),
            "temporal_validation_mcc": temporal_validation_mcc,
            "cwt_validation_mcc": cwt_validation_mcc,
            "best_individual_mcc": max(temporal_validation_mcc, cwt_validation_mcc),
            "minimum_expert_mcc": min(temporal_validation_mcc, cwt_validation_mcc),
            "temporal_threshold_from_oof": float(temporal_threshold),
            "cwt_threshold_from_oof": float(cwt_threshold),
            "elapsed_seconds": float(time.perf_counter() - seed_started),
        }
        records.append(record)
        logger.info(
            "candidate=%s seed=%s: finished; temporal_val_mcc=%.4f, "
            "cwt_val_mcc=%.4f, elapsed=%.1fs"
            % (candidate_id, seed, temporal_validation_mcc, cwt_validation_mcc, record["elapsed_seconds"])
        )
        del temporal_oof, cwt_oof, temporal_validation, cwt_validation
        _cleanup_cuda()
    logger.info(f"candidate finished: {candidate_id}; elapsed={time.perf_counter() - started:.1f}s")
    return {
        "compute_cost": float(candidate.get("compute_cost", 0.0)),
        "seed_records": records,
        "representation": representation_spec,
    }


def run_development(
    config_path: Path,
    cache_root: Path,
    records_path: Path,
    results_root: Path,
    selected_candidates: list[str] | None,
    num_workers: int,
    persistent_workers: bool,
    logger,
) -> dict[str, Any]:
    started = time.perf_counter()
    logger.info(f"V5 development training started: config={config_path}, cache_root={cache_root}")
    config = load_experiment_config(config_path)
    validate_v5_config(config)
    assert_v5_holdouts_closed(config)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers greater than zero")
    runtime = dict(config.get("runtime", {}))
    runtime["num_workers"] = int(num_workers)
    runtime["persistent_workers"] = bool(persistent_workers)
    config["runtime"] = runtime
    logger.info(
        "runtime loader policy: num_workers=%s, persistent_workers=%s"
        % (runtime["num_workers"], runtime["persistent_workers"])
    )
    device = require_cuda()
    arrays, cache_metadata = _load_caches(cache_root, logger)
    manifest = _load_manifest(cache_root, arrays)
    train_count = int((manifest["split"] == "train").sum())
    validation_count = int((manifest["split"] == "validation").sum())
    logger.info(
        "development data loaded: parents=%s, train=%s, validation=%s, device=%s"
        % (len(manifest), train_count, validation_count, device)
    )
    candidates = config["selection"]["model_candidates"]
    requested = selected_candidates or list(V5_CANDIDATE_IDS)
    unknown = set(requested) - set(V5_CANDIDATE_IDS)
    if unknown:
        raise ValueError(f"Unknown V5 candidates: {sorted(unknown)}")
    unavailable: dict[str, str] = {}
    candidate_results: dict[str, Any] = {}
    for candidate_id in requested:
        try:
            candidate_results[candidate_id] = _run_candidate(
                config, candidate_id, candidates[candidate_id], arrays, manifest, results_root, logger,
            )
        except CacheUnavailable as exc:
            unavailable[candidate_id] = str(exc)
            logger.warning(f"candidate unavailable: {candidate_id}: {exc}")
    if not candidate_results:
        raise RuntimeError("No V5 development candidate could be trained")
    summary = {
        "protocol_version": "v5-pulse-aware-v1",
        "config_version": config["config_version"],
        "holdouts_opened": False,
        "device": runtime_info().as_dict(),
        "runtime": dict(config["runtime"]),
        "cache_root": str(cache_root),
        "cache_fingerprints": {
            name: metadata["fingerprint"] for name, metadata in cache_metadata.items()
        },
        "development_manifest": str(cache_root / "v5_development_parent_manifest.csv"),
        "candidates": candidate_results,
        "unavailable_candidates": unavailable,
        "cycle_consistency_enabled": False,
        "cycle_consistency_status": "disabled_until_phase_alignment_is_proven",
        "started_at_epoch": time.time(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(summary, records_path)
    logger.info(f"V5 development records written: {records_path}")
    logger.info("V5 development training finished in %.1fs" % summary["elapsed_seconds"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/v5-pulse-aware"))
    parser.add_argument("--results-root", type=Path, default=Path("results/runs/v5-pulse-aware"))
    parser.add_argument(
        "--records", type=Path,
        default=Path("results/runs/v5-pulse-aware/development_records.json"),
    )
    parser.add_argument(
        "--candidate", action="append", dest="candidates",
        help="Repeat to restrict the run; defaults to the three cache-backed pulse candidates.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers per short-lived fit; default 0 avoids file-descriptor leaks.",
    )
    parser.add_argument(
        "--persistent-workers", action="store_true",
        help="Opt in only with --num-workers greater than zero after a resource-equivalence check.",
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    logger = configure_progress_logging("v5.development", args.log_file)
    summary = run_development(
        args.config, args.cache_root, args.records, args.results_root,
        args.candidates, args.num_workers, args.persistent_workers, logger,
    )
    print(args.records)
    print("trained_candidates=" + ",".join(summary["candidates"]))
    if summary["unavailable_candidates"]:
        print("unavailable_candidates=" + ",".join(summary["unavailable_candidates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
