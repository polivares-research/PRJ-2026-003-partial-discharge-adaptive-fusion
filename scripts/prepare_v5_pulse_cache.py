"""Prepare V5 VSB pulse and CWT caches from development-labelled signals only."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import (
    VSB_DATASET_ID, dataset_provenance, iter_vsb_signal_batches, load_vsb_metadata, resolve_dataset,
)
from partial_discharge_adaptive_fusion.logging_utils import configure_progress_logging, log_progress
from partial_discharge_adaptive_fusion.pulse import PulsePolicy, prepare_pulse_bag
from partial_discharge_adaptive_fusion.pulse_cache import CacheMetadata, cache_fingerprint, canonical_json, runtime_library_versions
from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.representations import cwt_log_power_batch
from partial_discharge_adaptive_fusion.splits import vsb_grouped_manifest
from partial_discharge_adaptive_fusion.v5_protocol import validate_v5_config


def _begin(path: Path, shape: tuple[int, ...], dtype: str) -> tuple[Path, np.memmap]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    if temporary.exists():
        temporary.unlink()
    return temporary, np.memmap(temporary, mode="w+", dtype=dtype, shape=shape)


def _finish(path: Path, temporary: Path, metadata: CacheMetadata) -> dict[str, Any]:
    payload = metadata.as_dict()
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".json").write_text(canonical_json(payload) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".complete").write_text(payload["fingerprint"] + "\n", encoding="utf-8")
    return payload


def _write_cache(path: Path, metadata: CacheMetadata, writer, logger, stage: str) -> dict[str, Any]:
    temporary, values = _begin(path, metadata.shape, metadata.dtype)
    started = time.perf_counter()
    try:
        writer(values, logger, stage, started)
        values.flush()
        del values
        payload = _finish(path, temporary, metadata)
        logger.info(f"{stage}: cache completed: {path} ({path.stat().st_size / 1024**3:.2f} GiB)")
        return payload
    except Exception:
        del values
        if temporary.exists():
            temporary.unlink()
        logger.exception(f"{stage}: cache failed")
        raise


def prepare_cache(config_path: Path, raw_root: Path | None, cache_root: Path, io_batch_size: int, logger) -> dict[str, Any]:
    logger.info(f"V5 cache preparation started: config={config_path}, io_batch_size={io_batch_size}")
    config = load_experiment_config(config_path)
    validate_v5_config(config)
    dataset = resolve_dataset(VSB_DATASET_ID, "2018-kaggle-snapshot", raw_root=raw_data_root(raw_root))
    metadata = load_vsb_metadata(dataset)
    manifest = vsb_grouped_manifest(
        metadata, dataset_id=VSB_DATASET_ID, dataset_version="2018-kaggle-snapshot",
        split_seed=config["splits"]["split_seed"], oof_splits=config["splits"]["oof_folds"],
    )
    selected = manifest.frame[manifest.frame["split"].isin(["train", "validation"])].reset_index(drop=True)
    by_id = metadata.set_index("signal_id", drop=False)
    frame = by_id.loc[selected["sample_id"].astype(str)].reset_index(drop=True)
    n_signals = len(frame)
    logger.info(f"development parent signals selected: {n_signals} (train+validation only)")
    policy = PulsePolicy(n_pulses_per_half=257)
    source_fingerprint = cache_fingerprint({"dataset": dataset_provenance(dataset), "metadata_rows": frame.to_dict("records")})
    split_hash = cache_fingerprint(selected.to_dict("records"))
    common = {
        "dataset_id": VSB_DATASET_ID, "dataset_version": "2018-kaggle-snapshot",
        "source_fingerprint": source_fingerprint, "split_manifest_hash": split_hash,
        "pulse_policy": policy.as_dict(), "preprocessing_version": policy.version,
        "library_versions": runtime_library_versions(),
    }
    temporal_path = cache_root / "pulse_temporal_np257.dat"
    cwt_path = cache_root / "pulse_cwt_np257_scales64.dat"
    valid_path = cache_root / "pulse_valid_np257.dat"
    peaks_path = cache_root / "pulse_peak_indices_np257.dat"
    temporal_meta = CacheMetadata(**common, transform={"representation": "raw_temporal", "length": policy.temporal_length}, dtype="float16", shape=(n_signals, 2, 257, 1, policy.temporal_length))
    cwt_meta = CacheMetadata(**common, transform={"representation": "cwt_log_power", "scales": 64, "time_bins": 128, "morlet_w0": 6.0}, dtype="float16", shape=(n_signals, 2, 257, 1, 64, 128))
    mask_meta = CacheMetadata(**common, transform={"representation": "validity_mask"}, dtype="bool", shape=(n_signals, 2, 257))
    peak_meta = CacheMetadata(**common, transform={"representation": "peak_indices"}, dtype="int64", shape=(n_signals, 2, 257))
    scales = np.geomspace(1.5, 64.0, 64)
    positions = {str(signal_id): index for index, signal_id in enumerate(frame["signal_id"].astype(str))}

    def batches(stage: str):
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            yield batch

    def writer_temporal(values, logger, stage, started):
        completed = 0
        for batch in batches(stage):
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                values[positions[str(signal_id)]] = prepare_pulse_bag(signal, policy).temporal[:, :, None, :].astype(np.float16)
            completed += len(batch.signal)
            log_progress(logger, stage, completed, n_signals, started, every=max(io_batch_size * 10, 1))
        log_progress(logger, stage, n_signals, n_signals, started, force=True)

    def writer_cwt(values, logger, stage, started):
        completed = 0
        for batch in batches(stage):
            segments, ids = [], []
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                segments.append(prepare_pulse_bag(signal, policy).cwt_segments.reshape(-1, policy.cwt_length))
                ids.append(str(signal_id))
            transformed = cwt_log_power_batch(np.concatenate(segments, axis=0), scales=scales, time_bins=128).reshape(len(ids), 2, 257, 64, 128)
            for row_index, signal_id in enumerate(ids):
                values[positions[signal_id]] = transformed[row_index, :, :, None].astype(np.float16)
            completed += len(ids)
            log_progress(logger, stage, completed, n_signals, started, every=max(io_batch_size * 10, 1))
        log_progress(logger, stage, n_signals, n_signals, started, force=True)

    def writer_mask(values, logger, stage, started):
        completed = 0
        for batch in batches(stage):
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                values[positions[str(signal_id)]] = prepare_pulse_bag(signal, policy).valid_mask
            completed += len(batch.signal)
            log_progress(logger, stage, completed, n_signals, started, every=max(io_batch_size * 10, 1))
        log_progress(logger, stage, n_signals, n_signals, started, force=True)

    def writer_peaks(values, logger, stage, started):
        completed = 0
        for batch in batches(stage):
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                values[positions[str(signal_id)]] = prepare_pulse_bag(signal, policy).peak_indices
            completed += len(batch.signal)
            log_progress(logger, stage, completed, n_signals, started, every=max(io_batch_size * 10, 1))
        log_progress(logger, stage, n_signals, n_signals, started, force=True)

    artifacts = {
        "temporal": _write_cache(temporal_path, temporal_meta, writer_temporal, logger, "stage 1/4 temporal pulses"),
        "cwt": _write_cache(cwt_path, cwt_meta, writer_cwt, logger, "stage 2/4 CWT pulses"),
        "valid_mask": _write_cache(valid_path, mask_meta, writer_mask, logger, "stage 3/4 validity masks"),
        "peak_indices": _write_cache(peaks_path, peak_meta, writer_peaks, logger, "stage 4/4 peak indices"),
    }
    manifest_path = cache_root / "v5_development_parent_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(split=selected["split"].to_numpy(), oof_fold=selected["oof_fold"].to_numpy()).to_csv(manifest_path, index=False)
    logger.info(f"V5 cache preparation finished: manifest={manifest_path}")
    return {"cache_contract_version": "v5-pulse-cache-v1", "dataset": dataset_provenance(dataset), "n_signals": n_signals, "split_values": selected["split"].value_counts().to_dict(), "artifacts": artifacts, "parent_manifest": str(manifest_path), "holdouts_opened": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/v5-pulse-aware"))
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--summary", type=Path, default=Path("reports/metrics/v5-pulse-aware/cache_preparation.json"))
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    logger = configure_progress_logging("v5.cache", args.log_file)
    summary = prepare_cache(args.config, args.raw_root, args.cache_root, args.io_batch_size, logger)
    write_json(summary, args.summary)
    logger.info(f"cache summary written: {args.summary}")
    print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
