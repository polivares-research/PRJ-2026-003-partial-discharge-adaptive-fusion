"""Prepare V5 VSB pulse and CWT caches from development-labelled signals only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import (
    VSB_DATASET_ID,
    dataset_provenance,
    iter_vsb_signal_batches,
    load_vsb_metadata,
    resolve_dataset,
)
from partial_discharge_adaptive_fusion.pulse import PulsePolicy, prepare_pulse_bag
from partial_discharge_adaptive_fusion.pulse_cache import (
    CacheMetadata,
    cache_fingerprint,
    canonical_json,
    runtime_library_versions,
)
from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.splits import vsb_grouped_manifest
from partial_discharge_adaptive_fusion.v5_protocol import validate_v5_config
from partial_discharge_adaptive_fusion.representations import cwt_log_power_batch


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


def _write_cache(
    path: Path,
    metadata: CacheMetadata,
    writer,
) -> dict[str, Any]:
    temporary, values = _begin(path, metadata.shape, metadata.dtype)
    try:
        writer(values)
        values.flush()
        del values
        return _finish(path, temporary, metadata)
    except Exception:
        del values
        if temporary.exists():
            temporary.unlink()
        raise


def prepare_cache(config_path: Path, raw_root: Path | None, cache_root: Path, io_batch_size: int) -> dict[str, Any]:
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
    policy = PulsePolicy(n_pulses_per_half=257)
    source_fingerprint = cache_fingerprint({"dataset": dataset_provenance(dataset), "metadata_rows": frame.to_dict("records")})
    split_hash = cache_fingerprint(selected.to_dict("records"))
    common = {
        "dataset_id": VSB_DATASET_ID,
        "dataset_version": "2018-kaggle-snapshot",
        "source_fingerprint": source_fingerprint,
        "split_manifest_hash": split_hash,
        "pulse_policy": policy.as_dict(),
        "preprocessing_version": policy.version,
        "library_versions": runtime_library_versions(),
    }
    temporal_path = cache_root / "pulse_temporal_np257.dat"
    cwt_path = cache_root / "pulse_cwt_np257_scales64.dat"
    valid_path = cache_root / "pulse_valid_np257.dat"
    peaks_path = cache_root / "pulse_peak_indices_np257.dat"
    temporal_meta = CacheMetadata(
        **common, transform={"representation": "raw_temporal", "length": policy.temporal_length},
        dtype="float16", shape=(n_signals, 2, 257, 1, policy.temporal_length),
    )
    cwt_meta = CacheMetadata(
        **common, transform={"representation": "cwt_log_power", "scales": 64, "time_bins": 128, "morlet_w0": 6.0},
        dtype="float16", shape=(n_signals, 2, 257, 1, 64, 128),
    )
    mask_meta = CacheMetadata(
        **common, transform={"representation": "validity_mask"}, dtype="bool", shape=(n_signals, 2, 257),
    )
    peak_meta = CacheMetadata(
        **common, transform={"representation": "peak_indices"}, dtype="int64", shape=(n_signals, 2, 257),
    )
    scales = np.geomspace(1.5, 64.0, 64)

    def writer(values: np.memmap) -> None:
        positions = {str(signal_id): index for index, signal_id in enumerate(frame["signal_id"].astype(str))}
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            for row_index, (signal_id, signal) in enumerate(zip(batch.sample_id, batch.signal)):
                bag = prepare_pulse_bag(signal, policy)
                values[positions[str(signal_id)]] = bag.temporal[:, :, None, :].astype(np.float16)

    def cwt_writer(values: np.memmap) -> None:
        positions = {str(signal_id): index for index, signal_id in enumerate(frame["signal_id"].astype(str))}
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            temporal_segments: list[np.ndarray] = []
            ids: list[str] = []
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                bag = prepare_pulse_bag(signal, policy)
                temporal_segments.append(bag.cwt_segments.reshape(-1, policy.cwt_length))
                ids.append(str(signal_id))
            transformed = cwt_log_power_batch(
                np.concatenate(temporal_segments, axis=0), scales=scales, time_bins=128,
            ).reshape(len(ids), 2, 257, 64, 128)
            for row_index, signal_id in enumerate(ids):
                values[positions[signal_id]] = transformed[row_index, :, :, None].astype(np.float16)

    def mask_writer(values: np.memmap) -> None:
        positions = {str(signal_id): index for index, signal_id in enumerate(frame["signal_id"].astype(str))}
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                values[positions[str(signal_id)]] = prepare_pulse_bag(signal, policy).valid_mask

    def peak_writer(values: np.memmap) -> None:
        positions = {str(signal_id): index for index, signal_id in enumerate(frame["signal_id"].astype(str))}
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            for signal_id, signal in zip(batch.sample_id, batch.signal):
                values[positions[str(signal_id)]] = prepare_pulse_bag(signal, policy).peak_indices

    artifacts = {
        "temporal": _write_cache(temporal_path, temporal_meta, writer),
        "cwt": _write_cache(cwt_path, cwt_meta, cwt_writer),
        "valid_mask": _write_cache(valid_path, mask_meta, mask_writer),
        "peak_indices": _write_cache(peaks_path, peak_meta, peak_writer),
    }
    manifest_path = cache_root / "v5_development_parent_manifest.csv"
    frame.assign(split=selected["split"].to_numpy(), oof_fold=selected["oof_fold"].to_numpy()).to_csv(manifest_path, index=False)
    return {
        "cache_contract_version": "v5-pulse-cache-v1",
        "dataset": dataset_provenance(dataset),
        "n_signals": n_signals,
        "split_values": selected["split"].value_counts().to_dict(),
        "artifacts": artifacts,
        "parent_manifest": str(manifest_path),
        "holdouts_opened": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/v5-pulse-aware"))
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--summary", type=Path, default=Path("reports/metrics/v5-pulse-aware/cache_preparation.json"))
    args = parser.parse_args(argv)
    summary = prepare_cache(args.config, args.raw_root, args.cache_root, args.io_batch_size)
    write_json(summary, args.summary)
    print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
