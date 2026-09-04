"""Run the frozen two-expert VSB stage with bounded Parquet streaming.

The neural batch size comes from the frozen configuration. ``--io-batch-size``
only controls how many columnar signals are read while building caches and
does not change the scientific training protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from partial_discharge_adaptive_fusion.cache import CacheRecord
from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import (
    iter_vsb_signal_batches,
    load_vsb_metadata,
    resolve_dataset,
)
from partial_discharge_adaptive_fusion.experiments import train_two_experts_for_seed
from partial_discharge_adaptive_fusion.pipeline import require_training_ready, run_provenance
from partial_discharge_adaptive_fusion.reporting import prediction_frame, write_json, write_predictions
from partial_discharge_adaptive_fusion.representations import (
    cwt_input,
    cwt_log_power_batch,
    temporal_input,
)
from partial_discharge_adaptive_fusion.splits import assert_complete_oof, vsb_grouped_manifest
from partial_discharge_adaptive_fusion.streaming import fit_stream_standardizer, write_stream_cache


DATASET_ID = "engineering-vsb-power-line-fault-detection"
DATASET_VERSION = "2018-kaggle-snapshot"
SCALES = np.geomspace(1.5, 64.0, 32).astype(np.float64)
SIGNAL_LENGTH = 800_000
CWT_SHAPE = (32, 120)


def _raw_factory(dataset, frame: pd.DataFrame, io_batch_size: int):
    def factory():
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            yield batch.signal
    return factory


def _cwt_factory(dataset, frame: pd.DataFrame, io_batch_size: int):
    raw_factory = _raw_factory(dataset, frame, io_batch_size)

    def factory():
        for values in raw_factory():
            yield cwt_log_power_batch(values, scales=SCALES, time_bins=CWT_SHAPE[1])
    return factory


def _cached_record(path: Path, expected_shape: tuple[int, ...]) -> CacheRecord | None:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(values.shape) != expected_shape or metadata.get("shape") != list(expected_shape):
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return CacheRecord(path, metadata_path, metadata)


def _write_or_reuse(
    *,
    path: Path,
    expected_shape: tuple[int, ...],
    batches,
    transform,
    dtype: str,
    metadata: dict[str, object],
    label: str,
) -> CacheRecord:
    existing = _cached_record(path, expected_shape)
    if existing is not None:
        print(f"VSB cache: reusing {label}", flush=True)
        return existing
    print(f"VSB cache: writing {label}", flush=True)
    write_stream_cache(
        batches, n_samples=expected_shape[0], sample_shape=expected_shape[1:],
        transform=transform, destination=path, dtype=dtype, metadata=metadata,
    )
    return _cached_record(path, expected_shape) or CacheRecord(
        path, path.with_suffix(".json"), metadata,
    )


def _prepare_split_caches(
    dataset,
    frames: dict[str, pd.DataFrame],
    standardizers,
    cache_root: Path,
    io_batch_size: int,
) -> dict[str, dict[str, CacheRecord]]:
    raw_standardizer, cwt_standardizer = standardizers
    prepared: dict[str, dict[str, CacheRecord]] = {}
    for split, frame in frames.items():
        n_samples = len(frame)
        temporal_path = cache_root / f"vsb_temporal_{split}.npy"
        cwt_path = cache_root / f"vsb_cwt_{split}.npy"
        common_metadata = {
            "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
            "split": split, "signal_length": SIGNAL_LENGTH,
            "standardized_on": "train_split_only",
        }
        temporal = _write_or_reuse(
            path=temporal_path, expected_shape=(n_samples, 1, SIGNAL_LENGTH),
            batches=_raw_factory(dataset, frame, io_batch_size),
            transform=lambda values: temporal_input(values, raw_standardizer),
            dtype="float32", metadata={**common_metadata, "representation": "temporal"},
            label=f"temporal/{split}",
        )
        cwt = _write_or_reuse(
            path=cwt_path, expected_shape=(n_samples, 1, *CWT_SHAPE),
            batches=_cwt_factory(dataset, frame, io_batch_size),
            transform=lambda values: cwt_input(values, cwt_standardizer),
            dtype="float16", metadata={
                **common_metadata, "representation": "cwt", "scales": SCALES.tolist(),
                "time_bins": CWT_SHAPE[1], "morlet_w0": 6.0, "floor": -10.0,
            },
            label=f"cwt/{split}",
        )
        prepared[split] = {"temporal": temporal, "cwt": cwt}
        gc.collect()
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path, help="Optional local raw-data root.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/vsb"))
    parser.add_argument("--output-root", type=Path, default=Path("results/runs/vsb"))
    args = parser.parse_args(argv)
    if args.io_batch_size < 1:
        parser.error("--io-batch-size must be positive")
    config = require_training_ready(args.config, dataset_id=DATASET_ID, confirmatory=True)
    dataset = resolve_dataset(DATASET_ID, DATASET_VERSION, raw_root=raw_data_root(args.raw_root))
    metadata = load_vsb_metadata(dataset)
    manifest = vsb_grouped_manifest(
        metadata, dataset_id=dataset.dataset_id, dataset_version=dataset.version,
        split_seed=config["splits"]["split_seed"], oof_splits=config["splits"]["oof_folds"],
    )
    assert_complete_oof(manifest)
    manifest_path = Path(config["outputs"]["manifest_root"]) / "vsb_manifest.csv"
    manifest.write_csv(manifest_path)
    metadata_by_id = metadata.set_index("signal_id", drop=False)

    def frame_for(split: str) -> pd.DataFrame:
        ids = manifest.frame.loc[manifest.frame["split"] == split, "sample_id"].astype(str)
        frame = metadata_by_id.reindex(ids).reset_index(drop=True)
        if frame["signal_id"].isna().any():
            raise RuntimeError(f"VSB metadata alignment failed for split {split}.")
        return frame

    frames = {split: frame_for(split) for split in ("train", "validation", "test")}
    print("VSB split sizes:", {key: len(value) for key, value in frames.items()}, flush=True)
    print("VSB standardizers: fitting on train split only", flush=True)
    raw_standardizer = fit_stream_standardizer(
        _raw_factory(dataset, frames["train"], args.io_batch_size),
        sample_shape=(SIGNAL_LENGTH,),
    )
    cwt_standardizer = fit_stream_standardizer(
        _cwt_factory(dataset, frames["train"], args.io_batch_size),
        sample_shape=CWT_SHAPE,
    )
    prepared = _prepare_split_caches(
        dataset, frames, (raw_standardizer, cwt_standardizer), args.cache_root, args.io_batch_size,
    )
    train_manifest = manifest.frame.loc[manifest.frame["split"] == "train"].reset_index(drop=True)
    folds = train_manifest["oof_fold"].to_numpy(np.int64)
    groups = train_manifest["group_id"].astype(str).to_numpy()
    arrays = {
        "train_temporal": np.load(prepared["train"]["temporal"].array_path, mmap_mode="r"),
        "train_cwt": np.load(prepared["train"]["cwt"].array_path, mmap_mode="r"),
        "validation_temporal": np.load(prepared["validation"]["temporal"].array_path, mmap_mode="r"),
        "validation_cwt": np.load(prepared["validation"]["cwt"].array_path, mmap_mode="r"),
        "test_temporal": np.load(prepared["test"]["temporal"].array_path, mmap_mode="r"),
        "test_cwt": np.load(prepared["test"]["cwt"].array_path, mmap_mode="r"),
    }
    for seed in args.seeds:
        seed_root = args.output_root / f"seed-{seed}"
        prediction_path = seed_root / "expert_predictions.parquet"
        if prediction_path.exists() and (seed_root / "expert_run.json").exists():
            print(f"completed VSB expert seed {seed} already exists: {seed_root}", flush=True)
            continue
        print(f"VSB experts: seed {seed}", flush=True)
        output = train_two_experts_for_seed(
            **arrays,
            train_labels=frames["train"]["target"].to_numpy(np.int64),
            oof_folds=folds,
            validation_labels=frames["validation"]["target"].to_numpy(np.int64),
            test_labels=frames["test"]["target"].to_numpy(np.int64),
            seed=seed,
            groups=groups,
            temporal_epochs=config["experts"]["epochs"]["temporal"],
            cwt_epochs=config["experts"]["epochs"]["cwt"],
            batch_size=config["experts"]["batch_size"],
            inference_batch_size=config["experts"].get(
                "inference_batch_size", max(config["experts"]["batch_size"], 512),
            ),
            gradient_accumulation_steps=config["experts"].get("gradient_accumulation_steps", 1),
            imbalance_strategy="fold_local_pos_weight",
        )
        sigmoid = lambda values: 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))
        rows = [
            prediction_frame(
                frames["train"]["signal_id"].to_numpy(), frames["train"]["target"].to_numpy(),
                dataset_id=DATASET_ID, split="train_oof", seed=seed, config_version=config["config_version"],
                temporal=sigmoid(output.temporal_oof_logits), cwt=sigmoid(output.cwt_oof_logits),
            ),
            prediction_frame(
                frames["validation"]["signal_id"].to_numpy(), frames["validation"]["target"].to_numpy(),
                dataset_id=DATASET_ID, split="validation", seed=seed, config_version=config["config_version"],
                temporal=sigmoid(output.temporal_validation_logits), cwt=sigmoid(output.cwt_validation_logits),
            ),
            prediction_frame(
                frames["test"]["signal_id"].to_numpy(), frames["test"]["target"].to_numpy(),
                dataset_id=DATASET_ID, split="test_grouped_holdout", seed=seed, config_version=config["config_version"],
                temporal=sigmoid(output.temporal_test_logits), cwt=sigmoid(output.cwt_test_logits),
            ),
        ]
        seed_root.mkdir(parents=True, exist_ok=True)
        write_predictions(pd.concat(rows, ignore_index=True), prediction_path)
        write_json({
            "provenance": run_provenance(
                config_path=args.config, dataset_id=DATASET_ID, split="grouped_train_validation_test",
                seed=seed, imbalance_strategy="fold_local_pos_weight",
            ),
            "io_batch_size": args.io_batch_size,
            "neural_batch_size": config["experts"]["batch_size"],
            "inference_batch_size": config["experts"].get(
                "inference_batch_size", max(config["experts"]["batch_size"], 512),
            ),
            "gradient_accumulation_steps": config["experts"].get("gradient_accumulation_steps", 1),
            "temporal_signal_length": SIGNAL_LENGTH,
            "oof_coverage": int(np.isfinite(output.temporal_oof_logits).sum()),
        }, seed_root / "expert_run.json")
        print(f"completed VSB expert seed {seed}: {seed_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
