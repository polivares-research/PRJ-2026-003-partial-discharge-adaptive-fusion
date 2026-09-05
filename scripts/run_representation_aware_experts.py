"""Run the frozen representation-aware expert stage for MATLAB or VSB.

This runner is intentionally separate from the historical native VSB runner.
It writes only under ``results/runs/v3-windowed`` and ``results/cache/v3-windowed``.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import (
    iter_vsb_signal_batches, load_mat_confirmatory_partition, load_mat_partition,
    load_vsb_metadata, resolve_dataset,
)
from partial_discharge_adaptive_fusion.experiments import train_two_experts_for_seed
from partial_discharge_adaptive_fusion.pipeline import require_training_ready, run_provenance
from partial_discharge_adaptive_fusion.protocol import MATLAB_DATASET_ID, VSB_DATASET_ID
from partial_discharge_adaptive_fusion.reporting import prediction_frame, write_json, write_predictions
from partial_discharge_adaptive_fusion.splits import assert_complete_oof, matlab_manifest, vsb_grouped_manifest
from partial_discharge_adaptive_fusion.windowed import (
    DEFAULT_CWT_TIME_BINS, DEFAULT_MORLET_W0, fit_window_standardizer,
    write_windowed_cache, windowed_cache_parameters,
)
from partial_discharge_adaptive_fusion.windowing import SUMMARY_NAMES, WindowSpec, validate_parent_assignments


DATASET_VERSIONS = {MATLAB_DATASET_ID: "v1", VSB_DATASET_ID: "2018-kaggle-snapshot"}
CWT_SCALES = np.geomspace(1.5, 64.0, 32).astype(np.float64)


def _factory_from_array(values: np.ndarray):
    def factory():
        yield np.asarray(values, dtype=np.float32)
    return factory


def _factory_from_vsb(dataset, frame: pd.DataFrame, io_batch_size: int):
    def factory():
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            yield batch.signal
    return factory


def _policy_spec(config: dict, dataset_id: str) -> WindowSpec:
    policy = config["datasets"][dataset_id]["input_policy"]
    temporal = policy["temporal"]
    if dataset_id == MATLAB_DATASET_ID:
        return WindowSpec(window_length=400, stride=400, aggregation="max", top_k_fraction=0.10)
    if temporal.get("input") != "signal_windows":
        raise ValueError("Frozen VSB protocol does not contain the representation-aware window policy.")
    return WindowSpec(
        window_length=int(temporal["window_length"]), stride=int(temporal["stride"]),
        aggregation=str(temporal["aggregation"]), top_k_fraction=float(temporal.get("top_k_fraction", 0.10)),
    )


def _valid_cache(path: Path, expected_shape: tuple[int, ...], expected_parameters: dict[str, object]) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return tuple(values.shape) == expected_shape and metadata.get("shape") == list(expected_shape) and metadata.get("parameters") == expected_parameters


def _prepare_caches(
    *,
    dataset_id: str,
    dataset_version: str,
    dataset,
    frames: dict[str, pd.DataFrame],
    raw_factories: dict[str, object],
    signal_length: int,
    spec: WindowSpec,
    cache_root: Path,
    io_batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    del io_batch_size, dataset  # I/O batching is already encoded in each factory.
    temporal_standardizer = fit_window_standardizer(
        raw_factories["train"], spec, representation="temporal",
    )
    cwt_standardizer = fit_window_standardizer(
        raw_factories["train"], spec, representation="cwt", scales=CWT_SCALES,
        time_bins=DEFAULT_CWT_TIME_BINS, morlet_w0=DEFAULT_MORLET_W0,
    )
    root = cache_root / ("matlab" if dataset_id == MATLAB_DATASET_ID else "vsb")
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for split, frame in frames.items():
        split_root = root / str(spec.window_length)
        split_root.mkdir(parents=True, exist_ok=True)
        n_windows = spec.n_windows(signal_length)
        common = {
            "dataset_id": dataset_id, "dataset_version": dataset_version,
            "sampling_frequency_hz": 40_000_000.0 if dataset_id == VSB_DATASET_ID else None,
        }
        temporal_parameters = windowed_cache_parameters(
            **common, partition=split, spec=spec, representation="temporal", standardizer=temporal_standardizer,
            scales=CWT_SCALES, time_bins=DEFAULT_CWT_TIME_BINS, signal_length=signal_length,
        )
        cwt_parameters = windowed_cache_parameters(
            **common, partition=split, spec=spec, representation="cwt", standardizer=cwt_standardizer,
            scales=CWT_SCALES, time_bins=DEFAULT_CWT_TIME_BINS, morlet_w0=DEFAULT_MORLET_W0, signal_length=signal_length,
        )
        temporal_path = split_root / f"temporal-{split}.npy"
        cwt_path = split_root / f"cwt-{split}.npy"
        temporal_shape = (len(frame), n_windows, 1, spec.window_length)
        cwt_shape = (len(frame), n_windows, 1, len(CWT_SCALES), DEFAULT_CWT_TIME_BINS)
        if not _valid_cache(temporal_path, temporal_shape, temporal_parameters):
            write_windowed_cache(
                raw_factories[split], n_samples=len(frame), spec=spec, representation="temporal",
                standardizer=temporal_standardizer, destination=str(temporal_path),
                dataset_id=dataset_id, dataset_version=dataset_version, partition=split,
                scales=CWT_SCALES, time_bins=DEFAULT_CWT_TIME_BINS,
                sampling_frequency_hz=common["sampling_frequency_hz"], dtype="float16",
            )
        if not _valid_cache(cwt_path, cwt_shape, cwt_parameters):
            write_windowed_cache(
                raw_factories[split], n_samples=len(frame), spec=spec, representation="cwt",
                standardizer=cwt_standardizer, destination=str(cwt_path),
                dataset_id=dataset_id, dataset_version=dataset_version, partition=split,
                scales=CWT_SCALES, time_bins=DEFAULT_CWT_TIME_BINS, morlet_w0=DEFAULT_MORLET_W0,
                sampling_frequency_hz=common["sampling_frequency_hz"], dtype="float16",
            )
        temporal = np.load(temporal_path, mmap_mode="r", allow_pickle=False)
        cwt = np.load(cwt_path, mmap_mode="r", allow_pickle=False)
        if tuple(temporal.shape) != temporal_shape or tuple(cwt.shape) != cwt_shape:
            raise RuntimeError(f"Cache shape mismatch for {dataset_id}/{split}.")
        prepared[split] = {"temporal": temporal, "cwt": cwt}
        write_json({
            "dataset_id": dataset_id, "dataset_version": dataset_version, "split": split,
            "window": spec.as_dict(), "signal_length": signal_length,
            "temporal_standardizer_mean": temporal_standardizer.mean.tolist(),
            "temporal_standardizer_std": temporal_standardizer.std.tolist(),
            "cwt_standardizer_mean": cwt_standardizer.mean.tolist(),
            "cwt_standardizer_std": cwt_standardizer.std.tolist(),
            "preprocessing_version": "representation-aware-mi-v1",
        }, split_root / "standardizers.json")
        gc.collect()
    return prepared


def _summary_columns(prefix: str, summaries: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    if not summaries:
        return {}
    return {
        f"{prefix}_window_{name}": summaries[name]
        for name in SUMMARY_NAMES if name in summaries
    }


def _parent_columns(
    dataset_id: str,
    frame: pd.DataFrame,
    temporal_summaries: dict[str, np.ndarray] | None,
    cwt_summaries: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    columns = {
        **_summary_columns("temporal", temporal_summaries),
        **_summary_columns("cwt", cwt_summaries),
    }
    if dataset_id == VSB_DATASET_ID:
        columns["id_measurement"] = frame["id_measurement"].astype(str).to_numpy()
    return columns


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -40, 40)))


def _run_dataset(args, config: dict, dataset_id: str) -> None:
    dataset_version = DATASET_VERSIONS[dataset_id]
    dataset = resolve_dataset(dataset_id, dataset_version, raw_root=raw_data_root(args.raw_root))
    spec = _policy_spec(config, dataset_id)
    if dataset_id == MATLAB_DATASET_ID:
        loaded = {name: load_mat_partition(dataset, name) for name in ("Tr1.mat", "Va1.mat", "Te1.mat")}
        loaded["Te2.mat"] = load_mat_confirmatory_partition(dataset, config)
        manifest = matlab_manifest(
            {name: batch.label for name, batch in loaded.items()},
            dataset_id=dataset_id, dataset_version=dataset_version,
            split_seed=config["splits"]["split_seed"], oof_splits=config["splits"]["oof_folds"],
        )
        frames = {
            "train": loaded["Tr1.mat"].metadata, "validation": loaded["Va1.mat"].metadata,
            "test": loaded["Te2.mat"].metadata, "test_historical": loaded["Te1.mat"].metadata,
        }
        raw_factories = {
            "train": _factory_from_array(loaded["Tr1.mat"].signal),
            "validation": _factory_from_array(loaded["Va1.mat"].signal),
            "test": _factory_from_array(loaded["Te2.mat"].signal),
            "test_historical": _factory_from_array(loaded["Te1.mat"].signal),
        }
        signal_length = 400
        train_labels = loaded["Tr1.mat"].label
        validation_labels = loaded["Va1.mat"].label
        test_labels = loaded["Te2.mat"].label
        groups = None
        manifest_path = Path(config["outputs"]["manifest_root"]) / "matlab_manifest_v3_windowed.csv"
    else:
        metadata = load_vsb_metadata(dataset)
        manifest = vsb_grouped_manifest(
            metadata, dataset_id=dataset_id, dataset_version=dataset_version,
            split_seed=config["splits"]["split_seed"], oof_splits=config["splits"]["oof_folds"],
        )
        metadata_by_id = metadata.set_index("signal_id", drop=False)
        frames = {
            split: metadata_by_id.loc[
                manifest.frame.loc[manifest.frame["split"] == split, "sample_id"].astype(str)
            ].reset_index(drop=True)
            for split in ("train", "validation", "test")
        }
        # The split manifest uses the common ``sample_id`` name while the raw
        # VSB metadata calls the same parent signal key ``signal_id``. Keep
        # both names so reporting and raw-signal iteration remain aligned.
        for frame in frames.values():
            frame["sample_id"] = frame["signal_id"].astype(str)
        raw_factories = {
            split: _factory_from_vsb(dataset, frame, args.io_batch_size)
            for split, frame in frames.items()
        }
        signal_length = 800_000
        train_labels = frames["train"]["target"].to_numpy(np.int64)
        validation_labels = frames["validation"]["target"].to_numpy(np.int64)
        test_labels = frames["test"]["target"].to_numpy(np.int64)
        train_manifest = manifest.frame.loc[manifest.frame["split"] == "train"].reset_index(drop=True)
        groups = train_manifest["group_id"].astype(str).to_numpy()
        manifest_path = Path(config["outputs"]["manifest_root"]) / "vsb_manifest_v3_windowed.csv"
        combined = pd.concat([frames["train"], frames["validation"], frames["test"]], ignore_index=True)
        split_values = np.concatenate([
            np.repeat("train", len(frames["train"])),
            np.repeat("validation", len(frames["validation"])),
            np.repeat("test", len(frames["test"])),
        ])
        validate_parent_assignments(
            combined["signal_id"].to_numpy(), combined["id_measurement"].to_numpy(),
            split_values, n_windows=spec.n_windows(signal_length),
        )
    assert_complete_oof(manifest)
    manifest.write_csv(manifest_path)
    prepared = _prepare_caches(
        dataset_id=dataset_id, dataset_version=dataset_version, dataset=dataset,
        frames={key: frames[key] for key in ("train", "validation", "test", "test_historical") if key in frames},
        raw_factories=raw_factories, signal_length=signal_length, spec=spec,
        cache_root=Path("results/cache/v3-windowed"), io_batch_size=args.io_batch_size,
    )
    train_manifest = manifest.frame.loc[manifest.frame["split"] == "train"].reset_index(drop=True)
    folds = train_manifest["oof_fold"].to_numpy(np.int64)
    for seed in config["seeds"] if args.seeds is None else args.seeds:
        dataset_slug = "matlab" if dataset_id == MATLAB_DATASET_ID else "vsb"
        seed_root = Path(config["outputs"]["results_root"]) / dataset_slug / f"seed-{seed}"
        prediction_path = seed_root / "expert_predictions.parquet"
        if prediction_path.exists() and (seed_root / "expert_run.json").exists():
            print(f"reusing completed {dataset_slug} seed {seed}", flush=True)
            continue
        extra_tests = {}
        if dataset_id == MATLAB_DATASET_ID:
            extra_tests["Te1"] = (prepared["test_historical"]["temporal"], prepared["test_historical"]["cwt"])
        output = train_two_experts_for_seed(
            train_temporal=prepared["train"]["temporal"], train_cwt=prepared["train"]["cwt"],
            train_labels=train_labels, oof_folds=folds,
            validation_temporal=prepared["validation"]["temporal"], validation_cwt=prepared["validation"]["cwt"],
            validation_labels=validation_labels,
            test_temporal=prepared["test"]["temporal"], test_cwt=prepared["test"]["cwt"],
            test_labels=test_labels, additional_tests=extra_tests, seed=int(seed), groups=groups,
            temporal_epochs=config["experts"]["epochs"]["temporal"], cwt_epochs=config["experts"]["epochs"]["cwt"],
            batch_size=config["experts"]["batch_size"], inference_batch_size=config["experts"].get("inference_batch_size", 4),
            gradient_accumulation_steps=config["experts"].get("gradient_accumulation_steps", 1),
            imbalance_strategy="fold_local_pos_weight" if dataset_id == VSB_DATASET_ID else "none",
            multi_instance=True, aggregation=spec.aggregation, top_k_fraction=spec.top_k_fraction,
            instance_microbatch_size=config["experts"].get("instance_microbatch_size", 32),
        )
        rows = [
            prediction_frame(
                frames["train"]["sample_id"].to_numpy(), train_labels, dataset_id=dataset_id,
                split="train_oof", seed=int(seed), config_version=config["config_version"],
                extra_columns=_parent_columns(dataset_id, frames["train"], output.temporal_oof_window_summaries, output.cwt_oof_window_summaries),
                temporal=_sigmoid(output.temporal_oof_logits), cwt=_sigmoid(output.cwt_oof_logits),
            ),
            prediction_frame(
                frames["validation"]["sample_id"].to_numpy(), validation_labels, dataset_id=dataset_id,
                split="validation", seed=int(seed), config_version=config["config_version"],
                extra_columns=_parent_columns(dataset_id, frames["validation"], output.temporal_validation_window_summaries, output.cwt_validation_window_summaries),
                temporal=_sigmoid(output.temporal_validation_logits), cwt=_sigmoid(output.cwt_validation_logits),
            ),
            prediction_frame(
                frames["test"]["sample_id"].to_numpy(), test_labels, dataset_id=dataset_id,
                split="test_confirmatory" if dataset_id == MATLAB_DATASET_ID else "test_grouped_holdout",
                seed=int(seed), config_version=config["config_version"],
                extra_columns=_parent_columns(dataset_id, frames["test"], output.temporal_test_window_summaries, output.cwt_test_window_summaries),
                temporal=_sigmoid(output.temporal_test_logits), cwt=_sigmoid(output.cwt_test_logits),
            ),
        ]
        if dataset_id == MATLAB_DATASET_ID:
            te1_temporal, te1_cwt = output.additional_test_logits["Te1"]
            rows.append(prediction_frame(
                frames["test_historical"]["sample_id"].to_numpy(), loaded["Te1.mat"].label,
                dataset_id=dataset_id, split="test_historical", seed=int(seed),
                config_version=config["config_version"], temporal=_sigmoid(te1_temporal), cwt=_sigmoid(te1_cwt),
            ))
        seed_root.mkdir(parents=True, exist_ok=True)
        write_predictions(pd.concat(rows, ignore_index=True), prediction_path)
        write_json({
            "provenance": run_provenance(
                config_path=args.config, dataset_id=dataset_id, split="train_oof_validation_confirmatory_test",
                seed=int(seed), imbalance_strategy="fold_local_pos_weight" if dataset_id == VSB_DATASET_ID else "none",
            ),
            "representation": {
                "policy_status": "frozen",
                "window": spec.as_dict(
                    signal_length=signal_length,
                    sampling_frequency_hz=40_000_000.0 if dataset_id == VSB_DATASET_ID else None,
                ),
                "signal_length": signal_length, "cwt_scales": CWT_SCALES.tolist(),
                "cwt_time_bins": DEFAULT_CWT_TIME_BINS, "morlet_w0": DEFAULT_MORLET_W0,
                "preprocessing_version": "representation-aware-mi-v1",
            },
            "multi_instance": {"loss_unit": "parent_signal", "instance_microbatch_size": config["experts"].get("instance_microbatch_size", 32)},
            "temporal_histories": output.temporal_histories, "cwt_histories": output.cwt_histories,
            "oof_coverage": int(np.isfinite(output.temporal_oof_logits).sum()),
            "test_tuning": False, "te2_accessed_after_freeze": dataset_id == MATLAB_DATASET_ID,
        }, seed_root / "expert_run.json")
        del output
        gc.collect()
        print(f"completed {dataset_slug} seed {seed}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("matlab", "vsb", "both"), default="both")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--io-batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.io_batch_size < 1:
        parser.error("--io-batch-size must be positive")
    config = require_training_ready(args.config, dataset_id=VSB_DATASET_ID if args.dataset in {"vsb", "both"} else MATLAB_DATASET_ID, confirmatory=True)
    datasets = [MATLAB_DATASET_ID, VSB_DATASET_ID] if args.dataset == "both" else [MATLAB_DATASET_ID if args.dataset == "matlab" else VSB_DATASET_ID]
    for dataset_id in datasets:
        _run_dataset(args, config, dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
