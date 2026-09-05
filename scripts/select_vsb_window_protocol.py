"""Select and freeze the representation-aware VSB window protocol.

Only the development train/validation partitions are materialized here.  The
grouped VSB holdout and MATLAB Te2 are intentionally never opened by this
script.  Candidate A is retained as a historical diagnostic reference and is
not eligible for the new selection.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from partial_discharge_adaptive_fusion.dataset import (
    audit_vsb_metadata, audit_vsb_signal_sample, inspect_vsb_parquet_schema,
    iter_vsb_signal_batches, load_vsb_metadata, resolve_dataset,
)
from partial_discharge_adaptive_fusion.evaluation import fast_mcc
from partial_discharge_adaptive_fusion.experiments import train_two_experts_for_seed
from partial_discharge_adaptive_fusion.pipeline import require_development_ready
from partial_discharge_adaptive_fusion.protocol import VSB_DATASET_ID, freeze_protocol
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.windowed import (
    DEFAULT_CWT_TIME_BINS, DEFAULT_MORLET_W0,
    VSB_SAMPLING_FREQUENCY_HZ, fit_window_standardizer, windowed_cache_parameters,
    write_windowed_cache,
)
from partial_discharge_adaptive_fusion.windowing import WindowSpec, validate_parent_assignments
from partial_discharge_adaptive_fusion.splits import assert_complete_oof, vsb_grouped_manifest
from partial_discharge_adaptive_fusion.config import raw_data_root


DATASET_ID = VSB_DATASET_ID
DATASET_VERSION = "2018-kaggle-snapshot"
SCALE_ARRAY = np.geomspace(1.5, 64.0, 32).astype(np.float64)
CANDIDATES = (
    {"id": "B_4096_max", "window_length": 4096, "stride": 4096, "aggregation": "max"},
    {"id": "C_8192_top10_mean", "window_length": 8192, "stride": 4096, "aggregation": "top_k_mean"},
    {"id": "D_16384_top10_mean", "window_length": 16384, "stride": 8192, "aggregation": "top_k_mean"},
)


def _raw_factory(dataset, frame: pd.DataFrame, io_batch_size: int):
    def factory():
        for batch in iter_vsb_signal_batches(dataset, frame, batch_size=io_batch_size):
            yield batch.signal
    return factory


def _spec(candidate: dict[str, object]) -> WindowSpec:
    return WindowSpec(
        window_length=int(candidate["window_length"]),
        stride=int(candidate["stride"]),
        aggregation=str(candidate["aggregation"]),
        top_k_fraction=0.10,
    )


def _prepare_candidate(
    dataset,
    frames: dict[str, pd.DataFrame],
    candidate: dict[str, object],
    *,
    cache_root: Path,
    io_batch_size: int,
) -> dict[str, np.ndarray]:
    spec = _spec(candidate)
    raw_train = _raw_factory(dataset, frames["train"], io_batch_size)
    print(f"{candidate['id']}: fitting train-only temporal standardizer", flush=True)
    temporal_standardizer = fit_window_standardizer(raw_train, spec, representation="temporal")
    print(f"{candidate['id']}: fitting train-only CWT standardizer", flush=True)
    cwt_standardizer = fit_window_standardizer(
        raw_train, spec, representation="cwt", scales=SCALE_ARRAY,
        time_bins=DEFAULT_CWT_TIME_BINS, morlet_w0=DEFAULT_MORLET_W0,
    )
    candidate_root = cache_root / str(candidate["id"])
    prepared: dict[str, np.ndarray] = {}
    for split, frame in frames.items():
        raw = _raw_factory(dataset, frame, io_batch_size)
        temporal_path = candidate_root / f"temporal-{split}.npy"
        cwt_path = candidate_root / f"cwt-{split}.npy"
        temporal_spec = spec.n_windows(800_000)
        cwt_shape = (temporal_spec, 1, len(SCALE_ARRAY), DEFAULT_CWT_TIME_BINS)
        temporal_shape = (len(frame), temporal_spec, 1, spec.window_length)
        cwt_shape = (len(frame), *cwt_shape)
        temporal_parameters = windowed_cache_parameters(
            dataset_id=DATASET_ID, dataset_version=DATASET_VERSION, partition=split,
            spec=spec, representation="temporal", standardizer=temporal_standardizer,
            scales=SCALE_ARRAY, time_bins=DEFAULT_CWT_TIME_BINS,
            sampling_frequency_hz=VSB_SAMPLING_FREQUENCY_HZ, signal_length=800_000,
        )
        cwt_parameters = windowed_cache_parameters(
            dataset_id=DATASET_ID, dataset_version=DATASET_VERSION, partition=split,
            spec=spec, representation="cwt", standardizer=cwt_standardizer,
            scales=SCALE_ARRAY, time_bins=DEFAULT_CWT_TIME_BINS, morlet_w0=DEFAULT_MORLET_W0,
            sampling_frequency_hz=VSB_SAMPLING_FREQUENCY_HZ, signal_length=800_000,
        )
        if not _valid_cache(temporal_path, temporal_shape, temporal_parameters):
            write_windowed_cache(
                raw, n_samples=len(frame), spec=spec, representation="temporal",
                standardizer=temporal_standardizer, destination=str(temporal_path),
                dataset_id=DATASET_ID, dataset_version=DATASET_VERSION, partition=split,
                sampling_frequency_hz=VSB_SAMPLING_FREQUENCY_HZ, dtype="float16",
            )
        if not _valid_cache(cwt_path, cwt_shape, cwt_parameters):
            write_windowed_cache(
                _raw_factory(dataset, frame, io_batch_size), n_samples=len(frame), spec=spec,
                representation="cwt", standardizer=cwt_standardizer, destination=str(cwt_path),
                dataset_id=DATASET_ID, dataset_version=DATASET_VERSION, partition=split,
                scales=SCALE_ARRAY, time_bins=DEFAULT_CWT_TIME_BINS, morlet_w0=DEFAULT_MORLET_W0,
                sampling_frequency_hz=VSB_SAMPLING_FREQUENCY_HZ, dtype="float16",
            )
        temporal = np.load(temporal_path, mmap_mode="r", allow_pickle=False)
        cwt = np.load(cwt_path, mmap_mode="r", allow_pickle=False)
        if tuple(temporal.shape) != temporal_shape or tuple(cwt.shape) != cwt_shape:
            raise RuntimeError(f"{candidate['id']}/{split}: cache shape mismatch.")
        prepared[f"{split}_temporal"] = temporal
        prepared[f"{split}_cwt"] = cwt
        del raw
        gc.collect()
    return prepared


def _valid_cache(path: Path, expected_shape: tuple[int, ...], expected_parameters: dict[str, object]) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        tuple(values.shape) == expected_shape
        and metadata.get("shape") == list(expected_shape)
        and metadata.get("parameters") == expected_parameters
    )


def _candidate_policy(candidate: dict[str, object]) -> dict[str, object]:
    spec = _spec(candidate)
    common = {
        "input": "signal_windows",
        "signal_length": 800_000,
        "sampling_frequency_hz": VSB_SAMPLING_FREQUENCY_HZ,
        "window_length": spec.window_length,
        "stride": spec.stride,
        "window_count": spec.n_windows(800_000),
        "window_duration_seconds": spec.window_length / VSB_SAMPLING_FREQUENCY_HZ,
        "padding": "none",
        "normalization": "z_score_fit_on_train_only",
        "aggregation": spec.aggregation,
        "top_k_fraction": spec.top_k_fraction,
        "top_k": spec.top_k(800_000),
    }
    cwt = {
        **common,
        "wavelet": "complex_morlet",
        "morlet_w0": DEFAULT_MORLET_W0,
        "n_scales": len(SCALE_ARRAY),
        "scales_min": float(SCALE_ARRAY.min()),
        "scales_max": float(SCALE_ARRAY.max()),
        "scales": SCALE_ARRAY.tolist(),
        "time_bins": DEFAULT_CWT_TIME_BINS,
        "transform": "log_power",
        "clip_floor": -10.0,
        "pseudo_frequency_range_hz": [
            float(VSB_SAMPLING_FREQUENCY_HZ * DEFAULT_MORLET_W0 / (2 * np.pi * SCALE_ARRAY.max())),
            float(VSB_SAMPLING_FREQUENCY_HZ * DEFAULT_MORLET_W0 / (2 * np.pi * SCALE_ARRAY.min())),
        ],
    }
    return {
        "status": "frozen", "decision_basis": "development_only_seed_42_selection",
        "selected_candidate": candidate["id"], "temporal": common, "cwt": cwt,
        "test_results_used": False, "preprocessing_version": "representation-aware-mi-v1",
    }


def _choose(rows: list[dict[str, object]], tolerance: float) -> dict[str, object]:
    eligible = [row for row in rows if row.get("status") == "eligible"]
    if not eligible:
        raise RuntimeError("No VSB window candidate produced a development score.")
    best_mean = max(float(row["mean_mcc"]) for row in eligible)
    near = [row for row in eligible if float(row["mean_mcc"]) >= best_mean - tolerance]
    return max(near, key=lambda row: (float(row["minimum_expert_mcc"]), -float(row["compute_cost"])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-vsb-window-selection-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/v3-windowed/vsb-selection"))
    parser.add_argument("--selection-output", type=Path, default=Path("results/manifests/vsb_representation_selection.json"))
    parser.add_argument("--frozen-output", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml"))
    args = parser.parse_args(argv)
    if args.io_batch_size < 1:
        parser.error("--io-batch-size must be positive")
    config = require_development_ready(args.config, dataset_id=DATASET_ID)
    dataset = resolve_dataset(DATASET_ID, DATASET_VERSION, raw_root=raw_data_root(args.raw_root))
    metadata = load_vsb_metadata(dataset)
    manifest = vsb_grouped_manifest(
        metadata, dataset_id=dataset.dataset_id, dataset_version=dataset.version,
        split_seed=config["splits"]["split_seed"], oof_splits=config["splits"]["oof_folds"],
    )
    assert_complete_oof(manifest)
    train_ids = manifest.frame.loc[manifest.frame["split"] == "train", "sample_id"].astype(str)
    validation_ids = manifest.frame.loc[manifest.frame["split"] == "validation", "sample_id"].astype(str)
    metadata_by_id = metadata.set_index("signal_id", drop=False)
    frames = {
        "train": metadata_by_id.loc[train_ids].reset_index(drop=True),
        "validation": metadata_by_id.loc[validation_ids].reset_index(drop=True),
    }
    combined = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    validate_parent_assignments(
        combined["signal_id"].to_numpy(), combined["id_measurement"].to_numpy(),
        np.concatenate([np.repeat("train", len(frames["train"])), np.repeat("validation", len(frames["validation"]))]),
        n_windows=WindowSpec(4096, 4096, aggregation="max").n_windows(800_000),
    )
    train_manifest = manifest.frame.loc[manifest.frame["split"] == "train"].reset_index(drop=True)
    folds = train_manifest["oof_fold"].to_numpy(np.int64)
    groups = train_manifest["group_id"].astype(str).to_numpy()
    labels = frames["train"]["target"].to_numpy(np.int64)
    validation_labels = frames["validation"]["target"].to_numpy(np.int64)
    rows: list[dict[str, object]] = [{
        "id": "A_historical_native", "status": "historical_diagnostic_only",
        "reason": "native 800000-sample pipeline is retained for historical comparison and excluded from final selection",
    }]
    for candidate in CANDIDATES:
        print(f"Selecting VSB candidate {candidate['id']} with seed 42", flush=True)
        prepared = _prepare_candidate(
            dataset, frames, candidate, cache_root=args.cache_root, io_batch_size=args.io_batch_size,
        )
        spec = _spec(candidate)
        output = train_two_experts_for_seed(
            train_temporal=prepared["train_temporal"], train_cwt=prepared["train_cwt"],
            train_labels=labels, oof_folds=folds,
            validation_temporal=prepared["validation_temporal"], validation_cwt=prepared["validation_cwt"],
            validation_labels=validation_labels,
            # The development validation is passed through the legacy output slot only so
            # the reusable trainer can be used; no VSB test/holdout data is loaded.
            test_temporal=prepared["validation_temporal"], test_cwt=prepared["validation_cwt"],
            test_labels=validation_labels, seed=42,
            temporal_epochs=config["experts"]["epochs"]["temporal"],
            cwt_epochs=config["experts"]["epochs"]["cwt"], batch_size=config["experts"]["batch_size"],
            inference_batch_size=config["experts"].get("inference_batch_size", 4),
            gradient_accumulation_steps=config["experts"].get("gradient_accumulation_steps", 1),
            imbalance_strategy="fold_local_pos_weight", groups=groups, multi_instance=True,
            aggregation=spec.aggregation, top_k_fraction=spec.top_k_fraction,
            instance_microbatch_size=config["experts"].get("instance_microbatch_size", 32),
        )
        sigmoid = lambda values: 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))
        temporal_mcc = fast_mcc(validation_labels, sigmoid(output.temporal_validation_logits) >= 0.5)
        cwt_mcc = fast_mcc(validation_labels, sigmoid(output.cwt_validation_logits) >= 0.5)
        k = spec.n_windows(800_000)
        rows.append({
            "id": candidate["id"], "status": "eligible", "window": spec.as_dict(),
            "window_count": k, "temporal_mcc": temporal_mcc, "cwt_mcc": cwt_mcc,
            "mean_mcc": float((temporal_mcc + cwt_mcc) / 2.0),
            "minimum_expert_mcc": float(min(temporal_mcc, cwt_mcc)),
            "compute_cost": float(k * (spec.window_length + len(SCALE_ARRAY) * DEFAULT_CWT_TIME_BINS)),
            "selection_data": "train_and_validation_only", "seed": 42,
        })
        del output, prepared
        gc.collect()
    selected = _choose(rows, tolerance=float(config["selection"]["tie_tolerance_mcc"]))
    audit_record = {
        "dataset": {"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION},
        "metadata_audit": audit_vsb_metadata(metadata),
        "native_signal_sample": audit_vsb_signal_sample(dataset, metadata, sample_size=3),
        "parquet_schema": inspect_vsb_parquet_schema(dataset),
    }
    policy = _candidate_policy(next(item for item in CANDIDATES if item["id"] == selected["id"]))
    result = {
        "protocol": config["config_version"], "selection_seed": 42,
        "blocked_splits": ["test", "Te2.mat", "official_unlabeled_test"],
        "candidates": rows, "selected": selected, "selected_policy": policy,
        "audit_record": audit_record, "test_results_used": False,
    }
    write_json(result, args.selection_output)
    frozen_input = dict(config)
    frozen_input["config_version"] = "two-dataset-confirmatory-v3-windowed-localraw"
    frozen_input["parent_config_version"] = config["config_version"]
    frozen_input["protocol_status"] = "development_selection"
    frozen_input["datasets"] = {**config["datasets"], VSB_DATASET_ID: {
        **config["datasets"][VSB_DATASET_ID], "input_policy": policy,
    }}
    frozen_input["selection_record"] = {
        "selection_manifest": str(args.selection_output), "selected_candidate": selected["id"],
        "selection_seed": 42, "holdout_opened": False,
    }
    freeze_protocol(
        frozen_input, vsb_input_policy=policy, audit_record=audit_record, output_path=args.frozen_output,
    )
    print(f"Selected and froze {selected['id']} -> {args.frozen_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
