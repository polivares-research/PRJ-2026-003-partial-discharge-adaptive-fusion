"""Run the frozen two-expert MATLAB stage for one or more seeds."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from partial_discharge_adaptive_fusion.cache import CacheRecord, cache_key, write_array_cache
from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import load_mat_confirmatory_partition, load_mat_partition, resolve_dataset
from partial_discharge_adaptive_fusion.experiments import train_two_experts_for_seed
from partial_discharge_adaptive_fusion.pipeline import require_training_ready, run_provenance
from partial_discharge_adaptive_fusion.reporting import prediction_frame, write_json, write_predictions
from partial_discharge_adaptive_fusion.representations import cwt_input, cwt_log_power_batch, fit_standardizer, temporal_input
from partial_discharge_adaptive_fusion.splits import matlab_manifest


DATASET_ID = "engineering-partial-discharge-noise-signals"
DATASET_VERSION = "v1"
CWT_SCALES = np.geomspace(1.5, 64.0, 32).astype(np.float64)


def _prepare_caches(batches, cache_root: Path):
    print("MATLAB cache preparation: fitting train-only standardizers", flush=True)
    train_signal = batches["Tr1.mat"].signal
    raw_standardizer = fit_standardizer(train_signal, axis=0)
    train_cwt = cwt_log_power_batch(train_signal, scales=CWT_SCALES, time_bins=120)
    cwt_standardizer = fit_standardizer(train_cwt, axis=0)
    del train_cwt
    gc.collect()
    prepared = {}
    for partition, batch in batches.items():
        temporal_parameters = {"input_length": 400}
        cwt_parameters = {
            "scales": CWT_SCALES.tolist(), "time_bins": 120, "morlet_w0": 6.0,
            "floor": -10.0, "standardized_on": "Tr1.mat",
        }
        temporal_key = cache_key(
            DATASET_ID, DATASET_VERSION, partition, "matlab_temporal", temporal_parameters,
        )
        cwt_key = cache_key(
            DATASET_ID, DATASET_VERSION, partition, "matlab_cwt", cwt_parameters,
        )
        temporal_record = _existing_cache(
            cache_root, f"matlab_temporal-{temporal_key}", expected_representation="matlab_temporal",
        )
        cwt_record = _existing_cache(
            cache_root, f"matlab_cwt-{cwt_key}", expected_representation="matlab_cwt",
        )
        if temporal_record is not None and cwt_record is not None:
            print(f"MATLAB cache preparation: reusing {partition}", flush=True)
            prepared[partition] = {"temporal": temporal_record, "cwt": cwt_record}
            continue
        print(f"MATLAB cache preparation: computing {partition}", flush=True)
        temporal = temporal_input(batch.signal, raw_standardizer)
        cwt_raw = cwt_log_power_batch(batch.signal, scales=CWT_SCALES, time_bins=120)
        cwt = cwt_input(cwt_raw, cwt_standardizer)
        prepared[partition] = {
            "temporal": write_array_cache(
                temporal, root=cache_root, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
                partition=partition, representation="matlab_temporal", parameters=temporal_parameters,
            ),
            "cwt": write_array_cache(
                cwt, root=cache_root, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
                partition=partition, representation="matlab_cwt", parameters=cwt_parameters,
                dtype="float16",
            ),
        }
        del temporal, cwt_raw, cwt
        gc.collect()
    write_json({
        "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
        "raw_standardizer_mean": raw_standardizer.mean.tolist(),
        "raw_standardizer_std": raw_standardizer.std.tolist(),
        "cwt_standardizer_mean": cwt_standardizer.mean.tolist(),
        "cwt_standardizer_std": cwt_standardizer.std.tolist(),
        "cwt_scales": CWT_SCALES.tolist(), "cwt_time_bins": 120,
    }, cache_root / "matlab_standardizers.json")
    return prepared


def _existing_cache(cache_root: Path, stem: str, *, expected_representation: str) -> CacheRecord | None:
    """Return a validated cache record so an interrupted run can resume safely."""

    array_path = cache_root / f"{stem}.npy"
    metadata_path = cache_root / f"{stem}.json"
    if not array_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("representation") != expected_representation:
            return None
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != metadata.get("shape"):
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return CacheRecord(array_path=array_path, metadata_path=metadata_path, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path, help="Optional local raw-data root.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--cache-root", type=Path, default=Path("results/cache/matlab"))
    parser.add_argument("--output-root", type=Path, default=Path("results/runs/matlab"))
    args = parser.parse_args(argv)
    config = require_training_ready(args.config, dataset_id=DATASET_ID, confirmatory=True)
    dataset = resolve_dataset(DATASET_ID, DATASET_VERSION, raw_root=raw_data_root(args.raw_root))
    batches = {
        name: load_mat_partition(dataset, name)
        for name in ("Tr1.mat", "Va1.mat", "Te1.mat")
    }
    batches["Te2.mat"] = load_mat_confirmatory_partition(dataset, config)
    print("MATLAB manifest: loaded Tr1, Va1, Te1 and frozen Te2", flush=True)
    manifest = matlab_manifest(
        {name: batch.label for name, batch in batches.items()},
        dataset_id=dataset.dataset_id, dataset_version=dataset.version,
    )
    manifest.write_csv(Path(config["outputs"]["manifest_root"]) / "matlab_manifest.csv")
    prepared = _prepare_caches(batches, args.cache_root)
    train = prepared["Tr1.mat"]
    validation = prepared["Va1.mat"]
    te1 = prepared["Te1.mat"]
    te2 = prepared["Te2.mat"]
    train_indices = manifest.frame.loc[manifest.frame["split"] == "train"].sort_values("sample_id")
    folds = train_indices["oof_fold"].to_numpy(np.int64)
    # Source-generated IDs sort in the same order as the adapter's partition rows.
    assert np.array_equal(train_indices["sample_id"].to_numpy(), batches["Tr1.mat"].sample_id)
    for seed in args.seeds:
        seed_root = args.output_root / f"seed-{seed}"
        prediction_path = seed_root / "expert_predictions.parquet"
        if prediction_path.exists() and (seed_root / "expert_run.json").exists():
            print(f"completed MATLAB expert seed {seed} already exists: {seed_root}", flush=True)
            continue
        print(f"MATLAB experts: seed {seed} — cross-fitting temporal expert", flush=True)
        output = train_two_experts_for_seed(
            train_temporal=np.load(train["temporal"].array_path, mmap_mode="r"),
            train_cwt=np.load(train["cwt"].array_path, mmap_mode="r"),
            train_labels=batches["Tr1.mat"].label, oof_folds=folds,
            validation_temporal=np.load(validation["temporal"].array_path, mmap_mode="r"),
            validation_cwt=np.load(validation["cwt"].array_path, mmap_mode="r"),
            validation_labels=batches["Va1.mat"].label,
            test_temporal=np.load(te2["temporal"].array_path, mmap_mode="r"),
            test_cwt=np.load(te2["cwt"].array_path, mmap_mode="r"),
            test_labels=batches["Te2.mat"].label, seed=seed,
            temporal_epochs=config["experts"]["epochs"]["temporal"],
            cwt_epochs=config["experts"]["epochs"]["cwt"],
            batch_size=config["experts"]["batch_size"], imbalance_strategy="none",
            inference_batch_size=config["experts"].get(
                "inference_batch_size", max(config["experts"]["batch_size"], 512),
            ),
            gradient_accumulation_steps=config["experts"].get("gradient_accumulation_steps", 1),
            additional_tests={
                "Te1": (
                    np.load(te1["temporal"].array_path, mmap_mode="r"),
                    np.load(te1["cwt"].array_path, mmap_mode="r"),
                ),
            },
        )
        print(f"MATLAB experts: seed {seed} — writing predictions", flush=True)
        seed_root.mkdir(parents=True, exist_ok=True)
        rows = [
            prediction_frame(
                batches["Tr1.mat"].sample_id, batches["Tr1.mat"].label,
                dataset_id=DATASET_ID, split="train_oof", seed=seed, config_version=config["config_version"],
                temporal=1.0 / (1.0 + np.exp(-output.temporal_oof_logits)),
                cwt=1.0 / (1.0 + np.exp(-output.cwt_oof_logits)),
            ),
            prediction_frame(
                batches["Va1.mat"].sample_id, batches["Va1.mat"].label,
                dataset_id=DATASET_ID, split="validation", seed=seed, config_version=config["config_version"],
                temporal=1.0 / (1.0 + np.exp(-output.temporal_validation_logits)),
                cwt=1.0 / (1.0 + np.exp(-output.cwt_validation_logits)),
            ),
            prediction_frame(
                batches["Te2.mat"].sample_id, batches["Te2.mat"].label,
                dataset_id=DATASET_ID, split="test_confirmatory", seed=seed, config_version=config["config_version"],
                temporal=1.0 / (1.0 + np.exp(-output.temporal_test_logits)),
                cwt=1.0 / (1.0 + np.exp(-output.cwt_test_logits)),
            ),
        ]
        te1_temporal, te1_cwt = output.additional_test_logits["Te1"]
        rows.append(prediction_frame(
            batches["Te1.mat"].sample_id, batches["Te1.mat"].label,
            dataset_id=DATASET_ID, split="test_historical", seed=seed, config_version=config["config_version"],
            temporal=1.0 / (1.0 + np.exp(-te1_temporal)), cwt=1.0 / (1.0 + np.exp(-te1_cwt)),
        ))
        write_predictions(pd.concat(rows, ignore_index=True), seed_root / "expert_predictions.parquet")
        write_json({
            "provenance": run_provenance(
                config_path=args.config, dataset_id=DATASET_ID, split="all", seed=seed,
                imbalance_strategy="none_balanced_partitions",
            ),
            "temporal_histories": output.temporal_histories,
            "cwt_histories": output.cwt_histories,
            "oof_coverage": int(np.isfinite(output.temporal_oof_logits).sum()),
            "neural_batch_size": config["experts"]["batch_size"],
            "inference_batch_size": config["experts"].get(
                "inference_batch_size", max(config["experts"]["batch_size"], 512),
            ),
            "gradient_accumulation_steps": config["experts"].get("gradient_accumulation_steps", 1),
            "te2_accessed_after_freeze": True,
        }, seed_root / "expert_run.json")
        print(f"completed MATLAB expert seed {seed}: {seed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
