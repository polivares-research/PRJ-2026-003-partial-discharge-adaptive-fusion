"""Run the V4 development-only VSB expert gate."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import (
    iter_vsb_signal_batches, load_vsb_metadata, resolve_dataset,
)
from partial_discharge_adaptive_fusion.experiments.runner import (
    cleanup_cuda, cross_fitted_logits, positive_class_weight, train_expert,
)
from partial_discharge_adaptive_fusion.modeling.predict import logits_and_window_summaries_for_array
from partial_discharge_adaptive_fusion.protocol import VSB_DATASET_ID, assert_development_selection_ready, load_experiment_config
from partial_discharge_adaptive_fusion.reporting import prediction_frame, write_json
from partial_discharge_adaptive_fusion.splits import vsb_grouped_manifest
from partial_discharge_adaptive_fusion.v4_protocol import gate_decision, select_v4_candidate
from partial_discharge_adaptive_fusion.windowed import DEFAULT_CWT_TIME_BINS, DEFAULT_MORLET_W0
from partial_discharge_adaptive_fusion.windowing import WindowSpec, validate_parent_assignments

from run_representation_aware_experts import CWT_SCALES, _factory_from_vsb, _prepare_caches, _policy_spec, _sigmoid


def _mcc_from_oof(labels: np.ndarray, logits: np.ndarray) -> float:
    from partial_discharge_adaptive_fusion.fusion import select_threshold

    probabilities = _sigmoid(logits)
    threshold = select_threshold(labels, probabilities)
    return float(matthews_corrcoef(labels, probabilities >= threshold))


def _validation_model(
    values: np.ndarray, labels: np.ndarray, folds: np.ndarray, validation_values: np.ndarray,
    *, kind: str, seed: int, epochs: int, batch_size: int, inference_batch_size: int,
    imbalance_strategy: str, aggregation: str, top_k_fraction: float, microbatch: int,
    num_workers: int, persistent_workers: bool, prefetch_factor: int, mixed_precision: bool,
) -> np.ndarray:
    inner_validation = np.flatnonzero(folds == np.min(folds))
    inner_train = np.flatnonzero(folds != np.min(folds))
    pos_weight = (
        positive_class_weight(labels[inner_train])
        if imbalance_strategy == "fold_local_pos_weight" else None
    )
    result = train_expert(
        values, labels, values, labels, kind=kind, seed=seed, epochs=epochs,
        batch_size=batch_size, pos_weight=pos_weight, train_indices=inner_train,
        validation_indices=inner_validation, validation_batch_size=inference_batch_size,
        multi_instance=True, aggregation=aggregation, top_k_fraction=top_k_fraction,
        instance_microbatch_size=microbatch, num_workers=num_workers,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        mixed_precision=mixed_precision,
    )
    logits, _ = logits_and_window_summaries_for_array(
        result.model, validation_values, batch_size=inference_batch_size,
    )
    del result
    cleanup_cuda()
    return logits


def _load_development_data(config: dict, args):
    dataset = resolve_dataset(
        VSB_DATASET_ID, "2018-kaggle-snapshot", raw_root=raw_data_root(args.raw_root),
    )
    metadata = load_vsb_metadata(dataset)
    manifest = vsb_grouped_manifest(
        metadata, dataset_id=VSB_DATASET_ID, dataset_version="2018-kaggle-snapshot",
        split_seed=config["splits"]["split_seed"], oof_splits=config["splits"]["oof_folds"],
    )
    metadata_by_id = metadata.set_index("signal_id", drop=False)
    frames = {}
    factories = {}
    for split in ("train", "validation"):
        frame = metadata_by_id.loc[
            manifest.frame.loc[manifest.frame["split"] == split, "sample_id"].astype(str)
        ].reset_index(drop=True)
        frame["sample_id"] = frame["signal_id"].astype(str)
        frames[split] = frame
        factories[split] = _factory_from_vsb(dataset, frame, args.io_batch_size)
    combined = pd.concat(list(frames.values()), ignore_index=True)
    split_values = np.concatenate([
        np.repeat("train", len(frames["train"])),
        np.repeat("validation", len(frames["validation"])),
    ])
    spec = _policy_spec(config, VSB_DATASET_ID)
    validate_parent_assignments(
        combined["signal_id"].to_numpy(), combined["id_measurement"].to_numpy(),
        split_values, n_windows=spec.n_windows(800_000),
    )
    prepared = _prepare_caches(
        dataset_id=VSB_DATASET_ID, dataset_version="2018-kaggle-snapshot", dataset=dataset,
        frames=frames, raw_factories=factories, signal_length=800_000, spec=spec,
        cache_root=Path(config["outputs"]["cache_root"]), io_batch_size=args.io_batch_size,
        preprocessing_version=config["preprocessing_version"], source_provenance=config["data_source"],
    )
    train_manifest = manifest.frame.loc[manifest.frame["split"] == "train"].reset_index(drop=True)
    return frames, prepared, train_manifest["oof_fold"].to_numpy(np.int64), spec


def _run_candidate(config: dict, candidate_id: str, candidate: dict, frames, prepared, folds, spec) -> dict:
    labels = frames["train"]["target"].to_numpy(np.int64)
    validation_labels = frames["validation"]["target"].to_numpy(np.int64)
    groups = frames["train"]["id_measurement"].astype(str).to_numpy()
    runtime = config.get("runtime", {})
    loader_options = {
        "num_workers": int(runtime.get("num_workers", 0)),
        "persistent_workers": bool(runtime.get("persistent_workers", False)),
        "prefetch_factor": int(runtime.get("prefetch_factor", 2)),
        "mixed_precision": bool(runtime.get("mixed_precision", False)),
    }
    seeds = [int(seed) for seed in config["selection"]["development_seeds"]]
    records = []
    for seed in seeds:
        temporal_oof = cross_fitted_logits(
            prepared["train"]["temporal"], labels, kind=candidate["temporal_kind"], folds=folds,
            seed=seed, groups=groups, epochs=config["experts"]["epochs"]["temporal"],
            batch_size=config["experts"]["batch_size"], validation_batch_size=config["experts"]["inference_batch_size"],
            imbalance_strategy="fold_local_pos_weight", multi_instance=True,
            aggregation=spec.aggregation, top_k_fraction=spec.top_k_fraction,
            instance_microbatch_size=config["experts"]["instance_microbatch_size"],
            **loader_options,
        )
        cwt_oof = cross_fitted_logits(
            prepared["train"]["cwt"], labels, kind=candidate["cwt_kind"], folds=folds,
            seed=seed + 100, groups=groups, epochs=config["experts"]["epochs"]["cwt"],
            batch_size=config["experts"]["batch_size"], validation_batch_size=config["experts"]["inference_batch_size"],
            imbalance_strategy="fold_local_pos_weight", multi_instance=True,
            aggregation=spec.aggregation, top_k_fraction=spec.top_k_fraction,
            instance_microbatch_size=config["experts"]["instance_microbatch_size"],
            **loader_options,
        )
        temporal_validation = _validation_model(
            prepared["train"]["temporal"], labels, folds, prepared["validation"]["temporal"],
            kind=candidate["temporal_kind"], seed=seed, epochs=config["experts"]["epochs"]["temporal"],
            batch_size=config["experts"]["batch_size"], inference_batch_size=config["experts"]["inference_batch_size"],
            imbalance_strategy="fold_local_pos_weight", aggregation=spec.aggregation,
            top_k_fraction=spec.top_k_fraction, microbatch=config["experts"]["instance_microbatch_size"],
            **loader_options,
        )
        cwt_validation = _validation_model(
            prepared["train"]["cwt"], labels, folds, prepared["validation"]["cwt"],
            kind=candidate["cwt_kind"], seed=seed + 100, epochs=config["experts"]["epochs"]["cwt"],
            batch_size=config["experts"]["batch_size"], inference_batch_size=config["experts"]["inference_batch_size"],
            imbalance_strategy="fold_local_pos_weight", aggregation=spec.aggregation,
            top_k_fraction=spec.top_k_fraction, microbatch=config["experts"]["instance_microbatch_size"],
            **loader_options,
        )
        temporal_mcc = _mcc_from_oof(labels, temporal_oof.logits)
        cwt_mcc = _mcc_from_oof(labels, cwt_oof.logits)
        temporal_val = _sigmoid(temporal_validation)
        cwt_val = _sigmoid(cwt_validation)
        from partial_discharge_adaptive_fusion.fusion import select_threshold
        temporal_threshold = select_threshold(labels, _sigmoid(temporal_oof.logits))
        cwt_threshold = select_threshold(labels, _sigmoid(cwt_oof.logits))
        temporal_val_mcc = float(matthews_corrcoef(validation_labels, temporal_val >= temporal_threshold))
        cwt_val_mcc = float(matthews_corrcoef(validation_labels, cwt_val >= cwt_threshold))
        rows = prediction_frame(
            frames["validation"]["sample_id"].to_numpy(), validation_labels,
            dataset_id=VSB_DATASET_ID, split="development_validation", seed=seed,
            config_version=config["config_version"], temporal=temporal_val, cwt=cwt_val,
        )
        output = Path(config["outputs"]["results_root"]) / "development" / candidate_id / f"seed-{seed}"
        output.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(output / "validation_predictions.parquet", index=False)
        records.append({
            "seed": seed, "temporal_oof_mcc": temporal_mcc, "cwt_oof_mcc": cwt_mcc,
            "temporal_validation_mcc": temporal_val_mcc, "cwt_validation_mcc": cwt_val_mcc,
            "best_individual_mcc": max(temporal_val_mcc, cwt_val_mcc),
            "minimum_expert_mcc": min(temporal_val_mcc, cwt_val_mcc),
        })
        del temporal_oof, cwt_oof
        gc.collect()
    decision = gate_decision(
        candidate_id, records, compute_cost=float(candidate.get("compute_cost", 0.0)),
    )
    return {"decision": decision.__dict__, "seed_records": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v4-windowed-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--candidate", action="append")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    assert_development_selection_ready(config)
    frames, prepared, folds, spec = _load_development_data(config, args)
    candidates = config["selection"]["model_candidates"]
    selected_ids = args.candidate or list(candidates)
    candidate_results = {
        candidate_id: _run_candidate(config, candidate_id, candidates[candidate_id], frames, prepared, folds, spec)
        for candidate_id in selected_ids
    }
    decisions = [
        type("Decision", (), result["decision"])() for result in candidate_results.values()
    ]
    selected = None
    if any(item.status == "PASS" for item in decisions):
        selected = select_v4_candidate(decisions).candidate_id
    summary = {
        "config_version": config["config_version"], "holdouts_opened": False,
        "candidate_decisions": {key: value["decision"] for key, value in candidate_results.items()},
        "seed_records": {key: value["seed_records"] for key, value in candidate_results.items()},
        "selected_candidate": selected,
        "gate_status": "PASS" if selected else "STOP_OR_WEAK",
    }
    output = Path(config["outputs"]["development_gate_summary"])
    write_json(summary, output)
    print(summary["gate_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
