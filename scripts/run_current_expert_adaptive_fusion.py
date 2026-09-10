#!/usr/bin/env python3
"""Run the current-expert adaptive-fusion confirmation in resumable stages.

Development stages consume only current full-signal expert predictions and
regenerate MATLAB temporal predictions with fixed epochs.  Confirmatory
holdouts are never read by the development stages and require an explicit
frozen configuration.
"""

from __future__ import annotations

import argparse
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
from partial_discharge_adaptive_fusion.current_expert_fusion import (  # noqa: E402
    ALL_METHODS,
    canonical_pair_frame,
    component_thresholds,
    cross_fitted_meta_oof,
    evaluate_pair_methods,
    expert_gate,
    fusion_gate,
    hierarchical_delta_ci,
    paired_statistics,
    sha256_file,
    validate_prediction_source,
)
from partial_discharge_adaptive_fusion.current_expert_fusion import _apply_components  # noqa: E402
from partial_discharge_adaptive_fusion.dataset import load_mat_partition, resolve_dataset  # noqa: E402
from partial_discharge_adaptive_fusion.evaluation import binary_metrics  # noqa: E402
from partial_discharge_adaptive_fusion.fusion import select_threshold  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.data import NumpyDataset  # noqa: E402
from partial_discharge_adaptive_fusion.modeling.train import (  # noqa: E402
    cross_fitted_logits,
    fold_assignments,
    predict_logits,
    train_expert,
)
from partial_discharge_adaptive_fusion.representations import fit_standardizer, temporal_input  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/current-expert-adaptive-fusion-localraw.yaml"
DEFAULT_SOURCE = ROOT / "results/audits/full-signal-spectrogram-cross-dataset/validation_predictions.parquet"
MATLAB_ID = "engineering-partial-discharge-noise-signals"
VSB_ID = "engineering-vsb-power-line-fault-detection"
DEVELOPMENT_SEEDS = (42, 43, 44)


def logger_setup(log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("current_expert_adaptive_fusion")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [elapsed=%(relativeCreated).1fms] %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(destination)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def complete(root: Path, stage: str) -> bool:
    return (root / f"{stage}.complete").is_file() and (root / f"{stage}.json").is_file()


def finish(root: Path, stage: str, payload: dict[str, Any]) -> None:
    write_json(root / f"{stage}.json", payload)
    (root / f"{stage}.complete").write_text("complete\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    if config.get("protocol_status") not in {"development_selection", "frozen"}:
        raise ValueError("Current fusion configuration must be development_selection or frozen")
    if config.get("data_source", {}).get("root_environment_variable") != "PD_RAW_DATA_ROOT":
        raise ValueError("Current fusion configuration must use PD_RAW_DATA_ROOT")
    if config.get("protection", {}).get("historical_metrics_in_active_selection") != "forbidden":
        raise ValueError("Historical metrics must be forbidden in active selection")
    return config


def output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["outputs"]["audit_root"]


def source_path(config: dict[str, Any], override: Path | None) -> Path:
    return (override or (ROOT / config["sources"]["development_predictions"])).resolve()


def stage_preflight(config: dict[str, Any], config_path: Path, raw_root: Path, output: Path, source_override: Path | None, logger: logging.Logger) -> None:
    if complete(output, "preflight"):
        logger.info("preflight already complete; reusing verified source")
        return
    started = time.perf_counter()
    runtime = runtime_info().as_dict()
    require_cuda()
    source = source_path(config, source_override)
    if not source.is_file():
        raise FileNotFoundError(f"Current expert prediction source is missing: {source}")
    frame = pd.read_parquet(source)
    validation = validate_prediction_source(frame, expected_seeds=DEVELOPMENT_SEEDS, allow_holdouts=False)
    validation.update({
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "raw_root": str(raw_root),
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
    })
    write_json(output / "preflight.json", validation)
    finish(output, "preflight", validation)
    logger.info("preflight PASS: rows=%d source_sha256=%s", len(frame), validation["source_sha256"][:12])


def _prediction_rows(base: pd.DataFrame, probabilities: np.ndarray, threshold: float, *, dataset: str, split: str, method: str, seed: int) -> pd.DataFrame:
    result = base.copy()
    result["dataset"] = dataset
    result["split"] = split
    result["method"] = method
    result["seed"] = int(seed)
    result["probability"] = np.asarray(probabilities, dtype=float)
    result["threshold"] = float(threshold)
    result["prediction"] = (result["probability"] >= threshold).astype(np.int64)
    return result


def stage_matlab_clean(config: dict[str, Any], raw_root: Path, output: Path, source_override: Path | None, logger: logging.Logger) -> None:
    if complete(output, "matlab_clean"):
        logger.info("MATLAB clean temporal stage already complete; reusing predictions")
        return
    if not complete(output, "preflight"):
        raise RuntimeError("Run preflight before MATLAB clean temporal regeneration")
    started = time.perf_counter()
    source = pd.read_parquet(source_path(config, source_override))
    matlab = resolve_dataset(MATLAB_ID, "v1", raw_root=raw_root)
    train = load_mat_partition(matlab, "Tr1.mat")
    validation = load_mat_partition(matlab, "Va1.mat")
    folds = fold_assignments(train.label, n_splits=5, seed=42)
    train_base = pd.DataFrame({"sample_id": train.sample_id.astype(str), "id_measurement": train.sample_id.astype(str), "phase": "NA", "target": train.label})
    validation_base = pd.DataFrame({"sample_id": validation.sample_id.astype(str), "id_measurement": validation.sample_id.astype(str), "phase": "NA", "target": validation.label})
    rows: list[pd.DataFrame] = []
    epochs = int(config["experts"]["matlab_temporal"]["epochs"])
    for seed in DEVELOPMENT_SEEDS:
        logger.info("MATLAB clean temporal seed %d: OOF (%d fixed epochs)", seed, epochs)

        def fold_transform(raw: np.ndarray, fit: np.ndarray, holdout: np.ndarray, fold: int) -> np.ndarray:
            standardizer = fit_standardizer(np.asarray(raw)[fit], axis=0)
            return temporal_input(np.asarray(raw), standardizer)

        oof = cross_fitted_logits(
            train.signal, train.label, kind="temporal", folds=folds, seed=seed,
            epochs=epochs, batch_size=int(config["experts"]["batch_size"]),
            validation_batch_size=int(config["experts"]["inference_batch_size"]),
            num_workers=0, persistent_workers=False, mixed_precision=False,
            fold_transform=fold_transform, epoch_selection="fixed",
        ).logits
        final_standardizer = fit_standardizer(train.signal, axis=0)
        train_values = temporal_input(train.signal, final_standardizer)
        validation_values = temporal_input(validation.signal, final_standardizer)
        result = train_expert(
            train_values, train.label, validation_values, validation.label,
            kind="temporal", seed=seed, epochs=epochs,
            batch_size=int(config["experts"]["batch_size"]), pos_weight=None,
            validation_batch_size=int(config["experts"]["inference_batch_size"]),
            num_workers=0, persistent_workers=False, mixed_precision=False,
            epoch_selection="fixed",
        )
        import torch
        loader = torch.utils.data.DataLoader(
            NumpyDataset(validation_values, validation.label),
            batch_size=int(config["experts"]["inference_batch_size"]), shuffle=False,
            num_workers=0, pin_memory=True,
        )
        oof_probability = 1.0 / (1.0 + np.exp(-np.clip(oof, -40, 40)))
        validation_probability = 1.0 / (1.0 + np.exp(-np.clip(predict_logits(result.model, loader), -40, 40)))
        threshold = select_threshold(train.label, oof_probability)
        rows.append(_prediction_rows(train_base, oof_probability, threshold, dataset="matlab", split="train_oof", method="temporal", seed=seed))
        rows.append(_prediction_rows(validation_base, validation_probability, threshold, dataset="matlab", split="validation", method="temporal", seed=seed))
        logger.info("MATLAB clean temporal seed %d ready: MCC=%.6f threshold=%.3f", seed, binary_metrics(validation.label, validation_probability, threshold)["mcc"], threshold)
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keep = ~((source["dataset"].astype(str) == "matlab") & (source["method"].astype(str) == "temporal"))
    merged = pd.concat([source.loc[keep], *rows], ignore_index=True)
    merged.to_parquet(output / "clean_development_predictions.parquet", index=False)
    payload = {
        "status": "PASS", "seeds": list(DEVELOPMENT_SEEDS), "epochs": epochs,
        "selection": "fixed", "normalization": "fold_local_train_only",
        "va1_used_for_epoch_selection": False, "source_sha256": sha256_file(source_path(config, source_override)),
        "prediction_rows": int(len(merged)), "elapsed_seconds": time.perf_counter() - started,
    }
    finish(output, "matlab_clean", payload)
    logger.info("MATLAB clean temporal PASS: merged rows=%d", len(merged))


def _thresholded(probabilities: pd.DataFrame, thresholds: dict[str, float], method: str) -> np.ndarray:
    return (probabilities[f"probability_{method}"].to_numpy(float) >= float(thresholds[method])).astype(np.int64)


def stage_development_fusion(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if complete(output, "development_fusion"):
        logger.info("development fusion already complete; reusing OOF-only selection")
        return
    if not complete(output, "matlab_clean"):
        raise RuntimeError("Run matlab_clean before development_fusion")
    started = time.perf_counter()
    source = pd.read_parquet(output / "clean_development_predictions.parquet")
    validate_prediction_source(source, expected_seeds=DEVELOPMENT_SEEDS, allow_holdouts=False)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    statistic_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for dataset in ("matlab", "vsb"):
        dataset_metrics: list[dict[str, Any]] = []
        per_seed_statistics: dict[str, list[dict[str, Any]]] = {}
        for seed in DEVELOPMENT_SEEDS:
            train_pair = canonical_pair_frame(source, dataset=dataset, seed=seed, split="train_oof")
            validation_pair = canonical_pair_frame(source, dataset=dataset, seed=seed, split="validation")
            oof_probabilities, components, folds = cross_fitted_meta_oof(train_pair, dataset=dataset, seed=seed, n_splits=5)
            from partial_discharge_adaptive_fusion.current_expert_fusion import _apply_components
            validation_probabilities = validation_pair[["sample_id", "id_measurement", "phase", "target"]].copy()
            applied = _apply_components(components, validation_pair["probability_temporal"].to_numpy(float), validation_pair["probability_global_spectrogram"].to_numpy(float))
            for method, values in applied.items():
                validation_probabilities[f"probability_{method}"] = values
            validation_probabilities["probability_best_individual"] = validation_probabilities[f"probability_{components.best_individual}"]
            thresholds = component_thresholds(components)
            metric_rows.extend(evaluate_pair_methods(validation_pair, validation_probabilities, split="validation", seed=seed, thresholds=thresholds, dataset=dataset))
            dataset_metrics.extend([row for row in metric_rows if row["dataset"] == dataset and row["seed"] == seed])
            oof_probabilities["dataset"] = dataset
            oof_probabilities["seed"] = int(seed)
            oof_probabilities["split"] = "meta_oof"
            prediction_rows.append(oof_probabilities)
            validation_probabilities["dataset"] = dataset
            validation_probabilities["seed"] = int(seed)
            validation_probabilities["split"] = "validation"
            for method, value in thresholds.items():
                validation_probabilities[f"threshold_{method}"] = float(value)
            prediction_rows.append(validation_probabilities)
            stats = paired_statistics(validation_pair, validation_probabilities, dataset=dataset, seed=seed, thresholds=thresholds, iterations=int(config["statistics"]["paired_bootstrap_iterations"]))
            statistic_rows.extend(stats)
            for row in stats:
                per_seed_statistics.setdefault(row["comparison"], []).append(row)
            logger.info("%s seed %d fusion development evaluated; best individual=%s", dataset, seed, components.best_individual)
        metric_frame = pd.DataFrame([row for row in metric_rows if row["dataset"] == dataset])
        expert = expert_gate(
            metric_frame,
            dataset=dataset,
            vsb_mean_min=float(config["gates"]["vsb_spectrogram_mean_mcc_min"]),
            vsb_seed_min=float(config["gates"]["vsb_spectrogram_seed_mcc_min"]),
            matlab_mean_min=float(config["gates"]["matlab_expert_mean_mcc_min"]),
            matlab_seed_min=float(config["gates"]["matlab_expert_seed_mcc_min"]),
        )
        delta_summary: dict[str, Any] = {}
        for comparison in ("best_fixed_minus_best_individual", "adaptive_minus_best_fixed", "adaptive_minus_best_individual"):
            left, right = {
                "best_fixed_minus_best_individual": ("best_fixed", "best_individual"),
                "adaptive_minus_best_fixed": ("adaptive", "best_fixed"),
                "adaptive_minus_best_individual": ("adaptive", "best_individual"),
            }[comparison]
            seed_deltas = []
            bootstrap_records = []
            for seed in DEVELOPMENT_SEEDS:
                metrics = metric_frame[metric_frame["seed"] == seed].set_index("method")
                seed_deltas.append(float(metrics.loc[left, "mcc"] - metrics.loc[right, "mcc"]))
                current = [frame for frame in prediction_rows if frame["dataset"].iloc[0] == dataset and int(frame["seed"].iloc[0]) == seed and frame["split"].iloc[0] == "validation"][0]
                labels = current["target"].to_numpy(np.int64)
                thresholds = {method: float(current[f"threshold_{method}"].iloc[0]) for method in ALL_METHODS if f"threshold_{method}" in current}
                bootstrap_records.append({"labels": labels, "prediction_a": _thresholded(current, thresholds, left), "prediction_b": _thresholded(current, thresholds, right), "group_ids": current["id_measurement"].astype(str).to_numpy()})
            delta_summary[comparison] = {
                "seed_deltas": seed_deltas,
                "hierarchical_ci": hierarchical_delta_ci(bootstrap_records, dataset=dataset, comparison=comparison, iterations=int(config["statistics"]["paired_bootstrap_iterations"]), seed=42042),
            }
        gate = fusion_gate(delta_summary, minimum_delta=float(config["gates"]["fusion_mean_delta_mcc_min"]), minimum_positive_seeds=int(config["gates"]["fusion_positive_development_seeds_min"]), ci_lower_bound=float(config["gates"]["fusion_ci_lower_bound_min"]))
        summaries[dataset] = {"expert_gate": expert, "fusion_gate": gate, "deltas": delta_summary}
        logger.info("%s development gate: experts=%s fusion=%s", dataset, expert["eligible"], gate["eligible"])
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "development_metrics.csv", index=False)
    write_frame(output / "development_predictions.parquet", pd.concat(prediction_rows, ignore_index=True))
    write_frame(output / "development_statistics.csv", pd.DataFrame(statistic_rows))
    payload = {"status": "PASS", "datasets": summaries, "eligible_datasets": [dataset for dataset, value in summaries.items() if value["expert_gate"]["eligible"] and value["fusion_gate"]["eligible"]], "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "development_gate.json", payload)
    finish(output, "development_fusion", payload)


def stage_freeze(config: dict[str, Any], config_path: Path, output: Path, logger: logging.Logger) -> None:
    if complete(output, "freeze"):
        logger.info("freeze already complete; reusing frozen protocol")
        return
    if not complete(output, "development_fusion"):
        raise RuntimeError("Run development_fusion before freeze")
    gate = json.loads((output / "development_gate.json").read_text(encoding="utf-8"))
    eligible = gate.get("eligible_datasets", [])
    if not eligible:
        raise RuntimeError("No dataset passed the development fusion gate; holdouts remain locked")
    frozen = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen["protocol_status"] = "frozen"
    frozen["freeze_record"] = {
        "source_commit": "d90c959 ancestry",
        "source_predictions_sha256": sha256_file(ROOT / config["sources"]["development_predictions"]),
        "development_fusion_gate": gate,
        "eligible_datasets": eligible,
        "historical_metrics_used_for_selection": False,
        "holdouts_opened_during_freeze": [],
    }
    frozen_path = output / "frozen_config.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    payload = {"status": "PASS", "eligible_datasets": eligible, "frozen_config": str(frozen_path), "holdouts_opened": []}
    finish(output, "freeze", payload)
    logger.info("freeze PASS: eligible datasets=%s; holdouts remain closed", ",".join(eligible))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def stage_report(config: dict[str, Any], output: Path, logger: logging.Logger) -> None:
    if not complete(output, "development_fusion"):
        raise RuntimeError("Run development_fusion before report")
    metrics = pd.read_csv(output / "development_metrics.csv")
    gate = json.loads((output / "development_gate.json").read_text(encoding="utf-8"))
    report_path = ROOT / config["outputs"]["report"]
    summary_path = ROOT / config["outputs"]["summary"]
    manifest_path = ROOT / config["outputs"]["manifest"]
    eligible = gate.get("eligible_datasets", [])
    verdict = "DEVELOPMENT_GATE_PASSED" if eligible else "DEVELOPMENT_GATE_FAILED_HOLDOUTS_LOCKED"
    summary = {
        "verdict": verdict,
        "eligible_datasets": eligible,
        "holdouts_opened": [],
        "historical_metrics_used_for_selection": False,
        "metrics": metrics.to_dict(orient="records"),
        "development_gate": gate,
        "source": json.loads((output / "preflight.json").read_text(encoding="utf-8")),
    }
    write_json(summary_path, summary)
    report = [
        "# Current-Expert Adaptive Fusion Confirmation",
        "",
        f"**Development verdict:** `{verdict}`",
        "",
        "## OBSERVED",
        "",
        "This report uses only current full-signal expert predictions and a clean MATLAB temporal regeneration with fixed nine epochs and fold-local normalization.",
        "Historical V3/V4/V5 metrics are provenance context only and do not enter selection, calibration, thresholding, fusion, or the verdict.",
        "",
        "VSB grouped holdout, official VSB test, MATLAB Te1, and MATLAB Te2 remain closed in this stage.",
        "",
        "### Development metrics",
        "",
        _markdown_table(metrics.round(6)),
        "",
        "### Gates",
        "",
        _markdown_table(pd.DataFrame([{"dataset": dataset, "expert_gate": value["expert_gate"]["eligible"], "fusion_gate": value["fusion_gate"]["eligible"]} for dataset, value in gate["datasets"].items()])),
        "",
        "## INTERPRETATION",
        "",
        "A dataset may open its locked holdout only after its expert and fusion development gates pass. The primary confirmatory contrasts are best_fixed minus best_individual, adaptive minus best_fixed, and adaptive minus best_individual.",
        "",
        "## UNRESOLVED",
        "",
        "Confirmatory five-seed holdout evaluation has not been opened by this development report.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "# Current Expert Adaptive Fusion Manifest\n\n"
        f"- Source prediction SHA-256: `{summary['source']['source_sha256']}`\n"
        f"- Config SHA-256: `{summary['source']['config_sha256']}`\n"
        f"- Holdouts opened: `none`\n"
        f"- Historical metrics used for selection: `false`\n"
        f"- Development verdict: `{verdict}`\n",
        encoding="utf-8",
    )
    logger.info("report written: %s", report_path)


def _run_confirmatory_handoff(output: Path, source_path_value: Path, logger: logging.Logger) -> None:
    """Evaluate a fresh all-development five-seed expert handoff.

    Expert generation is deliberately explicit: this stage consumes a parquet
    produced after freeze and validates that it contains only the eligible
    holdout split. It cannot be reached by the development all command.
    """
    frozen = yaml.safe_load((output / "frozen_config.yaml").read_text(encoding="utf-8"))
    eligible = set(frozen.get("freeze_record", {}).get("eligible_datasets", []))
    if not eligible:
        raise RuntimeError("No dataset was eligible at freeze; holdouts remain locked")
    if not source_path_value.is_file():
        raise FileNotFoundError(f"Fresh confirmatory expert handoff is missing: {source_path_value}")
    source = pd.read_parquet(source_path_value)
    validate_prediction_source(source, expected_seeds=(42, 43, 44, 45, 46), allow_holdouts=True)
    holdout_splits = {"matlab": "test_confirmatory", "vsb": "test_grouped_holdout"}
    holdout_rows = source[source["split"].isin(["test", "test_confirmatory", "test_grouped_holdout"])]
    allowed_holdout_pairs = {(dataset, holdout_splits[dataset]) for dataset in eligible}
    observed_holdout_pairs = set(zip(holdout_rows["dataset"].astype(str), holdout_rows["split"].astype(str)))
    if observed_holdout_pairs - allowed_holdout_pairs:
        raise RuntimeError("Confirmatory source contains a holdout for an ineligible dataset or an unexpected split")
    if not observed_holdout_pairs:
        raise RuntimeError("Confirmatory source contains no eligible holdout rows")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for dataset in sorted(eligible):
        split = holdout_splits[dataset]
        for seed in (42, 43, 44, 45, 46):
            train_pair = canonical_pair_frame(source, dataset=dataset, seed=seed, split="train_oof")
            holdout_pair = canonical_pair_frame(source, dataset=dataset, seed=seed, split=split)
            _, components, _ = cross_fitted_meta_oof(train_pair, dataset=dataset, seed=seed, n_splits=5)
            applied = _apply_components(
                components,
                holdout_pair["probability_temporal"].to_numpy(float),
                holdout_pair["probability_global_spectrogram"].to_numpy(float),
            )
            probability_frame = holdout_pair[["sample_id", "id_measurement", "phase", "target"]].copy()
            for method, values in applied.items():
                probability_frame[f"probability_{method}"] = values
            probability_frame["probability_best_individual"] = probability_frame[f"probability_{components.best_individual}"]
            thresholds = component_thresholds(components)
            metric_rows.extend(evaluate_pair_methods(holdout_pair, probability_frame, split=split, seed=seed, thresholds=thresholds, dataset=dataset))
            probability_frame["dataset"] = dataset
            probability_frame["seed"] = int(seed)
            probability_frame["split"] = split
            for method, threshold in thresholds.items():
                probability_frame[f"threshold_{method}"] = float(threshold)
            prediction_rows.append(probability_frame)
            logger.info("%s confirmatory seed %d evaluated on %s", dataset, seed, split)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_rows, ignore_index=True)
    write_frame(output / "confirmatory_metrics.csv", metrics_frame)
    write_frame(output / "confirmatory_predictions.parquet", predictions_frame)
    payload = {
        "status": "PASS", "datasets": sorted(eligible), "seeds": [42, 43, 44, 45, 46],
        "source_path": str(source_path_value), "source_sha256": sha256_file(source_path_value),
        "holdout_splits": holdout_splits, "holdouts_opened": sorted(eligible),
        "metrics_rows": int(len(metrics_frame)),
    }
    write_json(output / "confirmatory_summary.json", payload | {"metrics": metrics_frame.to_dict(orient="records")})
    finish(output, "confirmatory", payload)
    logger.info("confirmatory PASS: datasets=%s rows=%d", ",".join(sorted(eligible)), len(metrics_frame))


def stage_confirmatory(config: dict[str, Any], output: Path, source_path_value: Path | None, logger: logging.Logger) -> None:
    """Guard the future confirmatory entry point.

    The implementation intentionally refuses to open a holdout unless the
    frozen config and dataset-specific gate exist.  Expert generation for the
    selected holdout is a separate explicit stage and cannot be triggered by
    a development ``--stage all`` invocation.
    """

    if not complete(output, "freeze"):
        raise RuntimeError("Confirmatory evaluation is locked: run and pass the development freeze first")
    frozen = yaml.safe_load((output / "frozen_config.yaml").read_text(encoding="utf-8"))
    if frozen.get("protocol_status") != "frozen":
        raise RuntimeError("Frozen configuration marker is invalid")
    source_path_value = source_path_value or (output / "confirmatory_predictions.parquet")
    _run_confirmatory_handoff(output, source_path_value, logger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--source-predictions", type=Path, default=None)
    parser.add_argument("--confirmatory-source", type=Path, default=None)
    parser.add_argument("--stage", choices=("preflight", "matlab_clean", "development_fusion", "freeze", "report", "confirmatory", "all"), default="all")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    raw_root_value = raw_data_root(args.raw_root)
    output = output_root(config)
    logger = logger_setup(args.log_file)
    stages = {
        "preflight": lambda: stage_preflight(config, args.config, raw_root_value, output, args.source_predictions, logger),
        "matlab_clean": lambda: stage_matlab_clean(config, raw_root_value, output, args.source_predictions, logger),
        "development_fusion": lambda: stage_development_fusion(config, output, logger),
        "freeze": lambda: stage_freeze(config, args.config, output, logger),
        "report": lambda: stage_report(config, output, logger),
        "confirmatory": lambda: stage_confirmatory(config, output, args.confirmatory_source, logger),
    }
    order = ["preflight", "matlab_clean", "development_fusion", "freeze", "report"] if args.stage == "all" else [args.stage]
    for stage in order:
        logger.info("stage %s started", stage)
        stages[stage]()
        logger.info("stage %s finished", stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
