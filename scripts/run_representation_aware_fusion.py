"""Run signal-level fusion and paired evaluation for the revised experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from partial_discharge_adaptive_fusion.evaluation import (
    binary_metrics, mcnemar_counts, paired_bootstrap_delta, summarize_seed_deltas,
)
from partial_discharge_adaptive_fusion.fusion import (
    adaptive_probability, conservative_probability, fit_reliability_models,
    fit_temperature, select_fixed_weight, select_threshold, signal_level_reliability_features,
    sigmoid, predict_reliability,
)
from partial_discharge_adaptive_fusion.protocol import assert_protocol_frozen, load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json


DATASETS = {
    "matlab": "engineering-partial-discharge-noise-signals",
    "vsb": "engineering-vsb-power-line-fault-detection",
}
METHODS = ("temporal", "cwt", "fixed_50_50", "best_fixed", "adaptive", "conservative", "oracle")


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(values / (1.0 - values))


def _summary_frame(frame: pd.DataFrame, prefix: str) -> dict[str, np.ndarray] | None:
    marker = f"{prefix}_window_"
    columns = sorted(column for column in frame.columns if column.startswith(marker))
    if not columns:
        return None
    return {column.removeprefix(marker): frame[column].to_numpy(np.float64) for column in columns}


def _features(frame: pd.DataFrame, *, extended: bool) -> np.ndarray:
    temporal = frame["probability_temporal"].to_numpy(np.float64)
    cwt = frame["probability_cwt"].to_numpy(np.float64)
    return signal_level_reliability_features(
        _logit(temporal), _logit(cwt), temporal, cwt,
        window_summaries_temporal=_summary_frame(frame, "temporal") if extended else None,
        window_summaries_cwt=_summary_frame(frame, "cwt") if extended else None,
    )


def _complementarity(labels: np.ndarray, temporal: np.ndarray, cwt: np.ndarray) -> dict[str, float | int]:
    """Summarize disagreement without changing any selection decision."""

    labels = np.asarray(labels, dtype=np.int64)
    temporal_prediction = np.asarray(temporal) >= 0.5
    cwt_prediction = np.asarray(cwt) >= 0.5
    temporal_correct = temporal_prediction == (labels == 1)
    cwt_correct = cwt_prediction == (labels == 1)
    correlation = np.corrcoef(np.asarray(temporal), np.asarray(cwt))[0, 1]
    return {
        "probability_pearson": float(np.nan_to_num(correlation, nan=0.0)),
        "prediction_disagreement_rate": float(np.mean(temporal_prediction != cwt_prediction)),
        "temporal_only_correct": int(np.sum(temporal_correct & ~cwt_correct)),
        "cwt_only_correct": int(np.sum(cwt_correct & ~temporal_correct)),
        "both_correct": int(np.sum(temporal_correct & cwt_correct)),
        "both_wrong": int(np.sum(~temporal_correct & ~cwt_correct)),
    }


def _reliability_selection(train: pd.DataFrame, validation: pd.DataFrame, *, seed: int, model_names: list[str]):
    labels_train = train["label"].to_numpy(np.int64)
    labels_validation = validation["label"].to_numpy(np.int64)
    candidates = []
    for extended in (False, True):
        train_features = _features(train, extended=extended)
        validation_features = _features(validation, extended=extended)
        for model_name in model_names:
            models = fit_reliability_models(
                train_features, labels_train,
                train["probability_temporal"].to_numpy(np.float64),
                train["probability_cwt"].to_numpy(np.float64),
                model_name=model_name, seed=seed,
            )
            rt, rc = predict_reliability(models, validation_features)
            target_t = ((validation["probability_temporal"].to_numpy() >= 0.5) == (labels_validation == 1)).astype(float)
            target_c = ((validation["probability_cwt"].to_numpy() >= 0.5) == (labels_validation == 1)).astype(float)
            brier = float((np.mean((rt - target_t) ** 2) + np.mean((rc - target_c) ** 2)) / 2.0)
            try:
                from sklearn.metrics import roc_auc_score
                auc = float((roc_auc_score(target_t, rt) + roc_auc_score(target_c, rc)) / 2.0)
            except ValueError:
                auc = 0.5
            candidates.append({
                "extended_window_features": extended, "model_name": model_name,
                "validation_brier": brier, "validation_roc_auc": auc, "models": models,
                "train_features": train_features, "validation_features": validation_features,
            })
    baseline = [row for row in candidates if not row["extended_window_features"]]
    extended = [row for row in candidates if row["extended_window_features"]]
    best_baseline = min(baseline, key=lambda row: (row["validation_brier"], -row["validation_roc_auc"]))
    best_extended = min(extended, key=lambda row: (row["validation_brier"], -row["validation_roc_auc"]))
    selected = best_extended if best_extended["validation_brier"] < best_baseline["validation_brier"] else best_baseline
    return selected, {
        "selected_feature_mode": "proposed_window_statistics" if selected["extended_window_features"] else "base_signal_level_features",
        "selected_model": selected["model_name"],
        "baseline": {key: value for key, value in best_baseline.items() if key in {"model_name", "validation_brier", "validation_roc_auc"}},
        "extended": {key: value for key, value in best_extended.items() if key in {"model_name", "validation_brier", "validation_roc_auc"}},
        "features_label": "PROPOSED DUE TO MULTI-INSTANCE VSB REPRESENTATION" if selected["extended_window_features"] else "BASE SIGNAL-LEVEL FEATURES",
    }


def _evaluate_seed(frame: pd.DataFrame, config: dict, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    train = frame.loc[frame["split"] == "train_oof"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    test_split = "test_confirmatory" if (frame["split"] == "test_confirmatory").any() else "test_grouped_holdout"
    test = frame.loc[frame["split"] == test_split].reset_index(drop=True)
    for name, part in (("train_oof", train), ("validation", validation), (test_split, test)):
        if part.empty:
            raise ValueError(f"Missing required prediction split: {name}")
        if not np.isfinite(part[["probability_temporal", "probability_cwt"]].to_numpy()).all():
            raise ValueError(f"Non-finite signal-level probabilities in {name}.")
    temperatures = {
        "temporal": fit_temperature(_logit(validation["probability_temporal"]), validation["label"]),
        "cwt": fit_temperature(_logit(validation["probability_cwt"]), validation["label"]),
    }
    probabilities = {}
    for part_name, part in (("train", train), ("validation", validation), ("test", test)):
        probabilities[part_name] = {
            "temporal": sigmoid(_logit(part["probability_temporal"]) / temperatures["temporal"]),
            "cwt": sigmoid(_logit(part["probability_cwt"]) / temperatures["cwt"]),
        }
    p_train, p_validation, p_test = (probabilities[name] for name in ("train", "validation", "test"))
    labels_train = train["label"].to_numpy(np.int64)
    labels_validation = validation["label"].to_numpy(np.int64)
    labels_test = test["label"].to_numpy(np.int64)
    fixed_50_threshold = select_threshold(labels_validation, (p_validation["temporal"] + p_validation["cwt"]) / 2.0)
    best_fixed = select_fixed_weight(labels_validation, p_validation["temporal"], p_validation["cwt"])
    fixed_probabilities = {
        "train": best_fixed["weight_temporal"] * p_train["temporal"] + (1 - best_fixed["weight_temporal"]) * p_train["cwt"],
        "validation": best_fixed["weight_temporal"] * p_validation["temporal"] + (1 - best_fixed["weight_temporal"]) * p_validation["cwt"],
        "test": best_fixed["weight_temporal"] * p_test["temporal"] + (1 - best_fixed["weight_temporal"]) * p_test["cwt"],
    }
    fixed_50 = {
        name: (values["temporal"] + values["cwt"]) / 2.0 for name, values in probabilities.items()
    }
    selected_reliability, reliability_record = _reliability_selection(
        train, validation, seed=seed, model_names=list(config["fusion"]["reliability_models"]),
    )
    reliability = {}
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        reliability[name] = predict_reliability(selected_reliability["models"], _features(part, extended=selected_reliability["extended_window_features"]))
    adaptive = {
        name: adaptive_probability(values["temporal"], values["cwt"], reliability[name][0], reliability[name][1])[0]
        for name, values in probabilities.items()
    }
    adaptive_threshold = select_threshold(labels_validation, adaptive["validation"])
    conservative_rows = []
    for delta in config["fusion"]["conservative_delta_grid"]:
        value = conservative_probability(
            fixed_probabilities["validation"], adaptive["validation"],
            reliability["validation"][0], reliability["validation"][1], float(delta),
        )
        conservative_rows.append({"delta": float(delta), "mcc": binary_metrics(labels_validation, value, select_threshold(labels_validation, value))["mcc"]})
    selected_conservative = max(conservative_rows, key=lambda row: (row["mcc"], -row["delta"]))
    conservative = {
        name: conservative_probability(
            fixed_probabilities[name], adaptive[name], reliability[name][0], reliability[name][1], selected_conservative["delta"],
        ) for name in probabilities
    }
    thresholds = {
        "temporal": select_threshold(labels_validation, p_validation["temporal"]),
        "cwt": select_threshold(labels_validation, p_validation["cwt"]),
        "fixed_50_50": fixed_50_threshold, "best_fixed": best_fixed["threshold"],
        "adaptive": adaptive_threshold, "conservative": select_threshold(labels_validation, conservative["validation"]),
    }
    method_probabilities = {
        "temporal": {name: values["temporal"] for name, values in probabilities.items()},
        "cwt": {name: values["cwt"] for name, values in probabilities.items()},
        "fixed_50_50": fixed_50, "best_fixed": fixed_probabilities,
        "adaptive": adaptive, "conservative": conservative,
        "oracle": {
            name: np.where(
                ((values["temporal"] >= 0.5) == (part["label"].to_numpy() == 1)),
                values["temporal"] >= 0.5,
                values["cwt"] >= 0.5,
            ).astype(float)
            for name, values, part in (("train", p_train, train), ("validation", p_validation, validation), ("test", p_test, test))
        },
    }
    metric_rows: list[dict[str, object]] = []
    for split_name, labels, index in (("train_oof", labels_train, "train"), ("validation", labels_validation, "validation"), (test_split, labels_test, "test")):
        for method in METHODS:
            row = binary_metrics(labels, method_probabilities[method][index], thresholds.get(method, 0.5))
            metric_rows.append({"seed": seed, "split": split_name, "method": method, **row})
    return metric_rows, {
        "seed": seed, "test_split": test_split, "temperatures": temperatures,
        "thresholds": thresholds, "best_fixed": best_fixed,
        "reliability": reliability_record, "conservative": selected_conservative,
        "complementarity": {
            "validation": _complementarity(labels_validation, p_validation["temporal"], p_validation["cwt"]),
            "test": _complementarity(labels_test, p_test["temporal"], p_test["cwt"]),
        },
        "method_probabilities": method_probabilities,
    }


def _run_dataset(dataset_slug: str, config: dict, args) -> dict[str, object]:
    root = Path(config["outputs"]["results_root"]) / dataset_slug
    metric_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    seeds = config["seeds"] if args.seeds is None else args.seeds
    for seed in seeds:
        path = root / f"seed-{seed}" / "expert_predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing revised expert predictions: {path}")
        rows, record = _evaluate_seed(pd.read_parquet(path), config, int(seed))
        metric_rows.extend(rows)
        records.append(record)
    metrics = pd.DataFrame(metric_rows)
    output_root = Path(config["outputs"]["reports_root"]) / "metrics" / "v3-windowed"
    output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_root / f"{dataset_slug}_per_seed.csv", index=False)
    comparisons = {}
    test_split = "test_confirmatory" if (metrics["split"] == "test_confirmatory").any() else "test_grouped_holdout"
    for a, b in (("adaptive", "best_fixed"), ("adaptive", "fixed_50_50"), ("best_fixed", "temporal")):
        seed_deltas = []
        bootstrap = []
        mcnemar = []
        for record in records:
            seed = int(record["seed"])
            frame = pd.read_parquet(root / f"seed-{seed}" / "expert_predictions.parquet")
            test = frame.loc[frame["split"] == test_split]
            labels = test["label"].to_numpy(np.int64)
            pred_a = record["method_probabilities"][a]["test"] >= record["thresholds"].get(a, 0.5)
            pred_b = record["method_probabilities"][b]["test"] >= record["thresholds"].get(b, 0.5)
            delta = float(binary_metrics(labels, pred_a.astype(float), 0.5)["mcc"] - binary_metrics(labels, pred_b.astype(float), 0.5)["mcc"])
            seed_deltas.append(delta)
            bootstrap.append({"seed": seed, **paired_bootstrap_delta(labels, pred_a, pred_b, iterations=int(config["evaluation"]["paired_bootstrap_iterations_per_seed"]), seed=seed)})
            mcnemar.append({"seed": seed, **mcnemar_counts(labels, pred_a, pred_b)})
        comparisons[f"{a}_minus_{b}"] = {
            "seed_summary": summarize_seed_deltas(seed_deltas),
            "per_seed_bootstrap": bootstrap, "per_seed_mcnemar": mcnemar,
        }
    summary = {
        "dataset": DATASETS[dataset_slug], "config_version": config["config_version"],
        "metrics_csv": str(output_root / f"{dataset_slug}_per_seed.csv"),
        "comparisons": comparisons,
        "records": [{key: value for key, value in record.items() if key != "method_probabilities"} for record in records],
    }
    write_json(summary, output_root / f"{dataset_slug}_statistical_analysis.json")
    return summary


def _verdict(summaries: dict[str, dict[str, object]]) -> str:
    adaptive = [summaries[dataset]["comparisons"]["adaptive_minus_best_fixed"]["seed_summary"] for dataset in summaries]
    strong = all(item["positive_seeds"] >= 4 for item in adaptive) and all(
        all(float(row["ci_95"][0]) > 0 for row in summaries[dataset]["comparisons"]["adaptive_minus_best_fixed"]["per_seed_bootstrap"])
        for dataset in summaries
    )
    static = all(item["positive_seeds"] >= 4 for item in [summaries[dataset]["comparisons"]["best_fixed_minus_temporal"]["seed_summary"] for dataset in summaries])
    if strong:
        return "ADAPTIVE FUSION CONFIRMED ACROSS DATASETS"
    if static and any(item["positive_seeds"] >= 3 for item in adaptive):
        return "ADAPTIVE FUSION PARTIALLY SUPPORTED"
    if static:
        return "STATIC MULTIMODAL FUSION CONFIRMED; ADAPTIVE NOT CONFIRMED"
    if any(item["positive_seeds"] >= 3 for item in adaptive):
        return "DATASET-DEPENDENT FUSION BENEFIT"
    return "FUSION NOT CONFIRMED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml"))
    parser.add_argument("--dataset", choices=("matlab", "vsb", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    assert_protocol_frozen(config)
    slugs = ("matlab", "vsb") if args.dataset == "both" else (args.dataset,)
    summaries = {slug: _run_dataset(slug, config, args) for slug in slugs}
    output = Path(config["outputs"]["reports_root"]) / "statistical_analysis" / "v3-windowed"
    output.mkdir(parents=True, exist_ok=True)
    cross_dataset = {
        "datasets": {
            slug: {
                "dataset_id": DATASETS[slug],
                "test_split": summary["records"][0]["test_split"],
                "comparisons": summary["comparisons"],
            }
            for slug, summary in summaries.items()
        },
        "raw_samples_pooled": False,
        "verdict": _verdict(summaries) if len(summaries) == 2 else "DATASET-SPECIFIC-ONLY",
    }
    write_json(cross_dataset, output / "cross_dataset_summary.json")
    print(cross_dataset["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
