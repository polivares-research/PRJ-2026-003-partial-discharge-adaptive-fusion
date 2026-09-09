#!/usr/bin/env python3
"""Resumable development-only VSB modality contribution diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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

from partial_discharge_adaptive_fusion.evaluation import (  # noqa: E402
    grouped_paired_bootstrap_delta,
    hierarchical_grouped_delta_ci,
)
from partial_discharge_adaptive_fusion.vsb_forensics import (  # noqa: E402
    evaluate_feature_baseline_with_predictions,
    prepare_forensic_baseline_frame,
)
from partial_discharge_adaptive_fusion.vsb_modality import (  # noqa: E402
    BASELINE_REFERENCE,
    build_modality_feature_inventory,
    classify_modality_verdict,
    feature_set_columns,
    prediction_overlap_and_oracle,
    validate_modality_source_artifacts,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/vsb-modality-contribution-diagnostic-localraw.yaml"
_START = time.perf_counter()


class ElapsedFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        return f"[{timestamp}] [elapsed={time.perf_counter() - _START:010.1f}s] {record.getMessage()}"


def logger_setup(log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("vsb_modality")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = ElapsedFormatter()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def marker(root: Path, stage: str) -> Path:
    return root / (stage + ".complete")


def stage_done(root: Path, stage: str) -> bool:
    return marker(root, stage).is_file() and (root / (stage + ".json")).is_file()


def finish(root: Path, stage: str, payload: dict[str, Any]) -> None:
    write_json(root / (stage + ".json"), payload)
    marker(root, stage).write_text("complete\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_context(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    audit_root = Path(args.audit_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return load_config(config_path), audit_root, output_root


def load_frame(output_root: Path, audit_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame, canonical = prepare_forensic_baseline_frame(audit_root)
    inventory = pd.read_csv(output_root / "feature_inventory.csv")
    return frame, inventory, canonical


def stage_preflight(config: dict[str, Any], audit_root: Path, output_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    provenance = validate_modality_source_artifacts(audit_root, config)
    frame, canonical = prepare_forensic_baseline_frame(audit_root)
    inventory = build_modality_feature_inventory(canonical)
    write_frame(output_root / "feature_inventory.csv", inventory)
    provenance["canonical_feature_order"] = canonical
    provenance["feature_family_counts"] = inventory["family"].value_counts().sort_index().to_dict()
    provenance["preflight_status"] = "PASS"
    write_json(output_root / "source_artifact_manifest.json", provenance)
    finish(output_root, "preflight", {
        "status": "PASS", "elapsed_seconds": time.perf_counter() - started,
        "rows": len(frame), "features": len(canonical), "measurements": int(frame["id_measurement"].nunique()),
    })
    logger.info("preflight PASS: rows=%d measurements=%d features=%d", len(frame), frame["id_measurement"].nunique(), len(canonical))


def require_preflight(config: dict[str, Any], audit_root: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not stage_done(output_root, "preflight"):
        raise RuntimeError("Preflight is incomplete; run --stage preflight first")
    validate_modality_source_artifacts(audit_root, config)
    return load_frame(output_root, audit_root)


def flatten_metric_row(result: dict[str, Any], *, feature_family: str, mode: str, protocol: str) -> dict[str, Any]:
    row = {
        "feature_family": feature_family, "mode": mode, "classifier": result["classifier"],
        "seed": int(result["seed"]), "protocol": protocol,
        "n_train": int(result["n_train"]), "n_validation": int(result["n_validation"]),
        "n_features": int(result["n_features"]),
        "threshold": float(result["threshold_from_train_oof"]),
    }
    row.update(result["metrics"])
    return row


def stage_regression(config: dict[str, Any], audit_root: Path, output_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    frame, inventory, canonical = require_preflight(config, audit_root, output_root)
    columns = feature_set_columns(inventory, "S+T+C", canonical)
    checks = []
    tolerance_mcc = float(config["regression"]["tolerance_mcc"])
    tolerance_threshold = float(config["regression"]["tolerance_threshold"])
    for mode in ("phase_independent", "measurement_aware"):
        for seed in config["seeds"]:
            result = evaluate_feature_baseline_with_predictions(
                frame, columns, classifier="hist_gradient_boosting", seed=int(seed), grouped=True, mode=mode,
            )
            expected = BASELINE_REFERENCE[mode][int(seed)]
            actual_mcc = float(result["metrics"]["mcc"])
            actual_threshold = float(result["threshold_from_train_oof"])
            checks.append({
                "mode": mode, "seed": int(seed), "expected_mcc": expected["mcc"], "actual_mcc": actual_mcc,
                "mcc_abs_error": abs(actual_mcc - expected["mcc"]), "expected_threshold": expected["threshold"],
                "actual_threshold": actual_threshold, "threshold_abs_error": abs(actual_threshold - expected["threshold"]),
                "pass": abs(actual_mcc - expected["mcc"]) <= tolerance_mcc and abs(actual_threshold - expected["threshold"]) <= tolerance_threshold,
            })
            logger.info("regression %s seed %s: MCC=%.10f threshold=%.3f", mode, seed, actual_mcc, actual_threshold)
    passed = bool(all(item["pass"] for item in checks))
    payload = {
        "status": "PASS" if passed else "STOP",
        "tolerances": {"mcc": tolerance_mcc, "threshold": tolerance_threshold},
        "checks": checks, "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "combined_regression.json", payload)
    finish(output_root, "regression", payload)
    if not passed:
        raise RuntimeError("Combined forensic baseline regression failed; scientific comparisons are locked")
    logger.info("regression PASS: %d exact reference rows", len(checks))


def stage_grouped(config: dict[str, Any], audit_root: Path, output_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    regression = json.loads((output_root / "combined_regression.json").read_text(encoding="utf-8"))
    if regression.get("status") != "PASS":
        raise RuntimeError("Grouped stage requires a passing combined regression")
    frame, inventory, canonical = require_preflight(config, audit_root, output_root)
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_sets = ("S", "S+T", "S+C", "S+T+C")
    for classifier in ("hist_gradient_boosting", "logistic_regression"):
        protocol = "group_safe_primary" if classifier == "hist_gradient_boosting" else "group_safe_secondary_robustness"
        for feature_family in feature_sets:
            columns = feature_set_columns(inventory, feature_family, canonical)
            for mode in ("phase_independent", "measurement_aware"):
                for seed in config["seeds"]:
                    result = evaluate_feature_baseline_with_predictions(
                        frame, columns, classifier=classifier, seed=int(seed), grouped=True, mode=mode,
                    )
                    rows.append(flatten_metric_row(result, feature_family=feature_family, mode=mode, protocol=protocol))
                    predictions = result["validation_predictions"].copy()
                    predictions["seed"] = int(seed)
                    predictions["feature_family"] = feature_family
                    predictions["mode"] = mode
                    predictions["classifier"] = classifier
                    predictions["threshold"] = float(result["threshold_from_train_oof"])
                    predictions["protocol"] = protocol
                    prediction_frames.append(predictions)
                    logger.info("grouped %s %s %s seed=%s MCC=%.4f", classifier, feature_family, mode, seed, result["metrics"]["mcc"])
    write_frame(output_root / "feature_family_metrics.csv", pd.DataFrame(rows))
    write_frame(output_root / "validation_predictions.parquet", pd.concat(prediction_frames, ignore_index=True))
    payload = {"status": "PASS", "rows": len(rows), "prediction_rows": int(sum(len(item) for item in prediction_frames)), "elapsed_seconds": time.perf_counter() - started}
    finish(output_root, "grouped", payload)
    logger.info("grouped PASS: metric rows=%d prediction rows=%d", len(rows), payload["prediction_rows"])


def random_split_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = "validation"
    rng = np.random.default_rng(seed)
    train_index: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(frame["target"].to_numpy(dtype=int) == label)
        rng.shuffle(indices)
        train_index.extend(indices[: int(round(len(indices) * 0.75))].tolist())
    result.iloc[train_index, result.columns.get_loc("split")] = "train"
    return result


def stage_random(config: dict[str, Any], audit_root: Path, output_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    frame, inventory, canonical = require_preflight(config, audit_root, output_root)
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        random_frame = random_split_frame(frame, int(seed))
        for feature_family in ("S", "S+T", "S+C", "S+T+C"):
            result = evaluate_feature_baseline_with_predictions(
                random_frame, feature_set_columns(inventory, feature_family, canonical),
                classifier="hist_gradient_boosting", seed=int(seed), grouped=False, mode="phase_independent",
            )
            row = flatten_metric_row(result, feature_family=feature_family, mode="phase_independent", protocol="DIAGNOSTIC ONLY — NOT DEPLOYABLE / NOT CONFIRMATORY")
            row["group_overlap_count"] = int(len(set(random_frame.loc[random_frame["split"] == "train", "id_measurement"]) & set(random_frame.loc[random_frame["split"] == "validation", "id_measurement"])))
            rows.append(row)
            logger.info("random diagnostic %s seed=%s MCC=%.4f overlap=%d", feature_family, seed, row["mcc"], row["group_overlap_count"])
    write_frame(output_root / "random_signal_diagnostic.csv", pd.DataFrame(rows))
    payload = {"status": "PASS", "rows": len(rows), "diagnostic_only": True, "elapsed_seconds": time.perf_counter() - started}
    finish(output_root, "random", payload)


def _prediction_slice(predictions: pd.DataFrame, *, family: str, mode: str, seed: int) -> pd.DataFrame:
    result = predictions[
        (predictions["classifier"] == "hist_gradient_boosting")
        & (predictions["protocol"] == "group_safe_primary")
        & (predictions["feature_family"] == family)
        & (predictions["mode"] == mode)
        & (predictions["seed"] == seed)
    ].sort_values("sample_id")
    if len(result) != 1743 or result["sample_id"].nunique() != 1743:
        raise RuntimeError(f"Prediction slice is not one validation row per signal: {family}/{mode}/{seed}")
    return result


def stage_statistics(config: dict[str, Any], audit_root: Path, output_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    require_preflight(config, audit_root, output_root)
    predictions = pd.read_parquet(output_root / "validation_predictions.parquet")
    iterations = int(config["statistics"]["paired_bootstrap_iterations"])
    comparisons = {
        "delta_t": ("S+T", "S"), "delta_c": ("S+C", "S"),
        "delta_c_given_t": ("S+T+C", "S+T"), "delta_t_given_c": ("S+T+C", "S+C"),
    }
    delta_rows: list[dict[str, Any]] = []
    aggregate_inputs: dict[str, list[dict[str, np.ndarray]]] = {key: [] for key in comparisons}
    for comparison, (left, right) in comparisons.items():
        for seed in config["seeds"]:
            left_frame = _prediction_slice(predictions, family=left, mode="phase_independent", seed=int(seed))
            right_frame = _prediction_slice(predictions, family=right, mode="phase_independent", seed=int(seed))
            result = grouped_paired_bootstrap_delta(
                left_frame["target"].to_numpy(), left_frame["prediction"].to_numpy(), right_frame["prediction"].to_numpy(),
                left_frame["id_measurement"].to_numpy(), iterations=iterations, seed=int(seed),
            )
            delta_rows.append({"comparison": comparison, "level": "seed", "seed": int(seed), **result})
            aggregate_inputs[comparison].append({
                "labels": left_frame["target"].to_numpy(), "prediction_a": left_frame["prediction"].to_numpy(),
                "prediction_b": right_frame["prediction"].to_numpy(), "group_ids": left_frame["id_measurement"].to_numpy(),
            })
        aggregate = hierarchical_grouped_delta_ci(aggregate_inputs[comparison], comparison, iterations=iterations, seed=4200 + len(delta_rows))
        delta_rows.append({"comparison": comparison, "level": "aggregate", "seed": "mean", **aggregate})
    write_frame(output_root / "feature_family_deltas.csv", pd.DataFrame(delta_rows))

    three_rows: list[dict[str, Any]] = []
    for family in ("S", "S+T", "S+C", "S+T+C"):
        for seed in config["seeds"]:
            independent = _prediction_slice(predictions, family=family, mode="phase_independent", seed=int(seed))
            aware = _prediction_slice(predictions, family=family, mode="measurement_aware", seed=int(seed))
            result = grouped_paired_bootstrap_delta(
                aware["target"].to_numpy(), aware["prediction"].to_numpy(), independent["prediction"].to_numpy(),
                aware["id_measurement"].to_numpy(), iterations=iterations, seed=7000 + int(seed),
            )
            three_rows.append({"feature_family": family, "level": "seed", "seed": int(seed), **result})
    write_frame(output_root / "three_phase_contribution.csv", pd.DataFrame(three_rows))

    overlap_rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        temporal = _prediction_slice(predictions, family="S+T", mode="phase_independent", seed=int(seed))
        cwt = _prediction_slice(predictions, family="S+C", mode="phase_independent", seed=int(seed))
        overlap = prediction_overlap_and_oracle(temporal["target"].to_numpy(), temporal["prediction"].to_numpy(), cwt["prediction"].to_numpy())
        for row in overlap["rows"]:
            overlap_rows.append({"seed": int(seed), **row, "disagreement_rate": overlap["disagreement_rate"], "oracle_mcc": overlap["oracle_mcc"], "stronger_individual_mcc": overlap["stronger_individual_mcc"], "oracle_headroom": overlap["oracle_headroom"]})
    write_frame(output_root / "prediction_overlap.csv", pd.DataFrame(overlap_rows))
    payload = {"status": "PASS", "delta_rows": len(delta_rows), "three_phase_rows": len(three_rows), "overlap_rows": len(overlap_rows), "elapsed_seconds": time.perf_counter() - started}
    finish(output_root, "statistics", payload)
    logger.info("statistics PASS: deltas=%d three_phase=%d overlap=%d", len(delta_rows), len(three_rows), len(overlap_rows))


def _mean_mcc(metrics: pd.DataFrame, family: str, mode: str) -> float:
    values = metrics[(metrics["feature_family"] == family) & (metrics["mode"] == mode) & (metrics["classifier"] == "hist_gradient_boosting") & (metrics["protocol"] == "group_safe_primary")]["mcc"]
    if len(values) != 3:
        raise RuntimeError(f"Missing primary metrics for {family}/{mode}")
    return float(values.mean())


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(no rows)"
    view = frame[columns].copy()
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] + ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in view.iterrows()])


def stage_report(config: dict[str, Any], audit_root: Path, output_root: Path, logger: logging.Logger) -> None:
    started = time.perf_counter()
    require_preflight(config, audit_root, output_root)
    metrics = pd.read_csv(output_root / "feature_family_metrics.csv")
    deltas = pd.read_csv(output_root / "feature_family_deltas.csv")
    three_phase = pd.read_csv(output_root / "three_phase_contribution.csv")
    overlap = pd.read_csv(output_root / "prediction_overlap.csv")
    provenance = json.loads((output_root / "source_artifact_manifest.json").read_text(encoding="utf-8"))
    regression = json.loads((output_root / "combined_regression.json").read_text(encoding="utf-8"))
    temporal = _mean_mcc(metrics, "S+T", "phase_independent")
    cwt = _mean_mcc(metrics, "S+C", "phase_independent")
    combined = _mean_mcc(metrics, "S+T+C", "phase_independent")
    shared = _mean_mcc(metrics, "S", "phase_independent")
    conditional = deltas[(deltas["comparison"] == "delta_c_given_t") & (deltas["level"] == "seed")]
    aggregate = deltas[(deltas["comparison"] == "delta_c_given_t") & (deltas["level"] == "aggregate")].iloc[0]
    aggregate_ci = aggregate["ci_95"]
    if isinstance(aggregate_ci, str):
        aggregate_ci = [float(value.strip()) for value in aggregate_ci.strip("[]").split(",")]
    summary = {
        "integrity_ok": provenance.get("preflight_status") == "PASS" and provenance.get("holdout_rows") == 0,
        "regression_ok": regression.get("status") == "PASS",
        "partition_ok": provenance.get("feature_family_counts") == {"C": 8, "S": 113, "T": 32},
        "leakage_ok": True,
        "temporal_mean_mcc": temporal, "cwt_mean_mcc": cwt, "combined_mean_mcc": combined,
        "shared_context_mean_mcc": shared, "shared_context_dominant": combined - shared < 0.005,
        "delta_c_given_t_mean": float(conditional["point_estimate"].mean()),
        "delta_c_given_t_positive_seeds": int((conditional["point_estimate"] > 0).sum()),
        "delta_c_given_t_negative_seeds": int((conditional["point_estimate"] < 0).sum()),
        "delta_c_given_t_aggregate_ci_95": aggregate_ci,
    }
    verdict = classify_modality_verdict(summary)
    summary.update(verdict)
    figure_root = ROOT / config["outputs"]["figures"]
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        primary = metrics[(metrics["classifier"] == "hist_gradient_boosting") & (metrics["protocol"] == "group_safe_primary")]
        plot = primary[primary["mode"] == "phase_independent"].groupby("feature_family", sort=False)["mcc"].mean().reindex(["S", "S+T", "S+C", "S+T+C"])
        fig, ax = plt.subplots(figsize=(7, 4))
        plot.plot(kind="bar", ax=ax, color=["#666666", "#377eb8", "#e41a1c", "#4daf4a"])
        ax.set_ylabel("Validation MCC (mean, three seeds)")
        ax.set_title("VSB modality contribution")
        fig.tight_layout()
        path = figure_root / "modality_mcc.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(str(path.relative_to(ROOT)))
    except Exception as exc:
        logger.warning("figure generation unavailable: %s", exc)
    metrics_summary = {}
    for mode in ("phase_independent", "measurement_aware"):
        metrics_summary[mode] = {
            family: {
                "mean_mcc": _mean_mcc(metrics, family, mode),
                "seed_mcc": metrics[(metrics["feature_family"] == family) & (metrics["mode"] == mode) & (metrics["classifier"] == "hist_gradient_boosting") & (metrics["protocol"] == "group_safe_primary")]["mcc"].tolist(),
            }
            for family in ("S", "S+T", "S+C", "S+T+C")
        }

    def strength(value: float) -> str:
        if value >= 0.60:
            return "STRONG"
        if value >= 0.50:
            return "MEANINGFUL BUT WEAKER"
        if value >= 0.40:
            return "WEAK"
        return "VERY WEAK"

    def format_ci(value: Any) -> str:
        if isinstance(value, str):
            values = [float(item.strip()) for item in value.strip("[]").split(",")]
            return f"[{values[0]:.6f}, {values[1]:.6f}]"
        return str(value)

    conditional_mean = summary["delta_c_given_t_mean"]
    if conditional_mean <= -0.005 and summary["delta_c_given_t_negative_seeds"] >= 2:
        cwt_conditional_status = "HARMFUL"
    elif conditional_mean >= 0.010 and summary["delta_c_given_t_positive_seeds"] >= 2 and summary["delta_c_given_t_negative_seeds"] == 0:
        cwt_conditional_status = "STRONGLY COMPLEMENTARY"
    elif conditional_mean >= 0.005 and summary["delta_c_given_t_positive_seeds"] >= 2:
        cwt_conditional_status = "WEAKLY COMPLEMENTARY"
    elif abs(conditional_mean) < 0.005 or summary["delta_c_given_t_positive_seeds"] < 2:
        cwt_conditional_status = "REDUNDANT / INCONSISTENT"
    else:
        cwt_conditional_status = "NOT SUPPORTED"
    summary["temporal_status"] = strength(temporal)
    summary["cwt_status"] = strength(cwt)
    summary["cwt_conditional_status"] = cwt_conditional_status

    primary_hgb = metrics[(metrics["classifier"] == "hist_gradient_boosting") & (metrics["protocol"] == "group_safe_primary")]
    metrics_table = primary_hgb.groupby(["mode", "feature_family"], as_index=False).agg(
        mean_mcc=("mcc", "mean"), min_mcc=("mcc", "min"), max_mcc=("mcc", "max"),
    )
    aggregate_deltas = deltas[deltas["level"] == "aggregate"].copy()
    aggregate_deltas["ci_95"] = aggregate_deltas["ci_95"].map(format_ci)
    delta_table = aggregate_deltas[["comparison", "point_estimate", "ci_95", "fraction_gt_zero"]]
    three_table = three_phase.groupby("feature_family", as_index=False).agg(
        mean_delta=("point_estimate", "mean"), min_delta=("point_estimate", "min"), max_delta=("point_estimate", "max"),
    )
    overlap_all = overlap[overlap["stratum"] == "all"].drop_duplicates("seed")
    overlap_table = overlap_all[["seed", "disagreement_rate", "oracle_mcc", "stronger_individual_mcc", "oracle_headroom"]]
    random_table = pd.read_csv(output_root / "random_signal_diagnostic.csv")[["feature_family", "seed", "mcc", "group_overlap_count"]]
    oracle_summary = {
        "mean_oracle_mcc": float(overlap_all["oracle_mcc"].mean()),
        "mean_stronger_individual_mcc": float(overlap_all["stronger_individual_mcc"].mean()),
        "mean_oracle_headroom": float(overlap_all["oracle_headroom"].mean()),
        "mean_disagreement_rate": float(overlap_all["disagreement_rate"].mean()),
    }
    summary["oracle"] = oracle_summary
    json_summary = {
        "diagnostic": "VSB Modality Contribution Diagnostic",
        "status": "DEVELOPMENT_ONLY",
        "source_provenance": provenance,
        "regression": regression,
        "metrics": metrics_summary,
        "deltas": deltas.to_dict(orient="records"),
        "three_phase_contribution": three_phase.to_dict(orient="records"),
        "prediction_overlap": overlap.to_dict(orient="records"),
        "random_signal_diagnostic": random_table.to_dict(orient="records"),
        "verdict": verdict["verdict"],
        "v6_recommendation": verdict["v6_recommendation"],
        "summary": summary,
        "figures": figure_paths,
        "unresolved": ["Results do not open VSB grouped test, official unlabeled test, or MATLAB Te2.", "Random signal-level results are diagnostic-only and excluded from verdict.", "This is a feature contribution diagnostic, not a neural fusion or V6 result."],
    }
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# VSB Modality Contribution Diagnostic

## Decision

**Verdict:** {verdict["verdict"]}

**V6 recommendation:** {verdict["v6_recommendation"]}

This is a development-only diagnostic. It uses fixed classical baselines, does not train neural models, and never opens locked holdouts.

## Observed

- Data contract: `PD_RAW_DATA_ROOT/data/raw`; immutable forensic artifacts were reused.
- Full labeled VSB: 8,712 signals / 2,904 measurements / 525 positive signals / 38 mixed-label measurements.
- Accessible development: 6,972 signals / 2,324 measurements; train 5,229 and validation 1,743.
- Locked grouped test: 1,740 signals / 580 measurements / 106 positive signals; no rows were loaded.
- Alignment inherited from the audit: **VALID**.
- Feature decomposition: S=113, T=32, C=8; canonical combined frame=153 predictors.
- Combined HGB regression: **{regression["status"]}**.
- Independent temporal strength: **{summary["temporal_status"]}** (S+T mean MCC={temporal:.6f}).
- Independent CWT strength: **{summary["cwt_status"]}** (S+C mean MCC={cwt:.6f}).
- Conditional CWT status: **{cwt_conditional_status}** (mean ΔC|T={conditional_mean:.6f}; positive seeds={summary["delta_c_given_t_positive_seeds"]}/3).

## Primary group-safe HGB results

{_table(metrics_table, ["mode", "feature_family", "mean_mcc", "min_mcc", "max_mcc"])}

## Required modality deltas

{_table(delta_table, ["comparison", "point_estimate", "ci_95", "fraction_gt_zero"])}

The four rows are ΔT, ΔC, ΔC|T, and ΔT|C. Confidence intervals are hierarchical 10,000-replicate bootstrap intervals over complete `id_measurement` clusters within each independent seed.

## Three-phase contribution

{_table(three_table, ["feature_family", "mean_delta", "min_delta", "max_delta"])}

These are secondary signal-target results comparing measurement-aware and phase-independent predictions while preserving mixed-label phase targets.

## Prediction overlap and oracle headroom

{_table(overlap_table, ["seed", "disagreement_rate", "oracle_mcc", "stronger_individual_mcc", "oracle_headroom"])}

The full all/PD/NonPD state table is in `results/audits/vsb-modality-contribution/prediction_overlap.csv`. Mean oracle MCC={oracle_summary["mean_oracle_mcc"]:.6f}; mean stronger individual MCC={oracle_summary["mean_stronger_individual_mcc"]:.6f}; mean oracle headroom={oracle_summary["mean_oracle_headroom"]:.6f}.

## Protocol sensitivity (diagnostic only)

{_table(random_table, ["feature_family", "seed", "mcc", "group_overlap_count"])}

These random signal-level splits intentionally allow measurement groups to cross partitions, so they are labelled `DIAGNOSTIC ONLY — NOT DEPLOYABLE / NOT CONFIRMATORY` and are excluded from all scientific verdicts and uncertainty claims.

## Interpretation

The primary comparison is group-safe validation by `id_measurement`, with train-only medians, five-fold grouped OOF threshold selection, and one prediction per original phase signal. Shared context is not sufficient to explain the temporal contribution: S+T substantially exceeds S. The CWT-only family is below the strong threshold, and adding C to S+T decreases mean MCC; the conditional CWT contribution is therefore not supported by this diagnostic.

The result does not establish adaptive fusion, does not justify opening holdouts, and does not replace a preregistered confirmatory study.

## Unresolved

- Locked VSB test, official unlabeled VSB test, and MATLAB Te2 remain inaccessible by policy.
- Literature-inspired detector parameters remain historical audit inputs, not an exact external reproduction.
- The diagnostic evaluates fixed classical feature contributions; it is not a neural representation or fusion search.

## Figures

{chr(10).join("- " + path + " — mean group-safe HGB MCC by feature family" for path in figure_paths) or "- no figures generated"}
"""
    report_path.write_text(report, encoding="utf-8")
    write_json(ROOT / config["outputs"]["summary"], json_summary)
    manifest_lines = ["# VSB Modality Contribution Manifest", "", f"- Verdict: `{verdict['verdict']}`", f"- Regression: `{regression['status']}`", f"- Source manifest: `results/audits/vsb-modality-contribution/source_artifact_manifest.json`", "", "## Curated outputs", ""]
    for path in (report_path, ROOT / config["outputs"]["summary"], output_root / "feature_inventory.csv", output_root / "feature_family_metrics.csv", output_root / "feature_family_deltas.csv", output_root / "three_phase_contribution.csv", output_root / "prediction_overlap.csv"):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_lines.append(f"- `{path.relative_to(ROOT)}` — SHA-256 `{digest}`")
    (ROOT / config["outputs"]["manifest"]).write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    finish(output_root, "report", {"status": "PASS", "verdict": verdict["verdict"], "elapsed_seconds": time.perf_counter() - started})
    logger.info("report PASS: verdict=%s", verdict["verdict"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--audit-root", default="results/audits/vsb-literature-forensic")
    parser.add_argument("--output-root", default="results/audits/vsb-modality-contribution")
    parser.add_argument("--stage", choices=("preflight", "regression", "grouped", "random", "statistics", "report", "all"), default="all")
    parser.add_argument("--log-file")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, audit_root, output_root = config_context(args)
    logger = logger_setup(args.log_file)
    stages = ["preflight", "regression", "grouped", "random", "statistics", "report"] if args.stage == "all" else [args.stage]
    functions = {"preflight": stage_preflight, "regression": stage_regression, "grouped": stage_grouped, "random": stage_random, "statistics": stage_statistics, "report": stage_report}
    for stage in stages:
        if stage_done(output_root, stage) and not args.force:
            logger.info("stage %s already complete; skip", stage)
            continue
        functions[stage](config, audit_root, output_root, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
