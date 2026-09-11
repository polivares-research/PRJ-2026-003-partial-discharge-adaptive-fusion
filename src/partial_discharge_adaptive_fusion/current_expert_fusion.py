"""Leakage-safe fusion of the current temporal and global-spectrogram experts.

This module is intentionally independent of the historical V3/V4/V5 fusion
outputs.  It treats the current expert predictions as an immutable source,
performs all meta-learning by cross-fitting over expert OOF predictions, and
keeps MATLAB signals and VSB measurements as separate statistical units.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import (
    binary_metrics,
    grouped_paired_bootstrap_delta,
    hierarchical_grouped_delta_ci,
    hierarchical_paired_delta_ci,
    mcnemar_counts,
    paired_bootstrap_delta,
)
from .fusion import normalized_entropy, select_threshold
from .modeling.train import fold_assignments


METHODS = ("temporal", "global_spectrogram")
FUSION_METHODS = ("fixed_50_50", "best_fixed", "adaptive", "conservative")
ALL_METHODS = METHODS + ("best_individual",) + FUSION_METHODS
REQUIRED_SOURCE_COLUMNS = {
    "sample_id", "id_measurement", "phase", "target", "dataset", "split",
    "method", "seed", "probability", "threshold", "prediction",
}
THRESHOLDS = np.arange(0.05, 0.951, 0.005)
WEIGHTS = np.arange(0.0, 1.001, 0.05)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of an immutable source artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_source(
    frame: pd.DataFrame,
    *,
    expected_seeds: Iterable[int] = (42, 43, 44),
    allow_holdouts: bool = False,
) -> dict[str, Any]:
    """Validate the current long-form expert prediction artifact.

    The validation intentionally rejects historical/test rows by default.  It
    also rejects duplicate parent-method rows, non-finite probabilities, and
    incomplete temporal/spectrogram pairs.
    """

    missing = REQUIRED_SOURCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction source is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction source is empty")
    source = frame.copy()
    for column in ("sample_id", "id_measurement", "dataset", "split", "method"):
        source[column] = source[column].astype(str)
    source["seed"] = source["seed"].astype(int)
    if not source["method"].isin(METHODS).all():
        values = sorted(source.loc[~source["method"].isin(METHODS), "method"].unique())
        raise ValueError(f"Unexpected expert methods in source: {values}")
    if not allow_holdouts and source["split"].isin(
        ["test", "test_confirmatory", "test_grouped_holdout", "official_unlabeled_test"]
    ).any():
        raise ValueError("Holdout rows are forbidden in development expert sources")
    allowed_splits = {"train_oof", "validation"}
    if allow_holdouts:
        allowed_splits |= {"test", "test_confirmatory", "test_grouped_holdout"}
    if not source["split"].isin(sorted(allowed_splits)).all():
        raise ValueError(
            "Prediction source contains an unsupported split; allowed splits are "
            f"{sorted(allowed_splits)}"
        )
    seeds = set(int(value) for value in expected_seeds)
    if set(source["seed"].unique()) != seeds:
        raise ValueError(f"Expected seeds {sorted(seeds)}, found {sorted(source['seed'].unique())}")
    if not np.isfinite(source["probability"].to_numpy(float)).all():
        raise ValueError("Expert probabilities contain non-finite values")
    if not ((source["probability"].to_numpy(float) >= 0.0) & (source["probability"].to_numpy(float) <= 1.0)).all():
        raise ValueError("Expert probabilities must be in [0, 1]")
    if source["target"].isna().any() or not source["target"].isin([0, 1]).all():
        raise ValueError("Expert targets must be finite binary values")
    key = ["dataset", "seed", "split", "sample_id", "method"]
    if source.duplicated(key).any():
        raise ValueError("Duplicate dataset/seed/split/sample/method rows in expert source")
    pair_counts = source.groupby(["dataset", "seed", "split", "sample_id"])["method"].nunique()
    if not (pair_counts == 2).all():
        raise ValueError("Every parent signal must have exactly two expert predictions")
    label_counts = source.groupby(["dataset", "seed", "split", "sample_id"])["target"].nunique()
    if (label_counts != 1).any():
        raise ValueError("Expert pair has inconsistent parent targets")
    groups = source[["dataset", "seed", "split", "sample_id", "id_measurement", "phase"]].drop_duplicates()
    return {
        "status": "PASS",
        "rows": int(len(source)),
        "datasets": sorted(source["dataset"].unique().tolist()),
        "seeds": sorted(source["seed"].unique().tolist()),
        "splits": sorted(source["split"].unique().tolist()),
        "parent_rows": int(len(groups)),
        "holdouts_allowed": bool(allow_holdouts),
    }


def canonical_pair_frame(source: pd.DataFrame, *, dataset: str, seed: int, split: str) -> pd.DataFrame:
    """Collapse long expert rows to one aligned parent-signal row."""

    current = source[(source["dataset"].astype(str) == str(dataset)) & (source["seed"].astype(int) == int(seed)) & (source["split"].astype(str) == str(split))].copy()
    if current.empty:
        raise ValueError(f"No source rows for dataset={dataset}, seed={seed}, split={split}")
    current["method"] = current["method"].astype(str)
    if set(current["method"].unique()) != set(METHODS):
        raise ValueError("Canonical pair requires temporal and global_spectrogram methods")
    current = current.sort_values(["sample_id", "method"])
    if current.duplicated(["sample_id", "method"]).any():
        raise ValueError("Duplicate parent expert rows")
    base = current.drop_duplicates("sample_id")[["sample_id", "id_measurement", "phase", "target"]].copy()
    base["sample_id"] = base["sample_id"].astype(str)
    base["id_measurement"] = base["id_measurement"].astype(str)
    for method in METHODS:
        part = current[current["method"] == method].set_index("sample_id")
        if set(part.index) != set(base["sample_id"]):
            raise ValueError("Temporal and spectrogram parent IDs are not identical")
        part = part.reindex(base["sample_id"])
        base[f"probability_{method}"] = part["probability"].to_numpy(float)
        base[f"source_threshold_{method}"] = part["threshold"].to_numpy(float)
    if not np.isfinite(base[[f"probability_{method}" for method in METHODS]].to_numpy(float)).all():
        raise ValueError("Canonical pair probabilities are non-finite")
    return base.reset_index(drop=True)


def meta_fold_assignments(labels: np.ndarray, groups: np.ndarray | None, *, dataset: str, seed: int, n_splits: int = 5) -> np.ndarray:
    """Build deterministic folds for fusion meta-learning."""

    return fold_assignments(
        np.asarray(labels, dtype=np.int64),
        groups=None if dataset == "matlab" else np.asarray(groups).astype(str),
        n_splits=n_splits,
        seed=10000 + int(seed),
    )


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _fit_platt(probability: np.ndarray, labels: np.ndarray, seed: int) -> Any:
    model = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    x = _logit(probability).reshape(-1, 1)
    if np.unique(labels).size < 2:
        return DummyClassifier(strategy="constant", constant=int(labels[0])).fit(x, labels)
    return model.fit(x, np.asarray(labels, dtype=np.int64))


def _platt_predict(model: Any, probability: np.ndarray) -> np.ndarray:
    result = np.asarray(model.predict_proba(_logit(probability).reshape(-1, 1))[:, -1], dtype=float)
    return np.clip(result, 0.0, 1.0)


def reliability_features(temporal: np.ndarray, spectrogram: np.ndarray) -> np.ndarray:
    """Build fixed signal-level reliability features for both datasets."""

    temporal = np.clip(np.asarray(temporal, dtype=float), 1e-6, 1.0 - 1e-6)
    spectrogram = np.clip(np.asarray(spectrogram, dtype=float), 1e-6, 1.0 - 1e-6)
    agreement = ((temporal >= 0.5) == (spectrogram >= 0.5)).astype(float)
    return np.column_stack([
        temporal, spectrogram, _logit(temporal), _logit(spectrogram),
        np.abs(temporal - 0.5), np.abs(spectrogram - 0.5),
        normalized_entropy(temporal), normalized_entropy(spectrogram),
        np.abs(temporal - spectrogram), temporal - spectrogram, agreement,
    ]).astype(np.float64)


def _fit_reliability(features: np.ndarray, target: np.ndarray, seed: int) -> Any:
    if np.unique(target).size < 2:
        return DummyClassifier(strategy="constant", constant=int(target[0])).fit(features, target)
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=seed)),
    ]).fit(features, target)


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    index = np.flatnonzero(classes == 1)
    return np.full(len(features), float(classes[0] == 1)) if probabilities.shape[1] == 1 else probabilities[:, int(index[0])]


def _adaptive_probability(temporal: np.ndarray, spectrogram: np.ndarray, rt: np.ndarray, rs: np.ndarray) -> np.ndarray:
    denominator = np.asarray(rt, dtype=float) + np.asarray(rs, dtype=float) + 1e-8
    return np.clip((np.asarray(rt) * temporal + np.asarray(rs) * spectrogram) / denominator, 0.0, 1.0)


def _best_fixed(labels: np.ndarray, temporal: np.ndarray, spectrogram: np.ndarray) -> dict[str, float]:
    rows = []
    for weight in WEIGHTS:
        probability = weight * temporal + (1.0 - weight) * spectrogram
        threshold = select_threshold(labels, probability)
        metrics = binary_metrics(labels, probability, threshold)
        rows.append({"weight_temporal": float(weight), "threshold": float(threshold), "mcc": float(metrics["mcc"])})
    return max(rows, key=lambda row: (row["mcc"], -abs(row["weight_temporal"] - 0.5)))


@dataclass(frozen=True)
class MetaComponents:
    """Models and OOF-selected operating points for one seed/dataset."""

    temporal_calibrator: Any
    spectrogram_calibrator: Any
    temporal_threshold: float
    spectrogram_threshold: float
    fixed_50_threshold: float
    best_fixed_weight: float
    best_fixed_threshold: float
    adaptive_threshold: float
    conservative_delta: float
    conservative_threshold: float
    reliability_temporal: Any
    reliability_spectrogram: Any
    best_individual: str


def _fit_components(temporal: np.ndarray, spectrogram: np.ndarray, labels: np.ndarray, *, seed: int) -> MetaComponents:
    temporal_calibrator = _fit_platt(temporal, labels, seed)
    spectrogram_calibrator = _fit_platt(spectrogram, labels, seed + 1)
    t = _platt_predict(temporal_calibrator, temporal)
    s = _platt_predict(spectrogram_calibrator, spectrogram)
    t_threshold = select_threshold(labels, t)
    s_threshold = select_threshold(labels, s)
    fixed_50 = (t + s) / 2.0
    fixed_50_threshold = select_threshold(labels, fixed_50)
    fixed = _best_fixed(labels, t, s)
    features = reliability_features(t, s)
    rt_model = _fit_reliability(features, (t >= t_threshold).astype(int) == labels, seed + 2)
    rs_model = _fit_reliability(features, (s >= s_threshold).astype(int) == labels, seed + 3)
    rt = _positive_probability(rt_model, features)
    rs = _positive_probability(rs_model, features)
    adaptive = _adaptive_probability(t, s, rt, rs)
    adaptive_threshold = select_threshold(labels, adaptive)
    conservative_rows = []
    fixed_probability = fixed["weight_temporal"] * t + (1.0 - fixed["weight_temporal"]) * s
    for delta in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25):
        conservative = np.where(np.abs(rt - rs) < delta, fixed_probability, adaptive)
        threshold = select_threshold(labels, conservative)
        conservative_rows.append({"delta": delta, "threshold": threshold, "mcc": binary_metrics(labels, conservative, threshold)["mcc"]})
    selected_conservative = max(conservative_rows, key=lambda row: (row["mcc"], -row["delta"]))
    individual = {
        "temporal": binary_metrics(labels, t, t_threshold)["mcc"],
        "global_spectrogram": binary_metrics(labels, s, s_threshold)["mcc"],
    }
    return MetaComponents(
        temporal_calibrator, spectrogram_calibrator, float(t_threshold), float(s_threshold),
        float(fixed_50_threshold), float(fixed["weight_temporal"]), float(fixed["threshold"]),
        float(adaptive_threshold), float(selected_conservative["delta"]),
        float(selected_conservative["threshold"]), rt_model, rs_model,
        max(individual, key=individual.get),
    )


def _apply_components(components: MetaComponents, temporal: np.ndarray, spectrogram: np.ndarray) -> dict[str, np.ndarray]:
    t = _platt_predict(components.temporal_calibrator, temporal)
    s = _platt_predict(components.spectrogram_calibrator, spectrogram)
    features = reliability_features(t, s)
    rt = _positive_probability(components.reliability_temporal, features)
    rs = _positive_probability(components.reliability_spectrogram, features)
    fixed_50 = (t + s) / 2.0
    fixed = components.best_fixed_weight * t + (1.0 - components.best_fixed_weight) * s
    adaptive = _adaptive_probability(t, s, rt, rs)
    conservative = np.where(np.abs(rt - rs) < components.conservative_delta, fixed, adaptive)
    return {
        "temporal": t, "global_spectrogram": s, "fixed_50_50": fixed_50,
        "best_fixed": fixed, "adaptive": adaptive, "conservative": conservative,
    }


def cross_fitted_meta_oof(pair: pd.DataFrame, *, dataset: str, seed: int, n_splits: int = 5) -> tuple[pd.DataFrame, MetaComponents, np.ndarray]:
    """Create cross-fitted fusion OOF predictions and final OOF components."""

    labels = pair["target"].to_numpy(np.int64)
    temporal = pair["probability_temporal"].to_numpy(float)
    spectrogram = pair["probability_global_spectrogram"].to_numpy(float)
    groups = pair["id_measurement"].astype(str).to_numpy()
    folds = meta_fold_assignments(labels, groups, dataset=dataset, seed=seed, n_splits=n_splits)
    methods = {name: np.full(len(pair), np.nan, dtype=float) for name in FUSION_METHODS + METHODS}
    for fold in np.unique(folds):
        fit = folds != fold
        holdout = folds == fold
        components = _fit_components(temporal[fit], spectrogram[fit], labels[fit], seed=seed + int(fold) * 97)
        applied = _apply_components(components, temporal[holdout], spectrogram[holdout])
        for name, values in applied.items():
            methods[name][holdout] = values
    components = _fit_components(temporal, spectrogram, labels, seed=seed)
    # Keep individual expert columns cross-fitted. The all-OOF components are
    # returned separately for validation/holdout application.
    best = max(
        METHODS,
        key=lambda name: binary_metrics(
            labels, methods[name], select_threshold(labels, methods[name])
        )["mcc"],
    )
    methods["best_individual"] = methods[best]
    result = pair[["sample_id", "id_measurement", "phase", "target"]].copy()
    for name, values in methods.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Cross-fitted meta predictions incomplete for {name}")
        result[f"probability_{name}"] = values
    return result, components, folds


def component_thresholds(components: MetaComponents) -> dict[str, float]:
    """Return operating thresholds selected from OOF-only component fitting."""

    return {
        "temporal": components.temporal_threshold,
        "global_spectrogram": components.spectrogram_threshold,
        "fixed_50_50": components.fixed_50_threshold,
        "best_fixed": components.best_fixed_threshold,
        "adaptive": components.adaptive_threshold,
        "conservative": components.conservative_threshold,
        "best_individual": components.temporal_threshold if components.best_individual == "temporal" else components.spectrogram_threshold,
    }


def evaluate_pair_methods(
    pair: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    split: str,
    seed: int,
    thresholds: dict[str, float] | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Return complete metrics for every current expert/fusion method."""

    labels = pair["target"].to_numpy(np.int64)
    rows = []
    for method in ALL_METHODS:
        values = probabilities[f"probability_{method}"].to_numpy(float)
        threshold = float((thresholds or {}).get(method, 0.5))
        rows.append({"dataset": str(dataset or "unknown"), "seed": int(seed), "split": split, "method": method, **binary_metrics(labels, values, threshold)})
    return rows


def paired_statistics(
    pair: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    thresholds: dict[str, float] | None = None,
    iterations: int = 10000,
) -> list[dict[str, Any]]:
    """Calculate paired bootstrap and McNemar statistics for primary contrasts."""

    labels = pair["target"].to_numpy(np.int64)
    groups = pair["id_measurement"].astype(str).to_numpy()
    thresholds = thresholds or {method: 0.5 for method in ALL_METHODS}
    predictions = {method: (probabilities[f"probability_{method}"].to_numpy(float) >= float(thresholds.get(method, 0.5))).astype(np.int64) for method in ALL_METHODS}
    comparisons = (("best_fixed", "best_individual"), ("adaptive", "best_fixed"), ("adaptive", "best_individual"))
    rows = []
    for left, right in comparisons:
        if dataset == "vsb":
            bootstrap = grouped_paired_bootstrap_delta(labels, predictions[left], predictions[right], groups, iterations=iterations, seed=42042 + seed)
        else:
            bootstrap = paired_bootstrap_delta(labels, predictions[left], predictions[right], iterations=iterations, seed=42042 + seed)
        rows.append({"dataset": dataset, "seed": int(seed), "comparison": f"{left}_minus_{right}", **bootstrap, "mcnemar": mcnemar_counts(labels, predictions[left], predictions[right])})
    return rows


def hierarchical_delta_ci(records: Iterable[dict[str, Any]], *, dataset: str, comparison: str, iterations: int = 10000, seed: int = 42042) -> dict[str, Any]:
    """Aggregate paired deltas across independent seeds without pooling datasets."""

    records = list(records)
    if dataset == "vsb":
        return hierarchical_grouped_delta_ci(records, comparison, iterations=iterations, seed=seed)
    return hierarchical_paired_delta_ci(records, comparison, iterations=iterations, seed=seed)


def expert_gate(metrics: pd.DataFrame, *, dataset: str, vsb_mean_min: float = 0.55, vsb_seed_min: float = 0.50, matlab_mean_min: float = 0.90, matlab_seed_min: float = 0.85) -> dict[str, Any]:
    """Apply the predeclared expert eligibility gate to validation MCC."""

    current = metrics[(metrics["dataset"] == dataset) & (metrics["split"] == "validation")]
    values = {method: current[current["method"] == method]["mcc"].astype(float).tolist() for method in METHODS}
    if any(len(values[method]) != 3 for method in METHODS):
        raise ValueError(f"Expert gate requires three validation seeds for {dataset}")
    if dataset == "vsb":
        eligible = float(np.mean(values["global_spectrogram"])) >= vsb_mean_min and min(values["global_spectrogram"]) >= vsb_seed_min
    else:
        eligible = all(float(np.mean(values[method])) >= matlab_mean_min and min(values[method]) >= matlab_seed_min for method in METHODS)
    return {"dataset": dataset, "eligible": bool(eligible), "mcc_by_method": values, "mean_mcc": {method: float(np.mean(value)) for method, value in values.items()}}


def fusion_gate(delta_summary: dict[str, Any], *, minimum_delta: float = 0.005, minimum_positive_seeds: int = 2, ci_lower_bound: float = 0.0) -> dict[str, Any]:
    """Require OOF-selected development evidence before opening a holdout."""

    candidates = []
    for name in ("best_fixed_minus_best_individual", "adaptive_minus_best_fixed", "adaptive_minus_best_individual"):
        record = delta_summary.get(name, {})
        seed_values = [float(value) for value in record.get("seed_deltas", [])]
        ci = record.get("hierarchical_ci", {}).get("ci_95", [float("-inf"), float("-inf")])
        mean_delta = float(np.mean(seed_values)) if seed_values else float("nan")
        passed = bool(
            seed_values
            and mean_delta >= minimum_delta
            and sum(value > 0 for value in seed_values) >= minimum_positive_seeds
            and float(ci[0]) > ci_lower_bound
        )
        candidates.append({"comparison": name, "passed": passed, "mean_delta": mean_delta, "positive_seeds": int(sum(value > 0 for value in seed_values)), "ci_95": ci})
    return {"eligible": any(row["passed"] for row in candidates), "candidates": candidates}


__all__ = [
    "ALL_METHODS", "FUSION_METHODS", "METHODS", "MetaComponents", "canonical_pair_frame",
    "component_thresholds", "cross_fitted_meta_oof", "evaluate_pair_methods", "expert_gate", "fusion_gate",
    "hierarchical_delta_ci", "meta_fold_assignments", "paired_statistics", "reliability_features",
    "sha256_file", "validate_prediction_source",
]
