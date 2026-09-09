"""Metrics and paired statistical utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & ((probability < upper) if upper < 1 else (probability <= upper))
        if mask.any():
            result += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    return float(result)


def binary_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
        f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(labels, probability))
    except ValueError:
        roc_auc = float("nan")
    return {
        "threshold": float(threshold), "mcc": float(matthews_corrcoef(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall_pd": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "roc_auc": roc_auc,
        "brier": float(brier_score_loss(labels, probability)),
        "ece_10_bins": expected_calibration_error(labels, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "errors": int(fp + fn),
    }


def fast_mcc(labels: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    tp = np.sum((labels == 1) & (prediction == 1))
    tn = np.sum((labels == 0) & (prediction == 0))
    fp = np.sum((labels == 0) & (prediction == 1))
    fn = np.sum((labels == 1) & (prediction == 0))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def paired_bootstrap_delta(
    labels: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Compute paired sample bootstrap for one seed and two methods."""

    labels = np.asarray(labels)
    prediction_a = np.asarray(prediction_a)
    prediction_b = np.asarray(prediction_b)
    if len(labels) == 0 or len(prediction_a) != len(labels) or len(prediction_b) != len(labels):
        raise ValueError("Paired bootstrap inputs must be non-empty and equally sized.")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    if not all(np.isin(values, [0, 1]).all() for values in (labels, prediction_a, prediction_b)):
        raise ValueError("Paired bootstrap currently requires binary labels and predictions.")
    rng = np.random.default_rng(seed)
    # A multinomial count per joint (label, prediction_a, prediction_b) state is
    # exactly equivalent to resampling indices with replacement, but avoids
    # allocating 10,000 copies of a million-row MATLAB test set.
    categories = labels.astype(np.int64) * 4 + np.asarray(prediction_a, dtype=np.int64) * 2 + np.asarray(prediction_b, dtype=np.int64)
    observed_counts = np.bincount(categories, minlength=8).astype(np.float64)
    probabilities = observed_counts / len(labels)
    values = np.empty(iterations, dtype=np.float64)
    offset = 0
    chunk_size = 256
    while offset < iterations:
        count = min(chunk_size, iterations - offset)
        counts = rng.multinomial(len(labels), probabilities, size=count).astype(np.float64)
        labels_one = counts[:, 4:8].sum(axis=1)
        labels_zero = counts[:, :4].sum(axis=1)
        tp_a = counts[:, 6] + counts[:, 7]
        fp_a = counts[:, 2] + counts[:, 3]
        fn_a = labels_one - tp_a
        tn_a = labels_zero - fp_a
        tp_b = counts[:, 5] + counts[:, 7]
        fp_b = counts[:, 1] + counts[:, 3]
        fn_b = labels_one - tp_b
        tn_b = labels_zero - fp_b
        den_a = np.sqrt((tp_a + fp_a) * (tp_a + fn_a) * (tn_a + fp_a) * (tn_a + fn_a))
        den_b = np.sqrt((tp_b + fp_b) * (tp_b + fn_b) * (tn_b + fp_b) * (tn_b + fn_b))
        mcc_a = np.divide(tp_a * tn_a - fp_a * fn_a, den_a, out=np.zeros_like(den_a), where=den_a != 0)
        mcc_b = np.divide(tp_b * tn_b - fp_b * fn_b, den_b, out=np.zeros_like(den_b), where=den_b != 0)
        values[offset:offset + count] = mcc_a - mcc_b
        offset += count
    return {
        "point_estimate": fast_mcc(labels, prediction_a) - fast_mcc(labels, prediction_b),
        "bootstrap_mean": float(values.mean()), "bootstrap_median": float(np.median(values)),
        "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "fraction_gt_zero": float(np.mean(values > 0)), "iterations": int(iterations),
    }


def mcnemar_counts(labels: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict[str, int | float]:
    from scipy.stats import binomtest

    labels = np.asarray(labels)
    correct_a = np.asarray(prediction_a) == labels
    correct_b = np.asarray(prediction_b) == labels
    b = int(np.sum(correct_b & ~correct_a))
    c = int(np.sum(~correct_b & correct_a))
    p_value = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"a_wrong_b_correct": b, "a_correct_b_wrong": c, "exact_two_sided_p": p_value}


def summarize_seed_deltas(deltas: Iterable[float]) -> dict[str, float | int]:
    """Summarize one paired method delta per independent training seed."""

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one seed delta is required.")
    return {
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "positive_seeds": int(np.sum(values > 0)),
        "target_reached_seeds": int(np.sum(values >= 0.005)),
    }


def _group_bootstrap_values(
    labels: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    group_ids: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    prediction_a = np.asarray(prediction_a, dtype=np.int64)
    prediction_b = np.asarray(prediction_b, dtype=np.int64)
    group_ids = np.asarray(group_ids).astype(str)
    if len(labels) == 0 or not (len(labels) == len(prediction_a) == len(prediction_b) == len(group_ids)):
        raise ValueError("Grouped bootstrap inputs must be non-empty and equally sized")
    if not all(np.isin(values, [0, 1]).all() for values in (labels, prediction_a, prediction_b)):
        raise ValueError("Grouped bootstrap currently requires binary labels and predictions")
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    categories = labels * 4 + prediction_a * 2 + prediction_b
    group_categories = np.zeros((len(unique_groups), 8), dtype=np.float64)
    np.add.at(group_categories, (inverse, categories), 1.0)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    probabilities = np.full(len(unique_groups), 1.0 / len(unique_groups), dtype=np.float64)
    offset = 0
    while offset < iterations:
        count = min(128, iterations - offset)
        sampled = rng.multinomial(len(unique_groups), probabilities, size=count).astype(np.float64)
        state = sampled @ group_categories
        labels_one = state[:, 4:8].sum(axis=1)
        labels_zero = state[:, :4].sum(axis=1)
        tp_a = state[:, 6] + state[:, 7]
        fp_a = state[:, 2] + state[:, 3]
        fn_a = labels_one - tp_a
        tn_a = labels_zero - fp_a
        tp_b = state[:, 5] + state[:, 7]
        fp_b = state[:, 1] + state[:, 3]
        fn_b = labels_one - tp_b
        tn_b = labels_zero - fp_b
        den_a = np.sqrt((tp_a + fp_a) * (tp_a + fn_a) * (tn_a + fp_a) * (tn_a + fn_a))
        den_b = np.sqrt((tp_b + fp_b) * (tp_b + fn_b) * (tn_b + fp_b) * (tn_b + fn_b))
        mcc_a = np.divide(tp_a * tn_a - fp_a * fn_a, den_a, out=np.zeros_like(den_a), where=den_a != 0)
        mcc_b = np.divide(tp_b * tn_b - fp_b * fn_b, den_b, out=np.zeros_like(den_b), where=den_b != 0)
        values[offset:offset + count] = mcc_a - mcc_b
        offset += count
    return values


def grouped_paired_bootstrap_delta(
    labels: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    group_ids: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap a paired MCC delta by resampling complete measurement groups."""

    values = _group_bootstrap_values(
        labels, prediction_a, prediction_b, group_ids, iterations=iterations, seed=seed,
    )
    labels = np.asarray(labels, dtype=np.int64)
    prediction_a = np.asarray(prediction_a, dtype=np.int64)
    prediction_b = np.asarray(prediction_b, dtype=np.int64)
    return {
        "point_estimate": fast_mcc(labels, prediction_a) - fast_mcc(labels, prediction_b),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_median": float(np.median(values)),
        "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "fraction_gt_zero": float(np.mean(values > 0)),
        "iterations": int(iterations),
        "group_count": int(np.unique(np.asarray(group_ids).astype(str)).size),
    }


def hierarchical_grouped_delta_ci(
    seed_predictions: Iterable[dict[str, np.ndarray]],
    comparison: str,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Resample groups independently within seeds and average seed deltas."""

    records = list(seed_predictions)
    if not records:
        raise ValueError("At least one seed prediction record is required")
    child_seeds = np.random.SeedSequence(seed).spawn(len(records))
    values = []
    points = []
    for record, child in zip(records, child_seeds):
        labels = np.asarray(record["labels"], dtype=np.int64)
        prediction_a = np.asarray(record["prediction_a"], dtype=np.int64)
        prediction_b = np.asarray(record["prediction_b"], dtype=np.int64)
        group_ids = np.asarray(record["group_ids"])
        values.append(_group_bootstrap_values(
            labels, prediction_a, prediction_b, group_ids,
            iterations=iterations, seed=int(child.generate_state(1)[0]),
        ))
        points.append(fast_mcc(labels, prediction_a) - fast_mcc(labels, prediction_b))
    aggregate = np.mean(np.vstack(values), axis=0)
    return {
        "comparison": comparison,
        "point_estimate": float(np.mean(points)),
        "bootstrap_mean": float(aggregate.mean()),
        "ci_95": [float(np.quantile(aggregate, 0.025)), float(np.quantile(aggregate, 0.975))],
        "fraction_gt_zero": float(np.mean(aggregate > 0)),
        "iterations": int(iterations),
        "seed_count": int(len(records)),
    }
