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
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(labels), len(labels))
        values[index] = fast_mcc(labels[sample], prediction_a[sample]) - fast_mcc(labels[sample], prediction_b[sample])
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
