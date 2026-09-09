"""Reusable evaluation and protocol guards for the cross-dataset diagnostic."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .evaluation import fast_mcc
from .fusion import select_threshold


ALLOWED_VERDICTS = (
    "TEMPORAL+SPECTROGRAM SUPPORTED",
    "SPECTROGRAM SUPPORTED ONLY ON VSB",
    "SPECTROGRAM SUPPORTED ONLY ON MATLAB",
    "SPECTROGRAM NOT COMPLEMENTARY",
    "SPECTROGRAM NOT SUPPORTED",
    "INCONCLUSIVE",
)


def aggregate_event_predictions(
    probabilities: np.ndarray,
    mask: np.ndarray,
    *,
    top_k_fraction: float = 0.10,
) -> np.ndarray:
    """Aggregate masked event probabilities to exactly one parent probability."""

    values = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[1] != 2 or valid.shape != values.shape:
        raise ValueError("Expected probabilities and mask with shape [batch, 2, events]")
    if not 0.0 < top_k_fraction <= 1.0 or not np.isfinite(values).all():
        raise ValueError("Invalid event probabilities or top-k fraction")
    if not valid.any(axis=2).all():
        raise ValueError("Each half-cycle must contain at least one valid event")
    capacity = values.shape[2]
    k = max(1, int(np.ceil(capacity * top_k_fraction)))
    pooled: list[np.ndarray] = []
    for half in range(2):
        masked = np.where(valid[:, half], values[:, half], -np.inf)
        sorted_values = np.sort(masked, axis=1)[:, ::-1]
        valid_count = valid[:, half].sum(axis=1)
        count = np.minimum(valid_count, k)
        selected = np.where(np.isfinite(sorted_values[:, :k]), sorted_values[:, :k], 0.0)
        pooled.append(selected.sum(axis=1) / count)
    result = np.mean(np.stack(pooled, axis=1), axis=1)
    if not ((result >= 0.0) & (result <= 1.0)).all():
        raise RuntimeError("Aggregated event probabilities left [0,1]")
    return result


def prediction_overlap_oracle(
    labels: np.ndarray,
    temporal_prediction: np.ndarray,
    spectrogram_prediction: np.ndarray,
    group_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return overlap states and oracle headroom for aligned parent predictions."""

    y = np.asarray(labels, dtype=np.int64)
    temporal = np.asarray(temporal_prediction, dtype=np.int64)
    spectrogram = np.asarray(spectrogram_prediction, dtype=np.int64)
    if len(y) == 0 or not (len(y) == len(temporal) == len(spectrogram)):
        raise ValueError("Prediction arrays must be non-empty and aligned")
    if group_ids is not None and len(group_ids) != len(y):
        raise ValueError("group_ids must align with predictions")
    correct_t = temporal == y
    correct_s = spectrogram == y
    oracle = np.where(correct_t, temporal, spectrogram)
    states = {
        "both_correct": correct_t & correct_s,
        "temporal_only": correct_t & ~correct_s,
        "spectrogram_only": ~correct_t & correct_s,
        "both_wrong": ~correct_t & ~correct_s,
    }
    rows: list[dict[str, Any]] = []
    for stratum, stratum_mask in (("all", np.ones(len(y), bool)), ("PD", y == 1), ("NonPD", y == 0)):
        denominator = int(stratum_mask.sum())
        for state, state_mask in states.items():
            count = int((stratum_mask & state_mask).sum())
            rows.append({"stratum": stratum, "state": state, "count": count, "fraction": count / denominator if denominator else 0.0})
    return {
        "rows": rows,
        "disagreement_rate": float(np.mean(temporal != spectrogram)),
        "oracle_mcc": fast_mcc(y, oracle),
        "temporal_mcc": fast_mcc(y, temporal),
        "spectrogram_mcc": fast_mcc(y, spectrogram),
        "stronger_individual_mcc": max(fast_mcc(y, temporal), fast_mcc(y, spectrogram)),
        "oracle_headroom": fast_mcc(y, oracle) - max(fast_mcc(y, temporal), fast_mcc(y, spectrogram)),
        "group_count": int(np.unique(np.asarray(group_ids).astype(str)).size) if group_ids is not None else len(y),
    }


def fixed_probability_diagnostics(
    labels: np.ndarray,
    temporal_probability: np.ndarray,
    spectrogram_probability: np.ndarray,
    oof_temporal: np.ndarray,
    oof_spectrogram: np.ndarray,
    weights: Iterable[float] = tuple(np.arange(0.0, 1.001, 0.05)),
) -> list[dict[str, float]]:
    """Evaluate fixed probability mixtures; these results cannot determine a verdict."""

    y = np.asarray(labels, dtype=np.int64)
    t = np.asarray(temporal_probability, dtype=float)
    s = np.asarray(spectrogram_probability, dtype=float)
    if not (len(y) == len(t) == len(s)):
        raise ValueError("Validation probabilities are not aligned")
    oof_y = np.asarray(y if len(oof_temporal) == len(y) else np.zeros(0), dtype=np.int64)
    if len(oof_temporal) != len(oof_spectrogram):
        raise ValueError("OOF probability arrays are not aligned")
    rows = []
    for weight in weights:
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weights must be in [0,1]")
        validation_probability = weight * t + (1.0 - weight) * s
        if len(oof_temporal) == len(y):
            oof_probability = weight * np.asarray(oof_temporal) + (1.0 - weight) * np.asarray(oof_spectrogram)
            threshold = select_threshold(oof_y, oof_probability)
        else:
            threshold = 0.5
        rows.append({"temporal_weight": weight, "threshold": float(threshold), "mcc": fast_mcc(y, validation_probability >= threshold)})
    return rows


def validate_parent_predictions(frame: Any, predictions: Any, *, expected_rows: int | None = None) -> None:
    """Require one finite probability for every original signal identity."""

    if len(frame) != len(predictions) or (expected_rows is not None and len(predictions) != expected_rows):
        raise ValueError("Prediction row count does not match parent signals")
    if frame["sample_id"].astype(str).duplicated().any() or predictions["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate parent signal IDs in prediction alignment")
    left = set(frame["sample_id"].astype(str))
    right = set(predictions["sample_id"].astype(str))
    if left != right:
        raise ValueError("Prediction IDs do not match original signals")
    probability = np.asarray(predictions["probability"], dtype=float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Predictions must be finite probabilities in [0,1]")


def classify_cross_dataset_verdict(summary: dict[str, Any]) -> str:
    """Apply the registered cross-dataset spectrogram decision rules."""

    if not summary.get("integrity_ok", False) or not summary.get("regression_ok", False):
        return "INCONCLUSIVE"
    vsb_strong = summary.get("vsb_spectrogram_strong", False)
    matlab_strong = summary.get("matlab_spectrogram_strong", False)
    complementarity = summary.get("complementarity_supported", False)
    if vsb_strong and matlab_strong and complementarity:
        return "TEMPORAL+SPECTROGRAM SUPPORTED"
    if vsb_strong and not matlab_strong:
        return "SPECTROGRAM SUPPORTED ONLY ON VSB"
    if matlab_strong and not vsb_strong:
        return "SPECTROGRAM SUPPORTED ONLY ON MATLAB"
    if (vsb_strong or matlab_strong) and not complementarity:
        return "SPECTROGRAM NOT COMPLEMENTARY"
    if summary.get("spectrogram_evidence", False):
        return "SPECTROGRAM NOT SUPPORTED"
    return "SPECTROGRAM NOT SUPPORTED"


__all__ = [
    "ALLOWED_VERDICTS", "aggregate_event_predictions", "prediction_overlap_oracle",
    "fixed_probability_diagnostics", "validate_parent_predictions", "classify_cross_dataset_verdict",
]
