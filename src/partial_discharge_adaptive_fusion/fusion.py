"""Leakage-safe fixed, reliability-aware, and conservative fusion."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityModels:
    """Two correctness predictors fitted only from cross-fitted development data."""

    temporal: Any
    cwt: Any
    model_name: str
    seed: int


@dataclass(frozen=True)
class CentroidReference:
    """Class centroids fitted on a development training representation."""

    nonpd: np.ndarray
    pd: np.ndarray


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -40, 40)))


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))


def select_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    """Select a threshold by MCC on development labels only."""

    from sklearn.metrics import matthews_corrcoef

    thresholds = np.linspace(0.05, 0.95, 181)
    scores = np.asarray([matthews_corrcoef(labels, probability >= threshold) for threshold in thresholds])
    best = thresholds[np.isclose(scores, scores.max())]
    return float(best[np.argmin(np.abs(best - 0.5))])


def select_fixed_weight(labels: np.ndarray, temporal: np.ndarray, cwt: np.ndarray) -> dict[str, float]:
    """Select a global weight and threshold on validation data."""

    from sklearn.metrics import matthews_corrcoef

    rows = []
    for weight in np.linspace(0, 1, 21):
        probability = weight * temporal + (1 - weight) * cwt
        threshold = select_threshold(labels, probability)
        rows.append({
            "mcc": float(matthews_corrcoef(labels, probability >= threshold)),
            "weight_temporal": float(weight), "threshold": threshold,
        })
    return max(rows, key=lambda row: (row["mcc"], -abs(row["weight_temporal"] - 0.5)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on validation NLL."""

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
    nll = [np.mean(np.logaddexp(0, logits / temp) - labels * logits / temp) for temp in temperatures]
    return float(temperatures[int(np.argmin(nll))])


def make_reliability_model(name: str, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed))])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    if name == "small_mlp":
        return Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-3, batch_size=256, max_iter=150, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=seed))])
    raise ValueError(f"Unknown reliability model: {name}")


def fit_reliability_models(
    features: np.ndarray,
    labels: np.ndarray,
    temporal_probability: np.ndarray,
    cwt_probability: np.ndarray,
    *,
    model_name: str,
    seed: int,
) -> ReliabilityModels:
    """Fit reliability as expert correctness on cross-fitted predictions."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    temporal_probability = np.asarray(temporal_probability, dtype=np.float64)
    cwt_probability = np.asarray(cwt_probability, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Reliability features and labels have incompatible shapes.")
    target_temporal = (temporal_probability >= 0.5).astype(np.int64) == labels
    target_cwt = (cwt_probability >= 0.5).astype(np.int64) == labels
    temporal = make_reliability_model(model_name, seed)
    cwt = make_reliability_model(model_name, seed + 1)
    temporal.fit(features, target_temporal.astype(np.int64))
    cwt.fit(features, target_cwt.astype(np.int64))
    return ReliabilityModels(temporal=temporal, cwt=cwt, model_name=model_name, seed=seed)


def predict_reliability(
    models: ReliabilityModels,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict bounded temporal/CWT reliability values."""

    features = np.asarray(features, dtype=np.float64)
    temporal = _positive_class_probability(models.temporal, features)
    cwt = _positive_class_probability(models.cwt, features)
    return np.clip(temporal, 0.0, 1.0), np.clip(cwt, 0.0, 1.0)


def _positive_class_probability(model: Any, features: np.ndarray) -> np.ndarray:
    """Read class-one probability even when a fold has one correctness class."""

    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if probabilities.shape[1] == 1:
        return np.full(len(features), float(classes[0] == 1), dtype=np.float64)
    class_one = np.flatnonzero(classes == 1)
    return probabilities[:, int(class_one[0])] if len(class_one) else np.zeros(len(features))


def reliability_features(
    logit_temporal: np.ndarray,
    logit_cwt: np.ndarray,
    probability_temporal: np.ndarray,
    probability_cwt: np.ndarray,
    distance_temporal: np.ndarray,
    distance_cwt: np.ndarray,
    signal_summary: np.ndarray,
) -> np.ndarray:
    """Build the fixed PoC4 reliability feature vector."""

    pt = np.asarray(probability_temporal, dtype=np.float64)
    pc = np.asarray(probability_cwt, dtype=np.float64)
    pred_t, pred_c = pt >= 0.5, pc >= 0.5
    parts = [
        np.asarray(logit_temporal)[:, None], np.asarray(logit_cwt)[:, None],
        pt[:, None], pc[:, None], np.abs(pt - 0.5)[:, None], np.abs(pc - 0.5)[:, None],
        normalized_entropy(pt)[:, None], normalized_entropy(pc)[:, None],
        np.maximum(pt, 1 - pt)[:, None], np.maximum(pc, 1 - pc)[:, None],
        np.abs(pt - pc)[:, None], (pt - pc)[:, None], (pred_t == pred_c).astype(float)[:, None],
        np.asarray(distance_temporal), np.asarray(distance_cwt),
        (distance_temporal[:, 0] - distance_temporal[:, 1])[:, None],
        (distance_cwt[:, 0] - distance_cwt[:, 1])[:, None], np.asarray(signal_summary),
    ]
    return np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=1e6, neginf=-1e6)


def fit_centroid_reference(embedding: np.ndarray, labels: np.ndarray) -> CentroidReference:
    """Fit NonPD/PD centroids using labels from the training partition only."""

    embedding = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if embedding.ndim != 2 or len(embedding) != len(labels):
        raise ValueError("Embedding and labels have incompatible shapes.")
    centroids = []
    for label in (0, 1):
        if not np.any(labels == label):
            raise ValueError(f"Cannot build a centroid for missing class {label}.")
        centroids.append(embedding[labels == label].mean(axis=0))
    return CentroidReference(nonpd=centroids[0], pd=centroids[1])


def distances_to_centroids(embedding: np.ndarray, reference: CentroidReference) -> np.ndarray:
    """Transform any split using a previously fitted centroid reference."""

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError("Embedding must be a 2D array.")
    return np.column_stack([
        np.linalg.norm(embedding - reference.nonpd, axis=1),
        np.linalg.norm(embedding - reference.pd, axis=1),
    ])


def centroid_distances(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Backward-compatible train-only helper; use fit/transform for other splits."""

    return distances_to_centroids(embedding, fit_centroid_reference(embedding, labels))


def adaptive_probability(
    temporal_probability: np.ndarray,
    cwt_probability: np.ndarray,
    temporal_reliability: np.ndarray,
    cwt_reliability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse probabilities and return temporal/CWT sample-wise weights."""

    rt = np.asarray(temporal_reliability, dtype=np.float64)
    rc = np.asarray(cwt_reliability, dtype=np.float64)
    denominator = rt + rc + 1e-8
    weights = np.column_stack([rt / denominator, rc / denominator])
    probability = weights[:, 0] * temporal_probability + weights[:, 1] * cwt_probability
    return probability, weights


def conservative_probability(
    fixed_probability: np.ndarray,
    adaptive_probability_values: np.ndarray,
    temporal_reliability: np.ndarray,
    cwt_reliability: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Fall back to fixed fusion unless reliability disagreement is strong."""

    use_fixed = np.abs(np.asarray(temporal_reliability) - np.asarray(cwt_reliability)) < delta
    return np.where(use_fixed, fixed_probability, adaptive_probability_values)


def oracle_prediction(labels: np.ndarray, temporal_prediction: np.ndarray, cwt_prediction: np.ndarray) -> np.ndarray:
    """Return a non-deployable upper-bound prediction."""

    temporal_correct = np.asarray(temporal_prediction) == labels
    cwt_correct = np.asarray(cwt_prediction) == labels
    return np.where((~temporal_correct) & cwt_correct, cwt_prediction, temporal_prediction).astype(int)
