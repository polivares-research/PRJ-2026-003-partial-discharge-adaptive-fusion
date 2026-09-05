import numpy as np

from partial_discharge_adaptive_fusion.evaluation import binary_metrics, paired_bootstrap_delta
from partial_discharge_adaptive_fusion.fusion import (
    adaptive_probability,
    conservative_probability,
    select_fixed_weight,
    select_threshold,
)


def test_adaptive_weights_are_normalized_and_sample_specific():
    probability, weights = adaptive_probability(
        np.array([0.9, 0.2]), np.array([0.1, 0.8]),
        np.array([0.9, 0.2]), np.array([0.1, 0.8]),
    )
    assert probability.shape == (2,)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert not np.allclose(weights[0], weights[1])


def test_conservative_fusion_falls_back_when_reliability_is_close():
    fixed = np.array([0.4, 0.6])
    adaptive = np.array([0.2, 0.8])
    result = conservative_probability(fixed, adaptive, np.array([0.50, 0.9]), np.array([0.52, 0.1]), 0.05)
    assert np.allclose(result, [0.4, 0.8])


def test_fixed_weight_and_threshold_are_development_only_operations():
    labels = np.array([0, 0, 1, 1])
    temporal = np.array([0.1, 0.2, 0.8, 0.9])
    cwt = np.array([0.2, 0.3, 0.7, 0.8])
    selection = select_fixed_weight(labels, temporal, cwt)
    assert 0.0 <= selection["weight_temporal"] <= 1.0
    assert 0.05 <= selection["threshold"] <= 0.95
    assert 0.05 <= select_threshold(labels, temporal) <= 0.95
    metrics = binary_metrics(labels, temporal, 0.5)
    assert metrics["mcc"] == 1.0


def test_signal_level_reliability_features_accept_optional_window_statistics():
    from partial_discharge_adaptive_fusion.fusion import signal_level_reliability_features

    probabilities = np.array([0.2, 0.8])
    summaries = {"max_probability": np.array([0.3, 0.9]), "top1_top2_gap": np.array([0.1, 0.2])}
    features = signal_level_reliability_features(
        np.array([-1.0, 1.0]), np.array([-0.5, 0.5]), probabilities, probabilities,
        window_summaries_temporal=summaries,
    )
    assert features.shape == (2, 15)
    assert np.isfinite(features).all()


def test_paired_bootstrap_delta_is_finite_for_small_inputs():
    labels = np.array([0, 0, 1, 1, 0, 1])
    prediction_a = np.array([0, 1, 1, 1, 0, 0])
    prediction_b = np.array([0, 0, 1, 0, 0, 1])
    result = paired_bootstrap_delta(labels, prediction_a, prediction_b, iterations=200, seed=42)
    assert result["iterations"] == 200
    assert len(result["ci_95"]) == 2
