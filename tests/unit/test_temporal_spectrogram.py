import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from partial_discharge_adaptive_fusion.modeling.models import SmallSpectrogram2DCNN
from partial_discharge_adaptive_fusion.spectrogram import (
    IndexedArrayView, STFTSpec,
    compute_stft_log_power,
    fit_spectrogram_standardizer,
    require_valid_cache,
    write_atomic_array_cache,
)
from partial_discharge_adaptive_fusion.temporal_spectrogram import (
    aggregate_event_predictions,
    classify_cross_dataset_verdict,
    prediction_overlap_oracle, probability_summary, select_fixed_weight_oof, threshold_curve,
)


def test_registered_stft_shapes_are_deterministic_and_finite():
    values = np.arange(800, dtype=np.float32).reshape(2, 400)
    spec = STFTSpec(n_fft=128, win_length=64, hop_length=16)
    first = compute_stft_log_power(values, spec)
    second = compute_stft_log_power(values, spec)
    assert first.shape == (2, 1, 65, 22)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_vsb_stft_geometry():
    values = np.zeros((3, 512), dtype=np.float32)
    output = compute_stft_log_power(values, STFTSpec(128, 128, 32))
    assert output.shape == (3, 1, 65, 13)


def test_train_only_spectrogram_standardizer_and_cache(tmp_path):
    values = np.arange(4 * 2 * 3 * 1 * 4 * 5, dtype=np.float32).reshape(4, 2, 3, 1, 4, 5)
    mask = np.ones((4, 2, 3), dtype=bool)
    mask[0, 0, 1] = False
    standardizer = fit_spectrogram_standardizer(values, np.array([0, 1]), mask)
    transformed = standardizer.transform(values, mask)
    assert standardizer.fit_count == 11
    assert np.allclose(transformed[~mask], 0.0)
    path = tmp_path / "spectrogram.npy"
    metadata = write_atomic_array_cache(path, values.astype(np.float16), {"source": "test", "version": 1})
    loaded = require_valid_cache(path, metadata["fingerprint"])
    assert loaded.shape == values.shape
    with pytest.raises(ValueError, match="fingerprint"):
        require_valid_cache(path, "wrong")


def test_lazy_standardized_view_matches_train_only_transform():
    values = np.arange(4 * 1 * 3 * 1 * 4 * 5, dtype=np.float32).reshape(4, 1, 3, 1, 4, 5)
    mask = np.ones((4, 1, 3), dtype=bool)
    standardizer = fit_spectrogram_standardizer(values, np.array([0, 1]), mask)
    view = standardizer.view(values, valid_mask=mask)
    np.testing.assert_allclose(view[2], standardizer.transform(values[2:3], mask[2:3])[0])
    assert IndexedArrayView(values, np.array([3, 1])).shape == (2, 1, 3, 1, 4, 5)


def test_threshold_curve_and_fixed_weights_are_oof_only_interfaces():
    labels = np.array([0, 0, 1, 1])
    temporal = np.array([0.1, 0.2, 0.8, 0.9])
    spectrogram = np.array([0.2, 0.3, 0.7, 0.8])
    curve = threshold_curve(labels, temporal, thresholds=(0.2, 0.5))
    assert [row["threshold"] for row in curve] == [0.2, 0.5]
    assert probability_summary(temporal)["count"] == 4
    rows, best = select_fixed_weight_oof(labels, temporal, spectrogram, labels, temporal, spectrogram, (0.0, 0.5, 1.0))
    assert len(rows) == 3
    assert 0.0 <= best["temporal_weight"] <= 1.0


def test_masked_event_aggregation_returns_one_parent_probability():
    probabilities = np.linspace(0.0, 1.0, 2 * 2 * 86, dtype=float).reshape(2, 2, 86)
    mask = np.ones((2, 2, 86), dtype=bool)
    mask[0, :, 80:] = False
    result = aggregate_event_predictions(probabilities, mask)
    assert result.shape == (2,)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()


def test_overlap_and_verdict_rules():
    labels = np.array([0, 1, 0, 1])
    temporal = np.array([0, 0, 1, 1])
    spectrogram = np.array([0, 1, 0, 0])
    result = prediction_overlap_oracle(labels, temporal, spectrogram, np.array(["a", "a", "b", "b"]))
    assert result["oracle_headroom"] >= 0.0
    assert result["group_count"] == 2
    assert classify_cross_dataset_verdict({"integrity_ok": False, "regression_ok": True}) == "INCONCLUSIVE"
    assert classify_cross_dataset_verdict({"integrity_ok": True, "regression_ok": True, "vsb_spectrogram_strong": True, "matlab_spectrogram_strong": False, "complementarity_supported": True}) == "SPECTROGRAM SUPPORTED ONLY ON VSB"


def test_spectrogram_model_shape():
    import torch

    model = SmallSpectrogram2DCNN()
    output = model(torch.zeros(2, 1, 65, 22))
    assert output.shape == (2,)
