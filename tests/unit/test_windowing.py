import numpy as np
import pytest

from partial_discharge_adaptive_fusion.windowing import (
    WindowSpec, aggregate_window_probabilities, expand_parent_metadata,
    validate_parent_assignments, window_batch,
)


def test_end_aligned_windows_cover_full_signal_without_padding():
    spec = WindowSpec(window_length=4, stride=4, aggregation="max")
    starts = spec.starts(10)
    assert starts.tolist() == [0, 4, 6]
    values = np.arange(10, dtype=np.float32)[None, :]
    windows = window_batch(values, spec)
    assert windows.shape == (1, 3, 4)
    assert windows[0, 0].tolist() == [0, 1, 2, 3]
    assert windows[0, -1].tolist() == [6, 7, 8, 9]


def test_parent_and_group_assignments_cannot_cross_splits():
    validate_parent_assignments(
        np.array(["a", "b"]), np.array(["measurement-a", "measurement-b"]),
        np.array(["train", "validation"]), n_windows=3,
    )
    with pytest.raises(ValueError, match="Group"):
        validate_parent_assignments(
            np.array(["a", "b"]), np.array(["same", "same"]),
            np.array(["train", "validation"]), n_windows=2,
        )
    expanded = expand_parent_metadata(
        np.array(["a", "b"]), np.array(["measurement-a", "measurement-b"]),
        np.array(["train", "validation"]), n_windows=3,
    )
    assert expanded["parent_id"].tolist() == ["a", "a", "a", "b", "b", "b"]
    assert expanded["window_index"].tolist() == [0, 1, 2, 0, 1, 2]


def test_aggregation_returns_one_probability_per_parent():
    probabilities = np.array([[0.1, 0.2, 0.9], [0.3, 0.4, 0.5]])
    maximum = aggregate_window_probabilities(probabilities, WindowSpec(4, 4, aggregation="max"))
    top_mean = aggregate_window_probabilities(probabilities, WindowSpec(4, 4, aggregation="top_k_mean", top_k_fraction=0.5))
    assert maximum.shape == top_mean.shape == (2,)
    assert np.allclose(maximum, [0.9, 0.5])
    assert np.allclose(top_mean, [0.55, 0.45])
