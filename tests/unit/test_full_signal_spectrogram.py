"""Fast synthetic contracts for the full-signal global spectrogram path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from partial_discharge_adaptive_fusion.full_signal_spectrogram import (
    FullSignalSTFTSpec,
    compute_full_signal_log_power,
    fit_frequency_standardizer,
    require_global_spectrogram_cache,
    write_global_spectrogram_cache,
)
from partial_discharge_adaptive_fusion.modeling.models import GlobalSpectrogram2DCNN


def _small_spec() -> FullSignalSTFTSpec:
    return FullSignalSTFTSpec(
        sampling_frequency_hz=40.0,
        signal_length=8000,
        n_fft=512,
        win_length=512,
        hop_length=256,
        f_min_hz=0.5,
    )


def test_global_shape_frequency_crop_and_fingerprint() -> None:
    spec = _small_spec()
    assert spec.expected_shape == (1, 250, 30)
    assert spec.crop_start_bin == 7
    assert spec.as_dict()["frequency_spacing_hz"] == pytest.approx(40.0 / 512.0)
    assert spec.cache_metadata(dataset="vsb", dataset_version="test")["representation"]["f_min_hz"] == 0.5


def test_global_stft_is_finite_deterministic_and_has_no_event_axis() -> None:
    spec = _small_spec()
    values = np.zeros((2, spec.signal_length), dtype=np.float32)
    values[0, 100] = 1.0
    values[1, 5000] = 1.0
    first = compute_full_signal_log_power(values, spec)
    second = compute_full_signal_log_power(values, spec)
    assert first.shape == (2, 1, 250, 30)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_global_stft_does_not_depend_on_event_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from partial_discharge_adaptive_fusion import pulse

    monkeypatch.setattr(pulse, "detect_pulses", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("event detector called")))
    spec = _small_spec()
    result = compute_full_signal_log_power(np.zeros((1, spec.signal_length), dtype=np.float32), spec)
    assert result.shape[0] == 1


def test_frequency_standardizer_uses_only_selected_rows() -> None:
    values = np.zeros((3, 1, 2, 4), dtype=np.float32)
    values[0] = 1.0
    values[1] = 3.0
    values[2] = 1000.0
    standardizer = fit_frequency_standardizer(values, np.array([0, 1]))
    assert standardizer.fit_count == 8
    assert np.allclose(standardizer.mean.reshape(-1), [2.0, 2.0])
    assert np.allclose(standardizer.view(values)[2], (values[2] - 2.0) / np.sqrt(1.0))


def test_global_model_returns_one_parent_logit() -> None:
    torch = pytest.importorskip("torch")
    model = GlobalSpectrogram2DCNN().eval()
    output = model(torch.zeros((3, 1, 16, 32)))
    assert tuple(output.shape) == (3,)
    assert torch.isfinite(output).all()


def test_global_cache_is_atomic_and_fingerprint_checked(tmp_path) -> None:
    shape = (2, 1, 4, 5)
    path = tmp_path / "global.npy"
    metadata = {"dataset": "test", "representation": {"n_fft": 8}, "preprocessing_version": "unit"}

    def writer(values: np.ndarray) -> None:
        values[:] = 1.0

    payload = write_global_spectrogram_cache(path, shape=shape, metadata=metadata, writer=writer)
    loaded = require_global_spectrogram_cache(path, payload["fingerprint"])
    assert loaded.shape == shape
    assert loaded.dtype == np.float16
    assert np.all(np.asarray(loaded) == 1.0)
    with pytest.raises(ValueError, match="fingerprint"):
        require_global_spectrogram_cache(path, "wrong")
    assert json.loads((tmp_path / "global.npy.json").read_text())["shape"] == list(shape)
