"""Temporal and CWT representations with explicit, fit-on-train scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


def fit_standardizer(values: np.ndarray, axis: int | tuple[int, ...] = 0) -> Standardizer:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=axis, keepdims=True).astype(np.float32)
    std = values.std(axis=axis, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Standardizer(mean=mean, std=std)


def cwt_log_power_batch(
    values: np.ndarray,
    *,
    scales: np.ndarray,
    time_bins: int,
    morlet_w0: float = 6.0,
    floor: float = -10.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the PoC4 Morlet CWT approximation in bounded signal batches."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [samples, time], got {values.shape}")
    n_samples, n_time = values.shape
    indices = np.linspace(0, n_time - 1, time_bins).round().astype(int)
    signal_fft = np.fft.fft(values, axis=1)
    centered = np.arange(n_time, dtype=np.float64) - n_time / 2.0
    result = np.empty((n_samples, len(scales), time_bins), dtype=np.float32)
    for scale_index, scale in enumerate(np.asarray(scales, dtype=np.float64)):
        wavelet = np.exp(-0.5 * (centered / scale) ** 2) * np.exp(1j * morlet_w0 * centered / scale) / np.sqrt(scale)
        kernel = np.conj(np.fft.fft(np.fft.ifftshift(wavelet)))
        coefficients = np.fft.ifft(signal_fft * kernel[None, :], axis=1)
        result[:, scale_index, :] = np.maximum(
            np.log(np.abs(coefficients[:, indices]) ** 2 + eps), floor
        )
    return result


def temporal_input(values: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    """Prepare a 1D-CNN batch as [N, 1, T]."""

    transformed = standardizer.transform(values)
    if transformed.ndim != 2:
        raise ValueError("Temporal values must be [samples, time].")
    return transformed[:, None, :]


def cwt_input(values: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    """Prepare a 2D-CNN batch as [N, 1, scales, time]."""

    transformed = standardizer.transform(values)
    if transformed.ndim != 3:
        raise ValueError("CWT values must be [samples, scales, time].")
    return transformed[:, None, :, :]


def resample_signals(values: np.ndarray, output_length: int) -> np.ndarray:
    """Apply explicit Fourier resampling; only legal after a frozen policy exists."""

    from scipy.signal import resample

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or output_length < 2:
        raise ValueError("Expected [samples, time] and output_length >= 2.")
    return resample(values, output_length, axis=1).astype(np.float32)
