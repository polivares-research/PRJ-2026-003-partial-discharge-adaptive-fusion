"""Deterministic signal summaries used by the reliability models."""

from __future__ import annotations

import numpy as np


SIGNAL_FEATURE_NAMES = (
    "rms", "peak_abs", "energy", "std", "skewness", "kurtosis",
    "crest_factor", "spectral_centroid", "spectral_entropy", "high_frequency_ratio",
)


def signal_features(values: np.ndarray) -> np.ndarray:
    """Return finite, label-free signal descriptors for a batch."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [samples, time], got {values.shape}")
    centered = values - values.mean(axis=1, keepdims=True)
    std = values.std(axis=1) + 1e-8
    rms = np.sqrt(np.mean(values * values, axis=1))
    peak = np.max(np.abs(values), axis=1)
    energy = np.mean(values * values, axis=1)
    skew = np.mean(centered ** 3, axis=1) / (std ** 3)
    kurt = np.mean(centered ** 4, axis=1) / (std ** 4) - 3.0
    spectrum = np.abs(np.fft.rfft(values, axis=1)) + 1e-12
    power = spectrum ** 2
    frequency = np.arange(power.shape[1], dtype=np.float64)[None, :] / max(power.shape[1] - 1, 1)
    centroid = (power * frequency).sum(axis=1) / power.sum(axis=1)
    spectral_probability = power / power.sum(axis=1, keepdims=True)
    spectral_entropy = -(spectral_probability * np.log(spectral_probability)).sum(axis=1) / np.log(power.shape[1])
    high_frequency_ratio = power[:, int(0.25 * power.shape[1]):].sum(axis=1) / power.sum(axis=1)
    return np.nan_to_num(np.column_stack([
        rms, peak, energy, std, skew, kurt, peak / (rms + 1e-8),
        centroid, spectral_entropy, high_frequency_ratio,
    ]), nan=0.0, posinf=1e6, neginf=-1e6)


def temporal_summary_embedding(values: np.ndarray, blocks: int = 20) -> np.ndarray:
    """Create a fixed low-dimensional temporal embedding without labels."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % blocks:
        raise ValueError("Temporal embedding requires a 2D array divisible by blocks.")
    reshaped = values.reshape(len(values), blocks, values.shape[1] // blocks)
    return np.concatenate([reshaped.mean(axis=2), reshaped.std(axis=2)], axis=1).astype(np.float64)


def cwt_summary_embedding(values: np.ndarray, scale_groups: int = 8, time_groups: int = 8) -> np.ndarray:
    """Pool a CWT tensor into a compact embedding."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [samples, scales, time], got {values.shape}")
    n, scales, times = values.shape
    if scales % scale_groups or times % time_groups:
        raise ValueError("CWT shape must be divisible by the requested pooling groups.")
    pooled = values.reshape(n, scale_groups, scales // scale_groups, time_groups, times // time_groups)
    return pooled.mean(axis=(2, 4)).reshape(n, -1).astype(np.float64)
