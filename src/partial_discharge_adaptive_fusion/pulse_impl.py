"""Deterministic, signal-level pulse preparation for the V5 VSB protocol.

The detector is intentionally independent from labels and split assignment. It
derives a cycle reference, flattens the slowly varying component, ranks local
peaks, and returns fixed-size bags with a validity mask. Zero filling is used
only for missing bag entries; local pulse segments are never padded with an
invented signal value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, lfilter


PULSE_PREPROCESSING_VERSION = "v5-pulse-detector-v1"


@dataclass(frozen=True)
class PulsePolicy:
    """Frozen parameters for one VSB pulse representation."""

    signal_length: int = 800_000
    sampling_frequency_hz: float = 40_000_000.0
    moving_average_samples: int = 10_000
    alpha: float = 100.0
    beta: float = 1.0
    knee_smoothing_length: int = 9
    minimum_peak_distance: int = 512
    minimum_peaks_per_half: int = 8
    n_pulses_per_half: int = 86
    temporal_length: int = 128
    cwt_length: int = 512
    version: str = PULSE_PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        if self.signal_length < 2 or self.signal_length % 2:
            raise ValueError("signal_length must be a positive even number")
        if self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if self.moving_average_samples < 3:
            raise ValueError("moving_average_samples must be an integer >= 3")
        if self.alpha <= 0 or self.beta <= 0 or self.beta > self.alpha:
            raise ValueError("alpha and beta must satisfy 0 < beta <= alpha")
        if self.knee_smoothing_length < 3 or self.knee_smoothing_length % 2 == 0:
            raise ValueError("knee_smoothing_length must be an odd integer >= 3")
        if self.minimum_peak_distance < 1 or self.minimum_peaks_per_half < 1:
            raise ValueError("peak constraints must be positive")
        if self.n_pulses_per_half < 1 or self.temporal_length < 2 or self.cwt_length < 2:
            raise ValueError("pulse and segment lengths must be positive")

    @property
    def half_length(self) -> int:
        return self.signal_length // 2

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "preprocessing_version": self.version,
            "signal_length": self.signal_length,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "moving_average_samples": self.moving_average_samples,
            "alpha": self.alpha,
            "beta": self.beta,
            "knee_smoothing_length": self.knee_smoothing_length,
            "minimum_peak_distance": self.minimum_peak_distance,
            "minimum_peaks_per_half": self.minimum_peaks_per_half,
            "n_pulses_per_half": self.n_pulses_per_half,
            "temporal_length": self.temporal_length,
            "cwt_length": self.cwt_length,
            "padding": "bag_only_zero_fill",
            "phase_alignment": "circular_shift_to_rising_zero_crossing",
        }


@dataclass(frozen=True)
class CycleReference:
    """Cycle landmarks derived from the moving-average signal."""

    origin: int
    opposite: int
    half_period: int
    zero_crossings: np.ndarray
    origin_polarity: str


@dataclass(frozen=True)
class PeakDetection:
    """Label-free peak audit output for one parent signal."""

    aligned_signal: np.ndarray
    flattened_signal: np.ndarray
    reference: CycleReference
    peak_indices: tuple[np.ndarray, np.ndarray]
    peak_amplitudes: tuple[np.ndarray, np.ndarray]
    selected_counts: tuple[int, int]


@dataclass(frozen=True)
class PulseBag:
    """Paired half-cycle bags for one original signal."""

    temporal: np.ndarray
    cwt_segments: np.ndarray
    valid_mask: np.ndarray
    peak_indices: np.ndarray
    reference: CycleReference


def _crossings(values: np.ndarray) -> np.ndarray:
    rising = (values[:-1] <= 0) & (values[1:] > 0)
    falling = (values[:-1] >= 0) & (values[1:] < 0)
    result = np.flatnonzero(rising | falling) + 1
    return result.astype(np.int64, copy=False)


def detect_cycle_reference(signal: np.ndarray, policy: PulsePolicy) -> CycleReference:
    """Find a stable rising/falling pair near one half-cycle apart."""

    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1 or len(values) != policy.signal_length:
        raise ValueError(f"Expected one signal of length {policy.signal_length}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    smooth = uniform_filter1d(values.astype(np.float64), size=policy.moving_average_samples, mode="nearest")
    rising = np.flatnonzero((smooth[:-1] <= 0) & (smooth[1:] > 0)) + 1
    falling = np.flatnonzero((smooth[:-1] >= 0) & (smooth[1:] < 0)) + 1
    if len(rising) == 0 or len(falling) == 0:
        raise ValueError("Unable to identify both rising and falling zero crossings")
    expected = policy.half_length
    n_samples = len(values)
    pairs = []
    for origin in rising:
        for opposite in falling:
            separation = int(opposite - origin) if opposite > origin else int(opposite + n_samples - origin)
            pairs.append((int(origin), int(opposite), separation))
    origin, opposite, separation = min(pairs, key=lambda pair: abs(pair[2] - expected))
    if abs(separation - expected) > expected // 4:
        raise ValueError("Zero-crossing separation is incompatible with one VSB half-cycle")
    return CycleReference(
        origin=origin, opposite=opposite, half_period=expected,
        zero_crossings=_crossings(smooth), origin_polarity="rising",
    )


def flatten_signal(signal: np.ndarray, policy: PulsePolicy) -> np.ndarray:
    """Apply the paper-inspired alpha/beta first-order high-pass flattening."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) != policy.signal_length:
        raise ValueError("flatten_signal expects one complete VSB signal")
    pole = (policy.alpha - policy.beta) / policy.alpha
    baseline = lfilter([policy.beta / policy.alpha], [1.0, -pole], values)
    return (values - baseline).astype(np.float32)


def _knee_count(amplitudes: np.ndarray, policy: PulsePolicy) -> int:
    if len(amplitudes) == 0:
        return 0
    ordered = np.sort(np.asarray(amplitudes, dtype=np.float64))[::-1]
    if len(ordered) < 3:
        return len(ordered)
    log_values = np.log1p(np.maximum(ordered, 0.0))
    smoothed = uniform_filter1d(log_values, size=min(policy.knee_smoothing_length, len(log_values)), mode="nearest")
    curvature = np.abs(np.gradient(np.gradient(smoothed)))
    knee = int(np.argmax(curvature)) + 2
    return int(min(len(ordered), max(policy.minimum_peaks_per_half, knee)))


def detect_pulses(signal: np.ndarray, policy: PulsePolicy) -> PeakDetection:
    """Return ranked pulse centers for each aligned half-cycle."""

    reference = detect_cycle_reference(signal, policy)
    aligned = np.roll(np.asarray(signal, dtype=np.float32), -reference.origin)
    flattened = flatten_signal(aligned, policy)
    peaks, _ = find_peaks(np.abs(flattened), distance=policy.minimum_peak_distance)
    half = policy.half_length
    peak_sets: list[np.ndarray] = []
    amplitude_sets: list[np.ndarray] = []
    counts: list[int] = []
    for start, stop in ((0, half), (half, policy.signal_length)):
        valid = peaks[(peaks >= start) & (peaks < stop)]
        amplitudes = np.abs(flattened[valid]).astype(np.float64)
        order = np.argsort(-amplitudes, kind="mergesort")
        ranked_peaks = valid[order].astype(np.int64, copy=False)
        ranked_amplitudes = amplitudes[order].astype(np.float32, copy=False)
        count = _knee_count(ranked_amplitudes, policy)
        peak_sets.append(ranked_peaks[:count])
        amplitude_sets.append(ranked_amplitudes[:count])
        counts.append(count)
    return PeakDetection(
        aligned_signal=aligned, flattened_signal=flattened, reference=reference,
        peak_indices=(peak_sets[0], peak_sets[1]),
        peak_amplitudes=(amplitude_sets[0], amplitude_sets[1]),
        selected_counts=(counts[0], counts[1]),
    )


def _extract_segment(signal: np.ndarray, center: int, length: int) -> np.ndarray:
    """Extract a real circularly aligned segment without value padding."""

    half = length // 2
    offsets = np.arange(-half, length - half, dtype=np.int64)
    indices = (int(center) + offsets) % len(signal)
    return np.asarray(signal, dtype=np.float32)[indices]


def prepare_pulse_bag(signal: np.ndarray, policy: PulsePolicy) -> PulseBag:
    """Create temporal/CWT pulse bags and validity masks for one signal."""

    detected = detect_pulses(signal, policy)
    temporal = np.zeros((2, policy.n_pulses_per_half, policy.temporal_length), dtype=np.float32)
    cwt_segments = np.zeros((2, policy.n_pulses_per_half, policy.cwt_length), dtype=np.float32)
    valid_mask = np.zeros((2, policy.n_pulses_per_half), dtype=bool)
    peak_indices = np.full((2, policy.n_pulses_per_half), -1, dtype=np.int64)
    for half_index, peaks in enumerate(detected.peak_indices):
        for pulse_index, center in enumerate(peaks[:policy.n_pulses_per_half]):
            temporal[half_index, pulse_index] = _extract_segment(detected.aligned_signal, int(center), policy.temporal_length)
            cwt_segments[half_index, pulse_index] = _extract_segment(detected.aligned_signal, int(center), policy.cwt_length)
            peak_indices[half_index, pulse_index] = int(center)
            valid_mask[half_index, pulse_index] = True
    return PulseBag(
        temporal=temporal, cwt_segments=cwt_segments, valid_mask=valid_mask,
        peak_indices=peak_indices, reference=detected.reference,
    )


def pulse_coverage_report(valid_masks: np.ndarray, references: list[CycleReference] | None = None) -> dict[str, float | int | bool]:
    """Summarize pulse/cycle coverage before any labels are consulted."""

    masks = np.asarray(valid_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1] != 2:
        raise ValueError(f"Expected [signals, 2, pulses] validity masks, got {masks.shape}")
    per_half = masks.sum(axis=2)
    return {
        "n_signals": int(masks.shape[0]),
        "n_pulses_per_half_capacity": int(masks.shape[2]),
        "signals_with_both_half_cycles": int(np.sum(np.all(per_half > 0, axis=1))),
        "signals_with_minimum_eight_pulses_each_half": int(np.sum(np.all(per_half >= 8, axis=1))),
        "mean_valid_pulses_per_half": float(per_half.mean()),
        "minimum_valid_pulses_per_half": int(per_half.min()) if len(per_half) else 0,
        "coverage_ok": bool(np.all(np.all(per_half > 0, axis=1))),
        "references_available": references is None or len(references) == len(masks),
    }
