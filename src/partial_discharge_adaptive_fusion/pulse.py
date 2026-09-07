"""Public V5 pulse API with centered and circular cycle alignment."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import pulse_impl as _impl
from .pulse_impl import (  # noqa: F401
    PULSE_PREPROCESSING_VERSION,
    CycleReference,
    PeakDetection,
    PulseBag,
    PulsePolicy,
    flatten_signal,
    pulse_coverage_report,
)


def detect_cycle_reference(signal: np.ndarray, policy: PulsePolicy) -> CycleReference:
    """Find a circular rising/falling pair around the centered smooth signal."""

    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1 or len(values) != policy.signal_length:
        raise ValueError(f"Expected one signal of length {policy.signal_length}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values")
    smooth = uniform_filter1d(
        values.astype(np.float64), size=policy.moving_average_samples, mode="nearest",
    )
    centered = smooth - np.median(smooth)
    rising = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0)) + 1
    falling = np.flatnonzero((centered[:-1] >= 0) & (centered[1:] < 0)) + 1
    if len(rising) == 0 or len(falling) == 0:
        raise ValueError("Unable to identify both centered rising and falling zero crossings")
    expected = policy.half_length
    n_samples = len(values)
    pairs = []
    for origin in rising:
        for opposite in falling:
            separation = int(opposite - origin) if opposite > origin else int(opposite + n_samples - origin)
            pairs.append((int(origin), int(opposite), separation))
    origin, opposite, separation = min(pairs, key=lambda pair: abs(pair[2] - expected))
    if abs(separation - expected) > expected // 4:
        raise ValueError("Centered zero-crossing separation is incompatible with one VSB half-cycle")
    return CycleReference(
        origin=origin, opposite=opposite, half_period=expected,
        zero_crossings=_impl._crossings(centered), origin_polarity="rising",
    )


def detect_pulses(signal: np.ndarray, policy: PulsePolicy) -> PeakDetection:
    """Detect ranked peaks after centered circular phase alignment."""

    reference = detect_cycle_reference(signal, policy)
    aligned = np.roll(np.asarray(signal, dtype=np.float32), -reference.origin)
    flattened = flatten_signal(aligned, policy)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(np.abs(flattened), distance=policy.minimum_peak_distance)
    half = policy.half_length
    peak_sets, amplitude_sets, counts = [], [], []
    for start, stop in ((0, half), (half, policy.signal_length)):
        valid = peaks[(peaks >= start) & (peaks < stop)]
        amplitudes = np.abs(flattened[valid]).astype(np.float64)
        order = np.argsort(-amplitudes, kind="mergesort")
        ranked_peaks = valid[order].astype(np.int64, copy=False)
        ranked_amplitudes = amplitudes[order].astype(np.float32, copy=False)
        count = _impl._knee_count(ranked_amplitudes, policy)
        peak_sets.append(ranked_peaks[:count])
        amplitude_sets.append(ranked_amplitudes[:count])
        counts.append(count)
    return PeakDetection(
        aligned_signal=aligned, flattened_signal=flattened, reference=reference,
        peak_indices=(peak_sets[0], peak_sets[1]),
        peak_amplitudes=(amplitude_sets[0], amplitude_sets[1]),
        selected_counts=(counts[0], counts[1]),
    )


def prepare_pulse_bag(signal: np.ndarray, policy: PulsePolicy) -> PulseBag:
    """Create temporal/CWT bags using the public centered cycle detector."""

    detected = detect_pulses(signal, policy)
    temporal = np.zeros((2, policy.n_pulses_per_half, policy.temporal_length), dtype=np.float32)
    cwt_segments = np.zeros((2, policy.n_pulses_per_half, policy.cwt_length), dtype=np.float32)
    valid_mask = np.zeros((2, policy.n_pulses_per_half), dtype=bool)
    peak_indices = np.full((2, policy.n_pulses_per_half), -1, dtype=np.int64)
    for half_index, peaks in enumerate(detected.peak_indices):
        for pulse_index, center in enumerate(peaks[:policy.n_pulses_per_half]):
            temporal[half_index, pulse_index] = _impl._extract_segment(detected.aligned_signal, int(center), policy.temporal_length)
            cwt_segments[half_index, pulse_index] = _impl._extract_segment(detected.aligned_signal, int(center), policy.cwt_length)
            peak_indices[half_index, pulse_index] = int(center)
            valid_mask[half_index, pulse_index] = True
    return PulseBag(
        temporal=temporal, cwt_segments=cwt_segments, valid_mask=valid_mask,
        peak_indices=peak_indices, reference=detected.reference,
    )
