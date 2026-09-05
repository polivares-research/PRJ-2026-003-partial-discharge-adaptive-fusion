"""Representation-aware window preparation for signal-level experiments.

The functions in this module deliberately keep the parent signal as the
statistical unit.  Windows are an implementation detail of the representation
and are never assigned independent labels or split assignments.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from .representations import Standardizer, cwt_log_power_batch
from .streaming import fit_stream_standardizer, write_stream_cache
from .windowing import WindowSpec, window_batch


PREPROCESSING_VERSION = "representation-aware-mi-v1"
VSB_SAMPLING_FREQUENCY_HZ = 40_000_000.0
DEFAULT_SCALES = np.linspace(1.5, 64.0, 32, dtype=np.float64)
DEFAULT_CWT_TIME_BINS = 120
DEFAULT_MORLET_W0 = 6.0


def _validate_signal_batch(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected raw signals [signals, time], got {values.shape}.")
    if values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Raw signal batches must be finite and contain at least two samples.")
    return values


def iter_window_batches(
    batches: Callable[[], Iterator[np.ndarray]],
    spec: WindowSpec,
) -> Iterator[np.ndarray]:
    """Convert a repeatable raw-signal stream to ``[signals, windows, time]``."""

    for values in batches():
        yield window_batch(_validate_signal_batch(values), spec)


def fit_window_standardizer(
    batches: Callable[[], Iterator[np.ndarray]],
    spec: WindowSpec,
    *,
    representation: str,
    scales: np.ndarray = DEFAULT_SCALES,
    time_bins: int = DEFAULT_CWT_TIME_BINS,
    morlet_w0: float = DEFAULT_MORLET_W0,
    floor: float = -10.0,
) -> Standardizer:
    """Fit normalization on training windows only, streaming over parents."""

    if representation == "temporal":
        def transformed() -> Iterator[np.ndarray]:
            for values in iter_window_batches(batches, spec):
                yield values.reshape(-1, spec.window_length)

        return fit_stream_standardizer(transformed, sample_shape=(spec.window_length,))
    if representation == "cwt":
        scales = np.asarray(scales, dtype=np.float64)

        def transformed() -> Iterator[np.ndarray]:
            for values in iter_window_batches(batches, spec):
                flat = values.reshape(-1, spec.window_length)
                yield cwt_log_power_batch(
                    flat, scales=scales, time_bins=time_bins,
                    morlet_w0=morlet_w0, floor=floor,
                )

        return fit_stream_standardizer(transformed, sample_shape=(len(scales), time_bins))
    raise ValueError("representation must be 'temporal' or 'cwt'.")


def temporal_bag_batch(values: np.ndarray, spec: WindowSpec, standardizer: Standardizer) -> np.ndarray:
    """Create normalized temporal bags with shape ``[B, K, 1, W]``."""

    windows = window_batch(_validate_signal_batch(values), spec)
    flat = windows.reshape(-1, spec.window_length)
    transformed = standardizer.transform(flat).reshape(
        len(windows), spec.n_windows(values.shape[1]), 1, spec.window_length,
    )
    return transformed.astype(np.float32, copy=False)


def cwt_bag_batch(
    values: np.ndarray,
    spec: WindowSpec,
    standardizer: Standardizer,
    *,
    scales: np.ndarray = DEFAULT_SCALES,
    time_bins: int = DEFAULT_CWT_TIME_BINS,
    morlet_w0: float = DEFAULT_MORLET_W0,
    floor: float = -10.0,
) -> np.ndarray:
    """Create normalized per-window CWT bags with shape ``[B, K, 1, S, T]``."""

    windows = window_batch(_validate_signal_batch(values), spec)
    flat = windows.reshape(-1, spec.window_length)
    cwt = cwt_log_power_batch(
        flat, scales=np.asarray(scales, dtype=np.float64), time_bins=time_bins,
        morlet_w0=morlet_w0, floor=floor,
    )
    transformed = standardizer.transform(cwt).reshape(
        len(windows), spec.n_windows(values.shape[1]), 1, len(scales), time_bins,
    )
    return transformed.astype(np.float32, copy=False)


def windowed_cache_parameters(
    *,
    dataset_id: str,
    dataset_version: str,
    partition: str,
    spec: WindowSpec,
    representation: str,
    standardizer: Standardizer,
    scales: np.ndarray = DEFAULT_SCALES,
    time_bins: int = DEFAULT_CWT_TIME_BINS,
    morlet_w0: float = DEFAULT_MORLET_W0,
    floor: float = -10.0,
    sampling_frequency_hz: float | None = None,
    signal_length: int | None = None,
) -> dict[str, object]:
    """Return complete cache provenance, including preprocessing version."""

    parameters: dict[str, object] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "partition": partition,
        "window": spec.as_dict(signal_length=signal_length, sampling_frequency_hz=sampling_frequency_hz),
        "representation": representation,
        "normalization": "z_score_fit_on_train_only",
        "standardizer_mean_shape": list(standardizer.mean.shape),
        "standardizer_std_shape": list(standardizer.std.shape),
    }
    if representation == "cwt":
        scales = np.asarray(scales, dtype=np.float64)
        parameters.update({
            "wavelet": "complex_morlet",
            "morlet_w0": float(morlet_w0),
            "scales": scales.tolist(),
            "time_bins": int(time_bins),
            "transform": "log_power",
            "clip_floor": float(floor),
        })
        if sampling_frequency_hz is not None:
            frequencies = float(sampling_frequency_hz) * morlet_w0 / (2.0 * np.pi * scales)
            parameters["sampling_frequency_hz"] = float(sampling_frequency_hz)
            parameters["pseudo_frequency_hz"] = frequencies.tolist()
    return parameters


def write_windowed_cache(
    batches: Callable[[], Iterator[np.ndarray]],
    *,
    n_samples: int,
    spec: WindowSpec,
    representation: str,
    standardizer: Standardizer,
    destination: str,
    dataset_id: str,
    dataset_version: str,
    partition: str,
    scales: np.ndarray = DEFAULT_SCALES,
    time_bins: int = DEFAULT_CWT_TIME_BINS,
    morlet_w0: float = DEFAULT_MORLET_W0,
    floor: float = -10.0,
    sampling_frequency_hz: float | None = None,
    dtype: str = "float16",
) -> str:
    """Write one deterministic bag cache while reading raw signals sequentially."""

    signal_length = _infer_signal_length(batches)
    if representation == "temporal":
        sample_shape = (spec.n_windows(signal_length), 1, spec.window_length)
        transform = lambda values: temporal_bag_batch(values, spec, standardizer)
    elif representation == "cwt":
        sample_shape = (spec.n_windows(signal_length), 1, len(scales), time_bins)
        transform = lambda values: cwt_bag_batch(
            values, spec, standardizer, scales=scales, time_bins=time_bins,
            morlet_w0=morlet_w0, floor=floor,
        )
    else:
        raise ValueError("representation must be 'temporal' or 'cwt'.")
    metadata = windowed_cache_parameters(
        dataset_id=dataset_id, dataset_version=dataset_version, partition=partition,
        spec=spec, representation=representation, standardizer=standardizer,
        scales=scales, time_bins=time_bins, morlet_w0=morlet_w0, floor=floor,
        sampling_frequency_hz=sampling_frequency_hz, signal_length=signal_length,
    )
    return str(write_stream_cache(
        batches, n_samples=n_samples, sample_shape=sample_shape, transform=transform,
        destination=destination, dtype=dtype, metadata=metadata,
    ))


def _infer_signal_length(batches: Callable[[], Iterator[np.ndarray]]) -> int:
    """Read one bounded batch to validate the common signal length."""

    iterator = iter(batches())
    try:
        first = _validate_signal_batch(next(iterator))
    except StopIteration as exc:
        raise ValueError("Cannot infer window count from an empty signal stream.") from exc
    return int(first.shape[1])
