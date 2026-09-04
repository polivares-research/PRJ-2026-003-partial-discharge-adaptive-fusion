"""Two-pass representation preparation for large column-oriented datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .representations import Standardizer


ArrayBatchFactory = Callable[[], Iterator[np.ndarray]]


def fit_stream_standardizer(
    batches: ArrayBatchFactory,
    *,
    sample_shape: tuple[int, ...],
) -> Standardizer:
    """Fit feature-wise mean/std in one pass without retaining raw signals."""

    total = np.zeros(sample_shape, dtype=np.float64)
    total_squared = np.zeros(sample_shape, dtype=np.float64)
    count = 0
    for values in batches():
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != len(sample_shape) + 1 or values.shape[1:] != sample_shape:
            raise ValueError(f"Unexpected streaming batch shape: {values.shape}; expected [N, {sample_shape}]")
        total += values.sum(axis=0, dtype=np.float64)
        total_squared += np.square(values, dtype=np.float64).sum(axis=0)
        count += len(values)
    if count == 0:
        raise ValueError("Cannot fit a standardizer on an empty stream.")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return Standardizer(mean=mean[None, ...].astype(np.float32), std=std[None, ...].astype(np.float32))


def write_stream_cache(
    batches: ArrayBatchFactory,
    *,
    n_samples: int,
    sample_shape: tuple[int, ...],
    transform: Callable[[np.ndarray], np.ndarray],
    destination: str | Path,
    dtype: str = "float32",
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a transformed stream into a NumPy ``.npy`` memmap."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_shape = (int(n_samples), *sample_shape)
    mmap = np.lib.format.open_memmap(destination, mode="w+", dtype=dtype, shape=output_shape)
    offset = 0
    for values in batches():
        transformed = np.asarray(transform(np.asarray(values)), dtype=dtype)
        if transformed.ndim != len(sample_shape) + 1 or transformed.shape[1:] != sample_shape:
            raise ValueError(f"Unexpected transformed batch shape: {transformed.shape}")
        stop = offset + len(transformed)
        if stop > n_samples:
            raise ValueError("Streaming cache received more samples than declared.")
        mmap[offset:stop] = transformed
        offset = stop
    mmap.flush()
    del mmap
    if offset != n_samples:
        raise ValueError(f"Streaming cache coverage mismatch: wrote {offset}, expected {n_samples}.")
    record = {"shape": list(output_shape), "dtype": str(np.dtype(dtype)), **(metadata or {})}
    destination.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return destination
