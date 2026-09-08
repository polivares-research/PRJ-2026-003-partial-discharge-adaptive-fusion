"""Lazy V5 pulse views and train-only standardization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class PulseRepresentationView:
    """Select a candidate pulse count and CWT scale count without copying a cache."""

    def __init__(
        self,
        values: Any,
        *,
        representation: str,
        n_pulses: int,
        scale_indices: np.ndarray | None = None,
    ) -> None:
        shape = tuple(int(value) for value in getattr(values, "shape", ()))
        if representation == "temporal":
            if len(shape) != 5 or shape[1] != 2 or shape[2] < n_pulses:
                raise ValueError(f"Unexpected temporal cache shape for candidate view: {shape}")
            self.shape = (shape[0], shape[1], n_pulses, shape[3], shape[4])
        elif representation == "cwt":
            if len(shape) != 6 or shape[1] != 2 or shape[2] < n_pulses:
                raise ValueError(f"Unexpected CWT cache shape for candidate view: {shape}")
            indices = np.asarray(scale_indices, dtype=np.int64) if scale_indices is not None else np.arange(shape[4])
            if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= shape[4]):
                raise ValueError("CWT scale indices are invalid")
            self.shape = (shape[0], shape[1], n_pulses, shape[3], len(indices), shape[5])
            self.scale_indices = indices
        else:
            raise ValueError("representation must be temporal or cwt")
        self.values = values
        self.representation = representation
        self.n_pulses = int(n_pulses)

    def __len__(self) -> int:
        return int(self.shape[0])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: Any) -> np.ndarray:
        if np.isscalar(index):
            row = np.asarray(self.values[int(index)])
            if self.representation == "temporal":
                return row[:, : self.n_pulses]
            return row[:, : self.n_pulses, :, self.scale_indices, :]
        indices = np.asarray(index)
        return np.stack([self[int(item)] for item in indices.reshape(-1)], axis=0).reshape(
            tuple(indices.shape) + self.shape[1:]
        )


class PulseMaskView:
    """Select the same parent/pulse subset from a validity-mask cache."""

    def __init__(self, mask: Any, n_pulses: int) -> None:
        shape = tuple(int(value) for value in getattr(mask, "shape", ()))
        if len(shape) != 3 or shape[1] != 2 or shape[2] < n_pulses:
            raise ValueError(f"Unexpected pulse mask shape: {shape}")
        self.mask = mask
        self.shape = (shape[0], shape[1], int(n_pulses))

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, index: Any) -> np.ndarray:
        if np.isscalar(index):
            return np.asarray(self.mask[int(index)], dtype=bool)[:, : self.shape[2]]
        indices = np.asarray(index)
        return np.stack([self[int(item)] for item in indices.reshape(-1)], axis=0).reshape(
            tuple(indices.shape) + self.shape[1:]
        )


class IndexedPulseView:
    """Expose a deterministic parent-index subset without copying its source."""

    def __init__(self, values: Any, indices: np.ndarray) -> None:
        self.values = values
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices),) + tuple(values.shape[1:])

    def __len__(self) -> int:
        return int(self.shape[0])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: Any) -> np.ndarray:
        if np.isscalar(index):
            return np.asarray(self.values[int(self.indices[int(index)])])
        indices = np.asarray(index)
        return np.stack([self[int(item)] for item in indices.reshape(-1)], axis=0).reshape(
            tuple(indices.shape) + self.shape[1:]
        )


@dataclass(frozen=True)
class PulseStandardizer:
    """A scalar standardizer fitted only from valid training pulses."""

    mean: float
    std: float
    n_values: int


def fit_pulse_standardizer(
    values: Any,
    valid_mask: Any,
    indices: np.ndarray,
    *,
    logger=None,
    stage: str = "standardizer",
    chunk_size: int = 16,
) -> PulseStandardizer:
    """Fit mean/std without materializing a full memmap or including bag padding."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot fit a pulse standardizer from an empty training split")
    total = 0
    sum_values = 0.0
    sum_squares = 0.0
    for start in range(0, len(indices), max(1, int(chunk_size))):
        for index in indices[start : start + chunk_size]:
            row = np.asarray(values[int(index)], dtype=np.float64)
            mask = np.asarray(valid_mask[int(index)], dtype=bool)
            expanded = mask.reshape(mask.shape + (1,) * (row.ndim - 2))
            selected = row[np.broadcast_to(expanded, row.shape)]
            total += int(selected.size)
            sum_values += float(selected.sum(dtype=np.float64))
            sum_squares += float((selected * selected).sum(dtype=np.float64))
        if logger is not None:
            logger.info(f"{stage}: fitted over {min(start + chunk_size, len(indices))}/{len(indices)} parents")
    if total == 0:
        raise ValueError("No valid pulse values were available for standardization")
    mean = sum_values / total
    variance = max(sum_squares / total - mean * mean, 1e-12)
    return PulseStandardizer(mean=float(mean), std=float(np.sqrt(variance)), n_values=total)


class StandardizedPulseView:
    """Apply a fitted scalar standardizer lazily and restore invalid entries to zero."""

    def __init__(self, values: Any, valid_mask: Any, standardizer: PulseStandardizer) -> None:
        self.values = values
        self.valid_mask = valid_mask
        self.standardizer = standardizer
        self.shape = tuple(values.shape)

    def __len__(self) -> int:
        return int(self.shape[0])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: Any) -> np.ndarray:
        if not np.isscalar(index):
            indices = np.asarray(index)
            return np.stack([self[int(item)] for item in indices.reshape(-1)], axis=0).reshape(
                tuple(indices.shape) + self.shape[1:]
            )
        row = np.asarray(self.values[int(index)], dtype=np.float32)
        mask = np.asarray(self.valid_mask[int(index)], dtype=bool)
        expanded = mask.reshape(mask.shape + (1,) * (row.ndim - 2))
        transformed = (row - self.standardizer.mean) / self.standardizer.std
        return np.where(expanded, transformed, 0.0).astype(np.float32, copy=False)
