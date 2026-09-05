"""Deterministic windowing and signal-level multi-instance aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


WINDOWING_VERSION = "multi-instance-windowing-v1"
SUMMARY_NAMES = (
    "max_probability",
    "mean_probability",
    "top_k_mean_probability",
    "std_probability",
    "fraction_high_confidence",
    "top1_top2_gap",
)


@dataclass(frozen=True)
class WindowSpec:
    """A reproducible, end-aligned window definition for one parent signal."""

    window_length: int
    stride: int
    aggregation: str = "top_k_mean"
    top_k_fraction: float = 0.10
    padding: str = "none"
    version: str = WINDOWING_VERSION

    def __post_init__(self) -> None:
        if self.window_length < 2:
            raise ValueError("window_length must be at least 2")
        if self.stride < 1 or self.stride > self.window_length:
            raise ValueError("stride must be in [1, window_length]")
        if self.aggregation not in {"max", "top_k_mean"}:
            raise ValueError("aggregation must be 'max' or 'top_k_mean'")
        if not 0.0 < self.top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if self.padding != "none":
            raise ValueError("Only padding='none' is supported")

    def starts(self, signal_length: int) -> np.ndarray:
        """Return starts that cover the full signal without artificial padding."""

        if signal_length < self.window_length:
            raise ValueError(
                f"Signal length {signal_length} is shorter than window_length {self.window_length}."
            )
        last = signal_length - self.window_length
        starts = list(range(0, last + 1, self.stride))
        if not starts or starts[-1] != last:
            starts.append(last)
        result = np.asarray(starts, dtype=np.int64)
        if result[0] != 0 or result[-1] + self.window_length != signal_length:
            raise RuntimeError("Window construction failed to cover both signal boundaries.")
        if np.any(result[1:] > result[:-1] + self.window_length):
            raise RuntimeError("Window construction introduced an uncovered gap.")
        return result

    def n_windows(self, signal_length: int) -> int:
        return int(len(self.starts(signal_length)))

    def top_k(self, n_windows: int) -> int:
        if n_windows < 1:
            raise ValueError("n_windows must be positive")
        if self.aggregation == "max":
            return 1
        return max(1, int(np.ceil(self.top_k_fraction * n_windows)))

    def as_dict(self, signal_length: int | None = None, sampling_frequency_hz: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "windowing_version": self.version,
            "window_length": int(self.window_length),
            "stride": int(self.stride),
            "aggregation": self.aggregation,
            "top_k_fraction": float(self.top_k_fraction),
            "padding": self.padding,
        }
        if signal_length is not None:
            payload["signal_length"] = int(signal_length)
            payload["n_windows"] = self.n_windows(signal_length)
            payload["top_k"] = self.top_k(payload["n_windows"])
        if sampling_frequency_hz is not None:
            payload["window_duration_seconds"] = self.window_length / float(sampling_frequency_hz)
            payload["window_duration_microseconds"] = 1e6 * payload["window_duration_seconds"]
        return payload


def window_batch(values: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Convert ``[N, T]`` parent signals to ``[N, K, window_length]``."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected [samples, time], got {values.shape}")
    starts = spec.starts(values.shape[1])
    return np.stack([values[:, start : start + spec.window_length] for start in starts], axis=1)


def aggregate_window_probabilities(probabilities: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Aggregate window probabilities into one probability per parent signal."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected [signals, windows], got {probabilities.shape}")
    if probabilities.shape[1] < 1:
        raise ValueError("At least one window is required")
    if spec.aggregation == "max":
        return np.max(probabilities, axis=1)
    k = spec.top_k(probabilities.shape[1])
    return np.mean(np.partition(probabilities, -k, axis=1)[:, -k:], axis=1)


def window_probability_summary(probabilities: np.ndarray, spec: WindowSpec) -> dict[str, np.ndarray]:
    """Return small, label-free window summaries for reliability analysis."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError(f"Expected [signals, windows], got {probabilities.shape}")
    k = spec.top_k(probabilities.shape[1])
    top = np.partition(probabilities, -k, axis=1)[:, -k:]
    ordered = np.sort(probabilities, axis=1)
    top1 = ordered[:, -1]
    top2 = ordered[:, -2] if probabilities.shape[1] > 1 else top1
    return {
        "max_probability": np.max(probabilities, axis=1),
        "mean_probability": np.mean(probabilities, axis=1),
        "top_k_mean_probability": np.mean(top, axis=1),
        "std_probability": np.std(probabilities, axis=1),
        "fraction_high_confidence": np.mean(probabilities >= 0.5, axis=1),
        "top1_top2_gap": top1 - top2,
    }


def validate_parent_assignments(
    parent_ids: np.ndarray,
    group_ids: np.ndarray,
    splits: np.ndarray,
    *,
    n_windows: int,
) -> None:
    """Validate that window expansion has not changed parent split semantics."""

    parent_ids = np.asarray(parent_ids).astype(str)
    group_ids = np.asarray(group_ids).astype(str)
    splits = np.asarray(splits).astype(str)
    if not (len(parent_ids) == len(group_ids) == len(splits)):
        raise ValueError("Parent IDs, groups and splits must align")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    expanded_parent = np.repeat(parent_ids, n_windows)
    expanded_group = np.repeat(group_ids, n_windows)
    expanded_split = np.repeat(splits, n_windows)
    for parent in np.unique(expanded_parent):
        mask = expanded_parent == parent
        if len(np.unique(expanded_split[mask])) != 1:
            raise ValueError(f"Parent {parent} has windows in multiple splits")
    for group in np.unique(expanded_group):
        mask = expanded_group == group
        if len(np.unique(expanded_split[mask])) != 1:
            raise ValueError(f"Group {group} has windows in multiple splits")


def expand_parent_metadata(
    parent_ids: np.ndarray,
    group_ids: np.ndarray,
    splits: np.ndarray,
    *,
    n_windows: int,
) -> dict[str, np.ndarray]:
    """Expand IDs for auditing while retaining a parent-to-window mapping."""

    validate_parent_assignments(parent_ids, group_ids, splits, n_windows=n_windows)
    return {
        "parent_id": np.repeat(np.asarray(parent_ids).astype(str), n_windows),
        "group_id": np.repeat(np.asarray(group_ids).astype(str), n_windows),
        "split": np.repeat(np.asarray(splits).astype(str), n_windows),
        "window_index": np.tile(np.arange(n_windows, dtype=np.int64), len(parent_ids)),
    }
