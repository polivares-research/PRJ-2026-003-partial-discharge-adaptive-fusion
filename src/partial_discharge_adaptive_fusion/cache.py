"""Deterministic, dataset/config-specific representation caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .representations import cwt_log_power_batch


@dataclass(frozen=True)
class CacheRecord:
    """Location and provenance for one memory-mapped representation."""

    array_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def cache_key(
    dataset_id: str,
    dataset_version: str,
    partition: str,
    representation: str,
    parameters: dict[str, Any],
) -> str:
    """Create a stable short key without embedding a machine-specific path."""

    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "partition": partition,
            "representation": representation,
            "parameters": parameters,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _destination(root: str | Path, key: str, representation: str) -> tuple[Path, Path]:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{representation}-{key}.npy", base / f"{representation}-{key}.json"


def write_array_cache(
    values: np.ndarray,
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    partition: str,
    representation: str,
    parameters: dict[str, Any],
    dtype: str = "float32",
) -> CacheRecord:
    """Persist a bounded NumPy array and its exact construction metadata."""

    values = np.asarray(values, dtype=dtype)
    key = cache_key(dataset_id, dataset_version, partition, representation, parameters)
    array_path, metadata_path = _destination(root, key, representation)
    np.save(array_path, values, allow_pickle=False)
    metadata = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "partition": partition,
        "representation": representation,
        "parameters": parameters,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "cache_key": key,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return CacheRecord(array_path, metadata_path, metadata)


def write_cwt_cache(
    values: np.ndarray,
    *,
    root: str | Path,
    dataset_id: str,
    dataset_version: str,
    partition: str,
    scales: np.ndarray,
    time_bins: int,
    morlet_w0: float = 6.0,
    floor: float = -10.0,
    batch_size: int = 128,
    dtype: str = "float16",
) -> CacheRecord:
    """Build a CWT cache in bounded batches to avoid a large temporary tensor."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [samples, time], got {values.shape}")
    scales = np.asarray(scales, dtype=np.float64)
    parameters = {
        "scales": scales.tolist(), "time_bins": int(time_bins),
        "morlet_w0": float(morlet_w0), "floor": float(floor),
    }
    key = cache_key(dataset_id, dataset_version, partition, "cwt_log_power", parameters)
    array_path, metadata_path = _destination(root, key, "cwt_log_power")
    shape = (len(values), len(scales), int(time_bins))
    mmap = np.lib.format.open_memmap(array_path, mode="w+", dtype=dtype, shape=shape)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        mmap[start:stop] = cwt_log_power_batch(
            values[start:stop], scales=scales, time_bins=time_bins,
            morlet_w0=morlet_w0, floor=floor,
        )
    mmap.flush()
    del mmap
    metadata = {
        "dataset_id": dataset_id, "dataset_version": dataset_version,
        "partition": partition, "representation": "cwt_log_power",
        "parameters": parameters, "shape": list(shape), "dtype": str(np.dtype(dtype)),
        "cache_key": key,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return CacheRecord(array_path, metadata_path, metadata)


def load_cache(record: CacheRecord) -> np.ndarray:
    """Load a cache as a read-only memory map and verify its recorded shape."""

    values = np.load(record.array_path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != record.metadata["shape"]:
        raise ValueError(f"Cache shape mismatch: {values.shape} != {record.metadata['shape']}")
    return values


def load_cache_verified(
    record: CacheRecord,
    *,
    expected_parameters: dict[str, Any],
) -> np.ndarray:
    """Load a cache only when its complete preprocessing parameters match."""

    observed = record.metadata.get("parameters")
    if observed != expected_parameters:
        raise ValueError(
            "Incompatible representation cache: preprocessing parameters differ from the requested protocol."
        )
    return load_cache(record)
