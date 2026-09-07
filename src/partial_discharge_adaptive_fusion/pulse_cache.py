"""Versioned, atomic cache primitives for V5 pulse representations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


CACHE_CONTRACT_VERSION = "v5-pulse-cache-v1"


class CacheContractError(RuntimeError):
    """Raised when a cache is absent, partial, or incompatible."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def cache_fingerprint(metadata: dict[str, Any]) -> str:
    """Hash every scientific and runtime parameter that can affect an artifact."""

    return hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheMetadata:
    dataset_id: str
    dataset_version: str
    source_fingerprint: str
    split_manifest_hash: str
    pulse_policy: dict[str, Any]
    transform: dict[str, Any]
    dtype: str
    preprocessing_version: str
    library_versions: dict[str, str | None]
    shape: tuple[int, ...]
    contract_version: str = CACHE_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["fingerprint"] = cache_fingerprint(value)
        return value


def _paths(destination: str | Path) -> tuple[Path, Path, Path]:
    path = Path(destination)
    return path, path.with_suffix(path.suffix + ".json"), path.with_suffix(path.suffix + ".complete")


def write_memmap_cache(
    destination: str | Path,
    values: np.ndarray,
    metadata: CacheMetadata,
) -> dict[str, Any]:
    """Write an array, metadata, and completion marker atomically."""

    destination, metadata_path, marker = _paths(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tuple(values.shape) != tuple(metadata.shape):
        raise ValueError(f"Cache values have shape {values.shape}, metadata has {metadata.shape}")
    array = np.asarray(values)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        mapped = np.memmap(temporary, mode="w+", dtype=array.dtype, shape=array.shape)
        mapped[:] = array
        mapped.flush()
        del mapped
        os.replace(temporary, destination)
        payload = metadata.as_dict()
        metadata_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        marker.write_text(payload["fingerprint"] + "\n", encoding="utf-8")
        return payload
    finally:
        if temporary.exists():
            temporary.unlink()


def open_verified_memmap(
    destination: str | Path,
    expected_metadata: CacheMetadata,
    *,
    mode: str = "r",
) -> tuple[np.memmap, dict[str, Any]]:
    """Open only a complete cache whose fingerprint exactly matches."""

    destination, metadata_path, marker = _paths(destination)
    if not destination.is_file() or not metadata_path.is_file() or not marker.is_file():
        raise CacheContractError(f"Incomplete V5 cache: {destination}")
    observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = expected_metadata.as_dict()
    if observed.get("fingerprint") != expected.get("fingerprint"):
        raise CacheContractError(
            f"Cache fingerprint mismatch for {destination}: "
            f"observed={observed.get('fingerprint')} expected={expected.get('fingerprint')}"
        )
    if marker.read_text(encoding="utf-8").strip() != observed["fingerprint"]:
        raise CacheContractError(f"Cache completion marker mismatch: {destination}")
    dtype = np.dtype(observed["dtype"])
    shape = tuple(int(value) for value in observed["shape"])
    return np.memmap(destination, mode=mode, dtype=dtype, shape=shape), observed


def runtime_library_versions() -> dict[str, str | None]:
    """Return compact versions used in cache provenance."""

    from importlib.metadata import PackageNotFoundError, version

    def package(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "numpy": package("numpy"), "scipy": package("scipy"),
        "pandas": package("pandas"), "pyarrow": package("pyarrow"),
        "python": platform.python_version(),
    }
