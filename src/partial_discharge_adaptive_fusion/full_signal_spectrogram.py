"""Global full-signal log-spectrogram representation for the VSB diagnostic.

This module is intentionally independent of :mod:`pulse`.  A VSB parent signal
is converted once to one global STFT and produces one parent-level input.  No
event detector, local segment, circular boundary, MIL mask, or top-k pooling is
part of this representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class FullSignalSTFTSpec:
    """Registered global STFT geometry and physical frequency crop."""

    sampling_frequency_hz: float
    signal_length: int = 800_000
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 256
    epsilon: float = 1e-8
    f_min_hz: float = 500_000.0
    center: bool = False
    onesided: bool = True
    window: str = "hann"
    power: int = 2
    log_base: int = 10
    representation_version: str = "v1-global-full-signal-log10-power"

    def __post_init__(self) -> None:
        if self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if self.signal_length < self.win_length:
            raise ValueError("signal_length must be at least win_length")
        if self.n_fft < 2 or not 2 <= self.win_length <= self.n_fft:
            raise ValueError("n_fft and win_length must satisfy 2 <= win_length <= n_fft")
        if self.hop_length < 1 or self.epsilon <= 0:
            raise ValueError("hop_length and epsilon must be positive")
        if self.window != "hann" or not self.onesided or self.center:
            raise ValueError("The global diagnostic requires Hann, one-sided, center=False STFT")
        if self.power != 2 or self.log_base != 10:
            raise ValueError("The registered representation is log10 power")
        if not 0 <= self.f_min_hz < self.sampling_frequency_hz / 2:
            raise ValueError("f_min_hz must lie below Nyquist")

    @property
    def frame_count(self) -> int:
        return 1 + (self.signal_length - self.win_length) // self.hop_length

    @property
    def full_frequency_count(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frequency_spacing_hz(self) -> float:
        return self.sampling_frequency_hz / self.n_fft

    @property
    def crop_start_bin(self) -> int:
        return int(math.ceil(self.f_min_hz / self.frequency_spacing_hz))

    @property
    def crop_stop_bin(self) -> int:
        return self.full_frequency_count - 1

    @property
    def frequency_count(self) -> int:
        return self.crop_stop_bin - self.crop_start_bin + 1

    @property
    def expected_shape(self) -> tuple[int, int, int]:
        return (1, self.frequency_count, self.frame_count)

    @property
    def nyquist_hz(self) -> float:
        return self.sampling_frequency_hz / 2.0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "frame_count": self.frame_count,
            "full_frequency_count": self.full_frequency_count,
            "frequency_spacing_hz": self.frequency_spacing_hz,
            "crop_start_bin": self.crop_start_bin,
            "crop_stop_bin": self.crop_stop_bin,
            "frequency_count": self.frequency_count,
            "expected_shape": list(self.expected_shape),
            "nyquist_hz": self.nyquist_hz,
        })
        return payload

    def cache_metadata(
        self,
        *,
        dataset: str,
        dataset_version: str,
        raw_data_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "dataset_version": dataset_version,
            "representation": self.as_dict(),
            "raw_data_fingerprint": raw_data_fingerprint,
            "preprocessing_version": self.representation_version,
            "normalized": False,
            "dtype": "float16",
        }


def compute_full_signal_log_power(values: np.ndarray, spec: FullSignalSTFTSpec) -> np.ndarray:
    """Compute finite ``float32 [N,1,F,T]`` global log10-power STFTs."""

    import torch

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != spec.signal_length:
        raise ValueError(f"Expected [N,{spec.signal_length}] values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Global STFT input contains non-finite values")
    window = torch.hann_window(spec.win_length, periodic=True, dtype=torch.float32)
    frames = torch.from_numpy(array).unfold(1, spec.win_length, spec.hop_length)
    framed = frames * window.view(1, 1, -1)
    transformed = torch.fft.rfft(framed, n=spec.n_fft, dim=-1)
    power = transformed.real.square() + transformed.imag.square()
    log_power = 10.0 * torch.log10(power + spec.epsilon)
    result = (
        log_power.transpose(1, 2)
        .unsqueeze(1)[:, :, spec.crop_start_bin : spec.crop_stop_bin + 1, :]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    expected = (array.shape[0], *spec.expected_shape)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Unexpected or non-finite global STFT: {result.shape}, expected {expected}")
    return result


@dataclass(frozen=True)
class FrequencyStandardizer:
    """Train-only per-frequency standardizer applied lazily over time."""

    mean: np.ndarray
    std: np.ndarray
    fit_count: int
    fingerprint: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Global spectrogram values contain non-finite values")
        result = ((array - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Global spectrogram normalization produced non-finite values")
        return result

    def view(self, values: np.ndarray) -> "FrequencyStandardizedView":
        return FrequencyStandardizedView(values, self)


class FrequencyStandardizedView:
    """Array-like view that never materializes a second global cache."""

    def __init__(self, values: np.ndarray, standardizer: FrequencyStandardizer) -> None:
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) != 4 or shape[1] != 1:
            raise ValueError("Expected global spectrogram values with shape [N,1,F,T]")
        self.values = values
        self.standardizer = standardizer
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, index: Any) -> np.ndarray:
        return self.standardizer.transform(np.asarray(self.values[index], dtype=np.float32))


def fit_frequency_standardizer(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = 4,
) -> FrequencyStandardizer:
    """Fit frequency-only moments using selected signals and no other rows."""

    shape = getattr(values, "shape", None)
    selected = np.asarray(indices, dtype=np.int64)
    if shape is None or len(shape) != 4 or shape[1] != 1:
        raise ValueError("values must expose [N,1,F,T] global spectrograms")
    if selected.ndim != 1 or selected.size == 0 or np.any(selected < 0) or np.any(selected >= shape[0]):
        raise ValueError("indices must be non-empty valid signal indices")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = np.zeros(shape[2], dtype=np.float64)
    squares = np.zeros(shape[2], dtype=np.float64)
    count = 0
    for start in range(0, len(selected), chunk_size):
        chunk = np.asarray(values[selected[start : start + chunk_size]], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError("Cannot fit a standardizer from non-finite cache values")
        flattened = chunk[:, 0].transpose(0, 2, 1).reshape(-1, chunk.shape[2])
        total += flattened.sum(axis=0, dtype=np.float64)
        squares += np.square(flattened, dtype=np.float64).sum(axis=0)
        count += len(flattened)
    if count == 0:
        raise ValueError("No samples selected for standardization")
    mean = (total / count).astype(np.float32).reshape(1, -1, 1)
    variance = np.maximum(squares / count - np.square(mean.astype(np.float64).reshape(-1)), 0.0)
    std = np.sqrt(variance).astype(np.float32).reshape(1, -1, 1)
    std[std < 1e-6] = 1.0
    fingerprint = hashlib.sha256(mean.tobytes() + std.tobytes() + str(count).encode()).hexdigest()
    return FrequencyStandardizer(mean=mean, std=std, fit_count=int(count), fingerprint=fingerprint)


def cache_fingerprint(metadata: dict[str, Any]) -> str:
    """Return a stable fingerprint for a JSON-compatible cache contract."""

    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_global_spectrogram_cache(
    path: str | Path,
    *,
    shape: tuple[int, int, int, int],
    metadata: dict[str, Any],
    writer: Callable[[np.ndarray], None],
) -> dict[str, Any]:
    """Write a float16 memmap and completion metadata atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    writer(values)
    values.flush()
    del values
    temporary.replace(destination)
    payload = dict(metadata)
    payload.update({"shape": list(shape), "dtype": "float16"})
    payload["fingerprint"] = cache_fingerprint(payload)
    meta_path = Path(str(destination) + ".json")
    meta_tmp = Path(str(meta_path) + ".tmp")
    meta_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_tmp.replace(meta_path)
    Path(str(destination) + ".complete").write_text("complete\n", encoding="utf-8")
    return payload


def require_global_spectrogram_cache(path: str | Path, expected_fingerprint: str) -> np.ndarray:
    """Open only complete, shape-consistent, fingerprinted global caches."""

    destination = Path(path)
    meta_path = Path(str(destination) + ".json")
    marker = Path(str(destination) + ".complete")
    if not destination.is_file() or not meta_path.is_file() or not marker.is_file():
        raise ValueError(f"Incomplete global spectrogram cache: {destination}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    recorded = metadata.get("fingerprint")
    unsigned = dict(metadata)
    unsigned.pop("fingerprint", None)
    if recorded != cache_fingerprint(unsigned):
        raise ValueError("Global spectrogram cache metadata fingerprint is invalid")
    if recorded != expected_fingerprint:
        raise ValueError("Global spectrogram cache fingerprint mismatch")
    values = np.load(destination, mmap_mode="r", allow_pickle=False)
    if tuple(values.shape) != tuple(metadata["shape"]) or str(values.dtype) != metadata["dtype"]:
        raise ValueError("Global spectrogram cache shape or dtype mismatch")
    if not np.isfinite(np.asarray(values[:1], dtype=np.float32)).all():
        raise ValueError("Global spectrogram cache contains non-finite values")
    return values
