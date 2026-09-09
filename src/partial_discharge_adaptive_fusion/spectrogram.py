"""Deterministic natural-log-power STFTs and train-only standardization.

The module deliberately keeps the cache representation independent from model
training: raw log-power values are computed once, while standardizers are fit
on the active training subset and applied lazily by callers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class STFTSpec:
    """Registered STFT parameters for one signal geometry."""

    n_fft: int
    win_length: int
    hop_length: int
    epsilon: float = 1e-8
    center: bool = False
    onesided: bool = True
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.n_fft < 2 or self.win_length < 2 or self.win_length > self.n_fft:
            raise ValueError("n_fft and win_length must satisfy 2 <= win_length <= n_fft")
        if self.hop_length < 1 or self.epsilon <= 0:
            raise ValueError("hop_length must be positive and epsilon must be positive")
        if self.window != "hann" or not self.onesided or self.center:
            raise ValueError("This diagnostic requires Hann, one-sided, center=False STFTs")

    def frame_count(self, signal_length: int) -> int:
        if signal_length < self.win_length:
            raise ValueError("Signal is shorter than the STFT window")
        return 1 + (signal_length - self.win_length) // self.hop_length

    def frequency_count(self) -> int:
        return self.n_fft // 2 + 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectrogramStandardizer:
    """Per-frequency/time train-only standardizer."""

    mean: np.ndarray
    std: np.ndarray
    fit_count: int
    fingerprint: str

    def transform(self, values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Spectrogram values contain non-finite entries")
        output = ((array - self.mean) / self.std).astype(np.float32)
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if array.ndim < 4 or mask.shape != array.shape[:3]:
                raise ValueError("valid_mask must match [signals, halves, pulses]")
            output[~mask] = 0.0
        return output


def compute_stft_log_power(values: np.ndarray, spec: STFTSpec) -> np.ndarray:
    """Return finite float32 ``[N, 1, frequencies, time]`` log-power STFTs."""

    import torch

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError(f"Expected [N, time] values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("STFT input contains non-finite values")
    window = torch.hann_window(spec.win_length, periodic=True, dtype=torch.float32)
    tensor = torch.from_numpy(array)
    # Explicit framing is intentional. ``torch.stft`` uses ``n_fft`` as the
    # frame extent when ``center=False``; the registered MATLAB geometry uses
    # ``win_length`` frames zero-padded only inside each FFT instead.
    frames = tensor.unfold(1, spec.win_length, spec.hop_length)
    framed = frames * window.view(1, 1, -1)
    transformed = torch.fft.rfft(framed, n=spec.n_fft, dim=-1)
    power = transformed.real.square() + transformed.imag.square()
    result = torch.log(power + spec.epsilon).transpose(1, 2).unsqueeze(1).cpu().numpy().astype(np.float32)
    expected = (array.shape[0], 1, spec.frequency_count(), spec.frame_count(array.shape[1]))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"Unexpected or non-finite STFT result: {result.shape}, expected {expected}")
    return result


def fit_spectrogram_standardizer(
    values: np.ndarray,
    indices: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> SpectrogramStandardizer:
    """Fit statistics only on selected signals or valid bag events."""

    array = np.asarray(values, dtype=np.float32)
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or np.any(selected < 0) or np.any(selected >= len(array)):
        raise ValueError("indices must be non-empty valid signal indices")
    if not np.isfinite(array).all():
        raise ValueError("Cannot fit a standardizer from non-finite values")
    if valid_mask is None:
        samples = array[selected]
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if array.ndim != 6 or mask.shape != array.shape[:3]:
            raise ValueError("Bag standardization expects values [N,2,K,1,F,T] and [N,2,K] mask")
        chosen = mask[selected]
        samples = array[selected][chosen]
        if len(samples) == 0:
            raise ValueError("No valid bag events selected for standardization")
    flattened = samples.reshape(-1, *samples.shape[-2:])
    mean = flattened.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = flattened.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    payload = mean.tobytes() + std.tobytes() + str(int(len(flattened))).encode()
    return SpectrogramStandardizer(mean=mean, std=std, fit_count=int(len(flattened)), fingerprint=hashlib.sha256(payload).hexdigest())


def cache_fingerprint(metadata: dict[str, Any]) -> str:
    """Create a stable cache fingerprint from JSON-serializable metadata."""

    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_atomic_array_cache(path: str | Path, values: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write an unnormalized array and a completion marker atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError("Refusing to cache non-finite spectrogram values")
    payload = dict(metadata)
    payload.update({"shape": list(array.shape), "dtype": str(array.dtype)})
    payload["fingerprint"] = cache_fingerprint(payload)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    np.save(temporary, array)
    generated = Path(str(temporary) + ".npy") if not temporary.exists() else temporary
    generated.replace(destination)
    meta_path = destination.with_suffix(destination.suffix + ".json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_tmp.replace(meta_path)
    destination.with_suffix(destination.suffix + ".complete").write_text("complete\n", encoding="utf-8")
    return payload


def require_valid_cache(path: str | Path, expected_fingerprint: str) -> np.ndarray:
    """Open only a complete cache whose metadata fingerprint matches."""

    destination = Path(path)
    marker = destination.with_suffix(destination.suffix + ".complete")
    meta_path = destination.with_suffix(destination.suffix + ".json")
    if not destination.is_file() or not marker.is_file() or not meta_path.is_file():
        raise ValueError(f"Incomplete spectrogram cache: {destination}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("Spectrogram cache fingerprint mismatch")
    values = np.load(destination, mmap_mode="r", allow_pickle=False)
    if tuple(values.shape) != tuple(metadata["shape"]) or str(values.dtype) != metadata["dtype"]:
        raise ValueError("Spectrogram cache shape or dtype mismatch")
    return values


def build_matlab_spectrogram_dataset(signals: np.ndarray, spec: STFTSpec) -> np.ndarray:
    """Build the MATLAB spectrogram representation without touching holdouts."""

    values = np.asarray(signals, dtype=np.float32)
    return compute_stft_log_power(values, spec)


def build_vsb_event_spectrogram_dataset(
    signals: np.ndarray,
    policy: Any,
    spec: STFTSpec,
    *,
    segment_length: int = 512,
    capacity: int = 86,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``[N,2,capacity,1,F,T]`` STFT bags from complete VSB signals."""

    from .pulse import prepare_v5_stft_event_bag

    source = np.asarray(signals, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("VSB signals must be [N, time]")
    bags: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for signal in source:
        bag = prepare_v5_stft_event_bag(signal, policy, segment_length=segment_length, capacity=capacity)
        flat = compute_stft_log_power(bag.segments.reshape(-1, segment_length), spec)
        bags.append(flat.reshape(2, capacity, *flat.shape[1:]))
        masks.append(bag.valid_mask)
        indices.append(bag.peak_indices)
    return np.stack(bags).astype(np.float32), np.stack(masks), np.stack(indices)


__all__ = [
    "STFTSpec", "SpectrogramStandardizer", "cache_fingerprint",
    "compute_stft_log_power", "fit_spectrogram_standardizer",
    "write_atomic_array_cache", "require_valid_cache", "build_matlab_spectrogram_dataset",
    "build_vsb_event_spectrogram_dataset",
]
