"""Reusable, label-safe diagnostics for the local VSB forensic audit.

The functions in this module deliberately operate on parent signals and retain
``signal_id``/``id_measurement``/``phase`` provenance.  They do not open the
VSB grouped holdout, MATLAB Te2, or the official unlabeled VSB test.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import (
    VSB_DATASET_ID,
    DatasetHandle,
    audit_vsb_metadata,
    dataset_provenance,
    inspect_vsb_parquet_schema,
    iter_vsb_signal_batches,
    load_vsb_metadata,
)
from .pulse import detect_cycle_reference, detect_pulses
from .pulse_impl import PulsePolicy, flatten_signal
from .splits import SplitManifest, vsb_grouped_manifest


AUDIT_VERSION = "vsb-literature-forensic-v1"
DETECTOR_IDS = ("v5_current", "michau_anchor", "chen_anchor", "dualcycon_strict")
SEGMENT_LENGTHS = (30, 40, 128, 512)


@dataclass(frozen=True)
class DetectorSpec:
    """Fixed detector parameters recorded with every audit run."""

    detector_id: str
    locality_samples: int
    prominence_mad: float
    max_kept_peaks: int = 1024
    source_status: str = "PROJECT_LITERATURE_ANCHOR_UNLESS_VERIFIED"


@dataclass(frozen=True)
class AlignmentResult:
    """One label-free cycle-reference result."""

    method: str
    success: bool
    origin: int = -1
    opposite: int = -1
    half_period: int = -1
    inferred: bool = False
    frequency_hz: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class DetectorOutput:
    """Compact detector output for one parent signal."""

    detector_id: str
    peaks: np.ndarray
    scores: np.ndarray
    signs: np.ndarray
    raw_candidate_count: int
    noise_scale: float
    origin: int
    alignment_status: str
    boundary_count: int
    transformed: np.ndarray


@dataclass(frozen=True)
class AuditRuntime:
    """Runtime counters written to the audit manifest."""

    started_epoch: float
    elapsed_seconds: float
    signals_processed: int
    batches_processed: int
    failed_signals: int


def detector_specs() -> dict[str, DetectorSpec]:
    """Return the predeclared detector definitions."""

    return {
        "v5_current": DetectorSpec("v5_current", 128, 0.0, 257, "VERIFIED_FROM_REPOSITORY"),
        "michau_anchor": DetectorSpec("michau_anchor", 40, 6.0, 1024),
        "chen_anchor": DetectorSpec("chen_anchor", 30, 6.0, 1024),
        "dualcycon_strict": DetectorSpec("dualcycon_strict", 512, 0.0, 257),
    }


def canonical_fingerprint(value: Any) -> str:
    """Hash a JSON-compatible value deterministically."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return portable file identity without copying or modifying the file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(source), "size_bytes": source.stat().st_size, "sha256": digest.hexdigest()}


def audit_provenance(
    dataset: DatasetHandle,
    metadata: pd.DataFrame,
    *,
    config_path: str | Path,
    historical_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build a compact audit provenance record."""

    historical = []
    for raw_path in historical_paths:
        path = Path(raw_path)
        if path.is_file():
            historical.append(file_fingerprint(path))
        elif path.is_dir():
            files = [file_fingerprint(item) for item in sorted(path.rglob("*")) if item.is_file()]
            historical.append({"path": str(path), "files": files})
        else:
            historical.append({"path": str(path), "available": False})
    return {
        "audit_version": AUDIT_VERSION,
        "dataset": dataset_provenance(dataset),
        "metadata_fingerprint": canonical_fingerprint(metadata.to_dict(orient="records")),
        "config_fingerprint": file_fingerprint(config_path),
        "historical_inputs": historical,
        "data_contract": "local_raw/PD_RAW_DATA_ROOT",
        "researchdata": "forbidden",
        "dropbox": "forbidden",
    }


def build_development_manifest(
    metadata: pd.DataFrame,
    *,
    split_seed: int = 42,
    oof_folds: int = 5,
) -> tuple[SplitManifest, dict[str, Any]]:
    """Build and validate the V5 grouped manifest for audit-accessible data."""

    manifest = vsb_grouped_manifest(
        metadata,
        dataset_id=VSB_DATASET_ID,
        dataset_version="2018-kaggle-snapshot",
        split_seed=split_seed,
        oof_splits=oof_folds,
    )
    manifest.validate()
    frame = manifest.frame
    train_groups = set(frame.loc[frame["split"] == "train", "group_id"].astype(str))
    validation_groups = set(frame.loc[frame["split"] == "validation", "group_id"].astype(str))
    overlap = sorted(train_groups & validation_groups)
    development = frame[frame["split"].isin(["train", "validation"])].copy()
    summary = {
        "manifest_fingerprint": canonical_fingerprint(frame.to_dict(orient="records")),
        "development_signal_count": int(len(development)),
        "development_measurement_count": int(development["group_id"].nunique()),
        "split_counts": {
            str(split): {
                "signals": int((frame["split"] == split).sum()),
                "measurements": int(frame.loc[frame["split"] == split, "group_id"].nunique()),
                "positive_signals": int(frame.loc[frame["split"] == split, "label"].sum()),
            }
            for split in ("train", "validation", "test")
        },
        "train_validation_group_overlap": overlap,
        "group_isolation_status": "PASS" if not overlap else "FAIL",
        "holdout_rows_seen_by_audit": int((frame["split"] == "test").sum()),
    }
    if overlap:
        raise ValueError(f"VSB development groups overlap: {overlap[:5]}")
    return manifest, summary


def phase_pattern_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic three-phase label patterns ordered by phase."""

    required = {"id_measurement", "phase", "target"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    ordered = metadata.assign(_phase=metadata["phase"].astype(int)).sort_values(
        ["id_measurement", "_phase"]
    )
    grouped = ordered.groupby("id_measurement", sort=True)["target"]
    patterns = grouped.apply(lambda values: "".join(values.astype(int).astype(str)))
    counts = patterns.value_counts().reindex(list("000 001 010 011 100 101 110 111".split()), fill_value=0)
    result = pd.DataFrame({"pattern": counts.index, "measurements": counts.to_numpy(dtype=np.int64)})
    result["proportion"] = result["measurements"] / max(len(patterns), 1)
    return result


def dataset_identity_audit(dataset: DatasetHandle, metadata: pd.DataFrame) -> dict[str, Any]:
    """Combine metadata, parquet schema, and bounded signal integrity checks."""

    metadata_result = audit_vsb_metadata(metadata)
    schema = inspect_vsb_parquet_schema(dataset)
    sample = next(iter_vsb_signal_batches(dataset, metadata.head(3), batch_size=3)).signal
    return {
        "metadata": metadata_result,
        "metadata_columns": metadata.columns.tolist(),
        "parquet_schema": {
            "num_rows": schema["num_rows"],
            "num_columns": schema["num_columns"],
            "num_row_groups": schema["num_row_groups"],
            "column_type_counts": pd.Series([item["type"] for item in schema["columns"]]).value_counts().to_dict(),
        },
        "sample_signal": {
            "shape": list(sample.shape),
            "finite": bool(np.isfinite(sample).all()),
            "minimum": float(np.min(sample)),
            "maximum": float(np.max(sample)),
            "dtype": str(sample.dtype),
            "sampling_frequency_hz": 40_000_000.0,
            "duration_seconds": float(sample.shape[1] / 40_000_000.0),
        },
        "integrity_status": "PASS" if bool(np.isfinite(sample).all()) else "FAIL",
    }


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    return float(np.median(np.abs(values - median)) * 1.4826)


def robust_noise_stats(values: np.ndarray) -> dict[str, float]:
    """Return robust and percentile noise summaries for one vector."""

    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    centered = values - center
    return {
        "median": center,
        "mad_std": _mad(values),
        "std": float(np.std(values)),
        "p01_abs": float(np.percentile(np.abs(centered), 1)),
        "p50_abs": float(np.percentile(np.abs(centered), 50)),
        "p90_abs": float(np.percentile(np.abs(centered), 90)),
        "p99_abs": float(np.percentile(np.abs(centered), 99)),
    }


def _dft_alignment(signal: np.ndarray, fs: float = 40_000_000.0) -> AlignmentResult:
    """Fit fixed 50/60 Hz sinusoidal candidates to a smoothed signal."""

    values = np.asarray(signal, dtype=np.float64)
    smooth = uniform_filter1d(values, size=10_000, mode="nearest")
    step = 1000
    sample = smooth[::step]
    time_axis = np.arange(len(sample), dtype=np.float64) * step / fs
    best: tuple[float, float, np.ndarray] | None = None
    for frequency in (50.0, 60.0):
        design = np.column_stack((np.sin(2 * np.pi * frequency * time_axis), np.cos(2 * np.pi * frequency * time_axis), np.ones_like(time_axis)))
        coefficients, *_ = np.linalg.lstsq(design, sample, rcond=None)
        fitted = design @ coefficients
        residual = float(np.mean((sample - fitted) ** 2))
        candidate = (residual, frequency, coefficients)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None or not np.isfinite(best[0]):
        return AlignmentResult("dft_sinusoid", False, reason="non-finite fit")
    _, frequency, coefficients = best
    fitted = coefficients[0] * np.sin(2 * np.pi * frequency * time_axis) + coefficients[1] * np.cos(2 * np.pi * frequency * time_axis) + coefficients[2]
    centered = fitted - coefficients[2]
    crossings = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0))
    if len(crossings) == 0:
        return AlignmentResult("dft_sinusoid", False, frequency_hz=frequency, reason="no rising crossing")
    origin = int(crossings[0] * step)
    return AlignmentResult(
        "dft_sinusoid", True, origin=origin, opposite=int((origin + fs / (2 * frequency)) % len(values)),
        half_period=int(round(fs / (2 * frequency))), frequency_hz=frequency,
    )


def alignment_methods(signal: np.ndarray, policy: PulsePolicy | None = None) -> dict[str, AlignmentResult]:
    """Run current, strict, and DFT alignment without consulting labels."""

    policy = policy or PulsePolicy()
    results: dict[str, AlignmentResult] = {}
    try:
        current = detect_cycle_reference(signal, policy)
        results["v5_current"] = AlignmentResult(
            "v5_current", True, current.origin, current.opposite, current.half_period,
            bool(getattr(current, "inferred_missing_crossing", False)),
        )
    except Exception as exc:  # noqa: BLE001 - audit records the failure
        results["v5_current"] = AlignmentResult("v5_current", False, reason=type(exc).__name__ + ": " + str(exc))
    try:
        from . import pulse_impl

        strict = pulse_impl.detect_cycle_reference(signal, policy)
        results["dualcycon_strict"] = AlignmentResult(
            "dualcycon_strict", True, strict.origin, strict.opposite, strict.half_period,
        )
    except Exception as exc:  # noqa: BLE001 - audit records the failure
        results["dualcycon_strict"] = AlignmentResult("dualcycon_strict", False, reason=type(exc).__name__ + ": " + str(exc))
    results["dft_sinusoid"] = _dft_alignment(signal, policy.sampling_frequency_hz)
    return results


def _alignment_degrees(left: AlignmentResult, right: AlignmentResult, signal_length: int) -> float | None:
    if not left.success or not right.success:
        return None
    difference = ((right.origin - left.origin) / signal_length * 360.0 + 180.0) % 360.0 - 180.0
    return float(abs(difference))


def alignment_summary(records: pd.DataFrame, *, minimum_success_rate: float = 0.95, maximum_disagreement_degrees: float = 20.0) -> dict[str, Any]:
    """Summarize alignment success and assign VALID/PARTIAL/INVALID."""

    methods = ["v5_current", "dualcycon_strict", "dft_sinusoid"]
    success = {method: float(records[f"{method}_success"].mean()) for method in methods if f"{method}_success" in records}
    disagreement_columns = [column for column in records.columns if column.endswith("_disagreement_degrees")]
    median_disagreement = float(np.nanmedian(records[disagreement_columns].to_numpy())) if disagreement_columns else float("nan")
    all_method_success = records[[f"{method}_success" for method in methods if f"{method}_success" in records]].all(axis=1)
    all_phase = records.groupby("id_measurement")["v5_current_success"].sum() if "v5_current_success" in records else pd.Series(dtype=float)
    complete_measurements = float((all_phase == 3).mean()) if len(all_phase) else 0.0
    valid = bool(
        all(success_rate >= minimum_success_rate for success_rate in success.values())
        and complete_measurements >= minimum_success_rate
        and np.isfinite(median_disagreement)
        and median_disagreement <= maximum_disagreement_degrees
    )
    status = "VALID" if valid else "PARTIAL/AMBIGUOUS" if all(success_rate >= 0.5 for success_rate in success.values()) else "INVALID"
    return {
        "status": status,
        "method_success_rates": success,
        "all_method_success_rate": float(all_method_success.mean()),
        "complete_measurement_rate": complete_measurements,
        "median_disagreement_degrees": median_disagreement,
        "minimum_success_rate": minimum_success_rate,
        "maximum_disagreement_degrees": maximum_disagreement_degrees,
        "forced_120_degree_relation": False,
    }


def _signal_indices(peaks: np.ndarray, origin: int, length: int) -> np.ndarray:
    if origin < 0:
        return np.asarray(peaks, dtype=np.int64)
    return (np.asarray(peaks, dtype=np.int64) + int(origin)) % length


def _width_proxy(values: np.ndarray, peaks: np.ndarray, scores: np.ndarray, radius: int = 64) -> np.ndarray:
    widths = []
    values = np.abs(np.asarray(values, dtype=np.float64))
    for peak, score in zip(peaks, scores):
        start = max(0, int(peak) - radius)
        stop = min(len(values), int(peak) + radius + 1)
        local = values[start:stop]
        threshold = max(float(score) * 0.5, 1e-12)
        above = np.flatnonzero(local >= threshold)
        widths.append(float(above[-1] - above[0] + 1) if len(above) else 0.0)
    return np.asarray(widths, dtype=np.float64)


def _generic_peaks(values: np.ndarray, spec: DetectorSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    centered = np.asarray(values, dtype=np.float64) - np.median(values)
    noise = max(_mad(centered), float(np.std(centered) * 0.05), 1e-6)
    kwargs: dict[str, Any] = {"distance": spec.locality_samples}
    if spec.prominence_mad > 0:
        kwargs["prominence"] = spec.prominence_mad * noise
    indices, properties = find_peaks(np.abs(centered), **kwargs)
    scores = np.abs(centered[indices]).astype(np.float32)
    order = np.argsort(-scores, kind="mergesort")
    all_count = int(len(indices))
    kept = order[: spec.max_kept_peaks]
    indices = indices[kept].astype(np.int64)
    scores = scores[kept]
    signs = np.sign(centered[indices]).astype(np.int8)
    return indices, scores, signs, all_count, float(noise)


def detect_forensic_detector(signal: np.ndarray, detector_id: str, policy: PulsePolicy | None = None) -> DetectorOutput:
    """Run one of the predeclared label-free detector adaptations."""

    policy = policy or PulsePolicy()
    specs = detector_specs()
    if detector_id not in specs:
        raise ValueError(f"Unknown detector: {detector_id}")
    values = np.asarray(signal, dtype=np.float32)
    length = len(values)
    origin = -1
    alignment_status = "UNRESOLVED"
    if detector_id == "v5_current":
        detected = detect_pulses(values, policy)
        origin = int(detected.reference.origin)
        peaks = np.concatenate([_signal_indices(item, origin, length) for item in detected.peak_indices])
        scores = np.concatenate([item for item in detected.peak_amplitudes]).astype(np.float32)
        aligned = np.roll(values, -origin)
        aligned_peaks = np.concatenate(detected.peak_indices).astype(np.int64)
        signs = np.sign(aligned[aligned_peaks]).astype(np.int8)
        alignment_status = "VALID" if not getattr(detected.reference, "inferred_missing_crossing", False) else "INFERRED"
        transformed = detected.flattened_signal
        raw_count = int(len(peaks))
        boundary_count = int(np.sum((peaks < policy.temporal_length // 2) | (peaks >= length - policy.temporal_length // 2)))
    elif detector_id == "dualcycon_strict":
        from . import pulse_impl

        reference = pulse_impl.detect_cycle_reference(values, policy)
        origin = int(reference.origin)
        aligned = np.roll(values, -origin)
        transformed = flatten_signal(aligned, policy)
        all_peaks, _ = find_peaks(np.abs(transformed), distance=512)
        selected: list[np.ndarray] = []
        score_list: list[np.ndarray] = []
        for start, stop in ((0, policy.half_length), (policy.half_length, length)):
            half_peaks = all_peaks[(all_peaks >= start) & (all_peaks < stop)]
            score = np.abs(transformed[half_peaks])
            order = np.argsort(-score, kind="mergesort")[:257]
            selected.append(half_peaks[order])
            score_list.append(score[order].astype(np.float32))
        peaks = _signal_indices(np.concatenate(selected), origin, length)
        scores = np.concatenate(score_list)
        signs = np.sign(values[peaks]).astype(np.int8)
        raw_count = int(len(all_peaks))
        boundary_count = int(np.sum((peaks < 256) | (peaks >= length - 256)))
        alignment_status = "VALID"
    else:
        if detector_id == "michau_anchor":
            sos = butter(5, 20_000.0, btype="highpass", fs=policy.sampling_frequency_hz, output="sos")
            transformed = sosfiltfilt(sos, values.astype(np.float64)).astype(np.float32)
        else:
            baseline = savgol_filter(values.astype(np.float64), 99, 3, mode="interp")
            transformed = (values.astype(np.float64) - baseline).astype(np.float32)
        peaks, scores, signs, raw_count, noise = _generic_peaks(transformed, specs[detector_id])
        boundary_count = int(np.sum((peaks < specs[detector_id].locality_samples // 2) | (peaks >= length - specs[detector_id].locality_samples // 2)))
        return DetectorOutput(detector_id, peaks, scores, signs, raw_count, noise, origin, alignment_status, boundary_count, transformed)
    noise = max(_mad(transformed), 1e-6)
    return DetectorOutput(detector_id, peaks, scores, signs, raw_count, noise, origin, alignment_status, boundary_count, transformed)


def detector_features(output: DetectorOutput, signal_length: int = 800_000, bins: int = 20) -> dict[str, float]:
    """Create a compact, deterministic per-signal feature vector."""

    peaks = np.asarray(output.peaks, dtype=np.int64)
    scores = np.asarray(output.scores, dtype=np.float64)
    signs = np.asarray(output.signs, dtype=np.int8)
    prefix = output.detector_id
    features: dict[str, float] = {
        f"{prefix}_candidate_count": float(output.raw_candidate_count),
        f"{prefix}_selected_count": float(len(peaks)),
        f"{prefix}_noise_mad": float(output.noise_scale),
        f"{prefix}_boundary_fraction": float(output.boundary_count / max(len(peaks), 1)),
        f"{prefix}_score_median": float(np.median(scores)) if len(scores) else 0.0,
        f"{prefix}_score_p90": float(np.percentile(scores, 90)) if len(scores) else 0.0,
        f"{prefix}_score_max": float(np.max(scores)) if len(scores) else 0.0,
        f"{prefix}_snr_p90": float(np.percentile(scores / max(output.noise_scale, 1e-6), 90)) if len(scores) else 0.0,
        f"{prefix}_positive_fraction": float(np.mean(signs > 0)) if len(signs) else 0.0,
    }
    if len(peaks) > 1:
        distances = np.diff(np.sort(peaks)).astype(np.float64)
        features[f"{prefix}_spacing_median"] = float(np.median(distances))
        features[f"{prefix}_spacing_p10"] = float(np.percentile(distances, 10))
    else:
        features[f"{prefix}_spacing_median"] = 0.0
        features[f"{prefix}_spacing_p10"] = 0.0
    widths = _width_proxy(output.transformed, peaks, scores)
    features[f"{prefix}_width_median"] = float(np.median(widths)) if len(widths) else 0.0
    positions = peaks / max(signal_length, 1)
    for index in range(4):
        lower, upper = index / 4, (index + 1) / 4
        features[f"{prefix}_quadrant_{index:02d}_count"] = float(np.sum((positions >= lower) & (positions < upper)))
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        features[f"{prefix}_phase_bin_{index:02d}_count"] = float(np.sum((positions >= lower) & (positions < upper)))
    return features


def detector_match(left: DetectorOutput, right: DetectorOutput, tolerance: int = 32) -> dict[str, float]:
    """Match detector peaks greedily at a fixed temporal tolerance."""

    left_peaks = np.sort(np.asarray(left.peaks, dtype=np.int64))
    right_peaks = np.sort(np.asarray(right.peaks, dtype=np.int64))
    used: set[int] = set()
    matches = 0
    offsets: list[int] = []
    for peak in left_peaks:
        candidates = [int(index) for index, value in enumerate(right_peaks) if index not in used and abs(int(value) - int(peak)) <= tolerance]
        if not candidates:
            continue
        index = min(candidates, key=lambda item: abs(int(right_peaks[item]) - int(peak)))
        used.add(index)
        matches += 1
        offsets.append(int(right_peaks[index]) - int(peak))
    denominator = max(len(left_peaks) + len(right_peaks) - matches, 1)
    return {
        "left_count": float(len(left_peaks)),
        "right_count": float(len(right_peaks)),
        "matched_count": float(matches),
        "matched_fraction": float(matches / denominator),
        "median_offset_samples": float(np.median(offsets)) if offsets else float("nan"),
        "median_abs_offset_samples": float(np.median(np.abs(offsets))) if offsets else float("nan"),
    }


def phase_effect_table(features: pd.DataFrame, detector_id: str) -> pd.DataFrame:
    """Return PD/NonPD summaries for fixed phase-bin counts."""

    rows = []
    for index in range(20):
        column = f"{detector_id}_phase_bin_{index:02d}_count"
        if column not in features:
            continue
        for label, group in features.groupby("target", sort=True):
            values = group[column].to_numpy(dtype=float)
            rows.append({
                "detector": detector_id,
                "phase_bin": index,
                "target": int(label),
                "n": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else float("nan"),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "p90": float(np.percentile(values, 90)) if len(values) else float("nan"),
            })
    return pd.DataFrame(rows)


def _safe_metric(metric: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(metric(*args, **kwargs))
    except ValueError:
        return float("nan")


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute the audit's fixed classification metric set."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "mcc": _safe_metric(matthews_corrcoef, labels, predicted),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": _safe_metric(precision_score, labels, predicted, zero_division=0),
        "recall": _safe_metric(recall_score, labels, predicted, zero_division=0),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": _safe_metric(f1_score, labels, predicted, zero_division=0),
        "roc_auc": _safe_metric(roc_auc_score, labels, probabilities),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _make_estimator(classifier: str, seed: int):
    if classifier == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed),
        )
    if classifier == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"Unsupported audit classifier: {classifier}")


def _feature_columns(frame: pd.DataFrame, detector_id: str) -> list[str]:
    prefix = f"{detector_id}_"
    return [column for column in frame.columns if column.startswith(prefix) and frame[column].dtype.kind in "bifu"]


def make_three_phase_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Attach ordered phase-0/1/2 feature blocks to every signal row."""

    base = frame[["sample_id", "id_measurement", "phase", "target", "split", "oof_fold"]].copy()
    phase = frame.pivot(index="id_measurement", columns="phase", values=list(columns))
    phase.columns = [f"phase_{phase_id}_{column}" for column, phase_id in phase.columns]
    phase = phase.reset_index()
    result = base.merge(phase, on="id_measurement", how="left", validate="many_to_one")
    return result


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.arange(0.05, 0.951, 0.005)
    scores = [matthews_corrcoef(labels, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.nanargmax(scores))])


def _inner_oof_predictions(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classifier: str,
    seed: int,
    grouped: bool,
) -> np.ndarray:
    splitter = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        if grouped else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )
    result = np.full(len(labels), np.nan, dtype=float)
    split_args = (X, labels, groups) if grouped else (X, labels)
    for train_index, holdout_index in splitter.split(*split_args):
        estimator = _make_estimator(classifier, seed)
        estimator.fit(X[train_index], labels[train_index])
        result[holdout_index] = estimator.predict_proba(X[holdout_index])[:, 1]
    if not np.isfinite(result).all():
        raise ValueError("Inner OOF predictions are incomplete")
    return result


def evaluate_feature_baseline(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    classifier: str,
    seed: int,
    grouped: bool = True,
    mode: str = "phase_independent",
) -> dict[str, Any]:
    """Evaluate one fixed feature/classifier protocol without validation leakage."""

    if mode == "measurement_aware":
        prepared = make_three_phase_features(frame, columns)
        columns = [column for column in prepared.columns if column.startswith("phase_")]
    else:
        prepared = frame
    train = prepared[prepared["split"] == "train"].copy()
    validation = prepared[prepared["split"] == "validation"].copy()
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Baseline requires non-empty train and validation partitions")
    columns = list(columns)
    medians = train[columns].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    train_values = train[columns].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)
    validation_values = validation[columns].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)
    train_labels = train["target"].to_numpy(dtype=np.int64)
    validation_labels = validation["target"].to_numpy(dtype=np.int64)
    groups = train["id_measurement"].astype(str).to_numpy()
    oof = _inner_oof_predictions(train_values, train_labels, groups, classifier, seed, grouped)
    threshold = _select_threshold(train_labels, oof)
    estimator = _make_estimator(classifier, seed)
    estimator.fit(train_values, train_labels)
    probabilities = estimator.predict_proba(validation_values)[:, 1]
    metrics = classification_metrics(validation_labels, probabilities, threshold)
    return {
        "classifier": classifier,
        "seed": int(seed),
        "grouped": bool(grouped),
        "mode": mode,
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_features": int(len(columns)),
        "threshold_from_train_oof": threshold,
        "metrics": metrics,
    }


def evaluate_random_signal_protocol(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    classifier: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate an intentional signal-level split analogue on development data."""

    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    labels = frame["target"].to_numpy(dtype=np.int64)
    train_index: list[int] = []
    validation_index: list[int] = []
    for label in (0, 1):
        label_indices = indices[labels == label]
        rng.shuffle(label_indices)
        cut = int(round(len(label_indices) * 0.75))
        train_index.extend(label_indices[:cut].tolist())
        validation_index.extend(label_indices[cut:].tolist())
    prepared = frame.copy()
    prepared["split"] = "validation"
    prepared.loc[train_index, "split"] = "train"
    result = evaluate_feature_baseline(
        prepared, columns, classifier=classifier, seed=seed, grouped=False, mode="phase_independent",
    )
    result["diagnostic_protocol"] = "random_signal_level_split"
    result["group_overlap_count"] = int(
        len(set(prepared.iloc[train_index]["id_measurement"]) & set(prepared.iloc[validation_index]["id_measurement"]))
    )
    return result


def cwt_log_power_summary(values: np.ndarray, *, scales: np.ndarray, time_bins: int, morlet_w0: float = 6.0) -> dict[str, float]:
    """Compute compact CWT summaries without retaining a full scalogram."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("CWT summary expects one one-dimensional segment")
    indices = np.linspace(0, len(values) - 1, time_bins).round().astype(int)
    signal_fft = np.fft.fft(values)
    centered = np.arange(len(values), dtype=np.float64) - len(values) / 2.0
    powers = []
    for scale in np.asarray(scales, dtype=np.float64):
        wavelet = np.exp(-0.5 * (centered / scale) ** 2) * np.exp(1j * morlet_w0 * centered / scale) / np.sqrt(scale)
        kernel = np.conj(np.fft.fft(np.fft.ifftshift(wavelet)))
        coefficients = np.fft.ifft(signal_fft * kernel)
        powers.append(np.abs(coefficients[indices]) ** 2 + 1e-8)
    power = np.asarray(powers, dtype=np.float64)
    log_power = np.log(power)
    energy = power.sum(axis=1)
    normalized = power / max(float(power.sum()), 1e-12)
    entropy = -float(np.sum(normalized * np.log(normalized + 1e-12)))
    temporal = power.sum(axis=0)
    concentration = float(np.sort(temporal)[-max(1, len(temporal) // 10):].sum() / max(temporal.sum(), 1e-12))
    return {
        "cwt_energy_low": float(energy[: max(1, len(energy) // 3)].mean()),
        "cwt_energy_mid": float(energy[max(1, len(energy) // 3): max(2, 2 * len(energy) // 3)].mean()),
        "cwt_energy_high": float(energy[max(2, 2 * len(energy) // 3):].mean()),
        "cwt_log_power_mean": float(log_power.mean()),
        "cwt_log_power_std": float(log_power.std()),
        "cwt_entropy": entropy,
        "cwt_temporal_concentration": concentration,
        "cwt_scale_of_max": float(np.argmax(energy)),
    }


def morphology_table(waveforms: np.ndarray, metadata: pd.DataFrame, *, random_state: int = 42, clusters: int = 12) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit bounded exploratory PCA/KMeans morphology summaries."""

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError("Morphology waveforms and metadata are misaligned")
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    normalized = values / np.maximum(scale, 1e-6)
    pca = PCA(n_components=min(10, normalized.shape[1], len(normalized)), random_state=random_state)
    components = pca.fit_transform(normalized)
    kmeans = MiniBatchKMeans(n_clusters=min(clusters, len(components)), random_state=random_state, n_init=10, batch_size=512)
    labels = kmeans.fit_predict(components)
    result = metadata.reset_index(drop=True).copy()
    result["morphology_cluster"] = labels
    result["polarity"] = np.sign(values[np.arange(len(values)), np.argmax(np.abs(values), axis=1)]).astype(int)
    summary = {
        "n_waveforms": int(len(values)),
        "waveform_length": int(values.shape[1]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_clusters": int(kmeans.n_clusters),
        "cluster_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
    }
    return result, summary


def runtime_record(started: float, signals: int, batches: int, failures: int) -> dict[str, Any]:
    """Create a serializable runtime record."""

    return asdict(AuditRuntime(time.time(), time.perf_counter() - started, signals, batches, failures))
