"""Development-only VSB modality contribution diagnostics.

This module consumes immutable outputs from the forensic audit. It does not
open raw VSB holdouts, train neural models, or select detectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .evaluation import fast_mcc
from .vsb_forensics import (
    canonical_fingerprint,
    file_fingerprint,
    make_three_phase_features,
    prepare_forensic_baseline_frame,
)


class ModalityDiagnosticError(RuntimeError):
    """Raised when a source lock, partition, or feature invariant fails."""


REQUIRED_MARKERS = ("integrity.complete", "scan.complete", "cwt.complete", "baseline.complete")
DETECTOR_PREFIXES = ("v5_current", "michau_anchor", "chen_anchor", "dualcycon_strict")
DETECTOR_SUFFIXES = (
    "candidate_count", "selected_count", "noise_mad", "boundary_fraction",
    "score_median", "score_p90", "score_max", "snr_p90", "positive_fraction",
    "spacing_median", "spacing_p10", "width_median",
)
STRUCTURAL_SUFFIXES = (
    "candidate_count", "selected_count", "noise_mad", "boundary_fraction",
    "quadrant_00_count", "quadrant_01_count", "quadrant_02_count", "quadrant_03_count",
) + tuple(f"phase_bin_{index:02d}_count" for index in range(20))
TEMPORAL_SUFFIXES = (
    "score_median", "score_p90", "score_max", "snr_p90", "positive_fraction",
    "spacing_median", "spacing_p10", "width_median",
)
CWT_EVENT = "cwt_event_count"
CWT_SUMMARIES = (
    "cwt_energy_low", "cwt_energy_mid", "cwt_energy_high", "cwt_log_power_mean",
    "cwt_log_power_std", "cwt_entropy", "cwt_temporal_concentration", "cwt_scale_of_max",
)
BASELINE_REFERENCE = {
    "phase_independent": {
        42: {"mcc": 0.627081395599982, "threshold": 0.145},
        43: {"mcc": 0.632049896620652, "threshold": 0.185},
        44: {"mcc": 0.6194280719688356, "threshold": 0.335},
    },
    "measurement_aware": {
        42: {"mcc": 0.6622834661822181, "threshold": 0.380},
        43: {"mcc": 0.6581113302240205, "threshold": 0.555},
        44: {"mcc": 0.6891576471408442, "threshold": 0.215},
    },
}
REQUIRED_SOURCE_HASHES = {
    "audit_config_sha256": "b1fb66d3216c81e23bd05653b933b2c5c1080f1ffac9abf4ff30040acc3fd2c3",
    "development_manifest_fingerprint": "234be431f062fe2058d3c5d4a7508b39eb4ce5f6fb55dfa49b68c81554ad78c7",
    "development_metadata_sha256": "20b10a4648045a36ab96279bd4d2f045ceb6025134cd95c5891442807682ddb6",
    "detector_features_sha256": "8fac617c3553326285c7d9b899d19ba12b3cef9c142274b3eadc29160c2e9686",
    "cwt_features_sha256": "7dbc80e427a46b25da4f6fb274bc3c2d2e55a0202b32b25ec76199077ee94354",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModalityDiagnosticError(message)


def validate_modality_source_artifacts(
    audit_root: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate immutable forensic inputs and development-only group boundaries."""

    root = Path(audit_root)
    _require(root.is_dir(), f"Forensic audit root is missing: {root}")
    for marker in REQUIRED_MARKERS:
        _require((root / marker).is_file(), f"Missing forensic completion marker: {marker}")
    paths = {
        "development_metadata": root / "development_metadata.csv",
        "detector_features": root / "detector_features.parquet",
        "cwt_features": root / "cwt_features.parquet",
        "manifest": root / "recomputed_vsb_manifest.json",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"Missing forensic source artifact: {name}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    _require(
        manifest.get("manifest_fingerprint") == REQUIRED_SOURCE_HASHES["development_manifest_fingerprint"],
        "Development manifest fingerprint mismatch",
    )
    metadata = pd.read_csv(paths["development_metadata"], dtype={"sample_id": str, "id_measurement": str, "phase": str})
    required_metadata = {"sample_id", "id_measurement", "phase", "target", "split", "oof_fold"}
    _require(required_metadata.issubset(metadata.columns), "Development metadata schema is incomplete")
    _require(len(metadata) == 6972 and metadata["sample_id"].nunique() == 6972, "Development metadata identity mismatch")
    _require(set(metadata["split"].astype(str)) <= {"train", "validation"}, "A holdout row is present in modality inputs")
    _require(metadata.groupby("id_measurement").size().eq(3).all(), "Development measurements do not have three rows")
    _require(metadata.groupby("id_measurement")["phase"].nunique().eq(3).all(), "Development phases are incomplete")
    _require(set(metadata["phase"].astype(str)) == {"0", "1", "2"}, "Unexpected phase IDs")
    detector = pd.read_parquet(paths["detector_features"])
    cwt = pd.read_parquet(paths["cwt_features"])
    _require(detector["sample_id"].nunique() == 6972, "Detector artifact does not cover all development signals")
    _require(cwt["sample_id"].nunique() == 6972 and not cwt["sample_id"].duplicated().any(), "CWT identity mismatch")
    frame, canonical_order = prepare_forensic_baseline_frame(root)
    _require(set(frame["sample_id"].astype(str)) == set(metadata["sample_id"].astype(str)), "Feature/metadata IDs differ")
    metadata_key = metadata.set_index("sample_id")[["id_measurement", "phase", "target", "split"]].astype(str)
    frame_key = frame.set_index(frame["sample_id"].astype(str))[["id_measurement", "phase", "target", "split"]].astype(str)
    _require(metadata_key.sort_index().equals(frame_key.sort_index()), "Feature metadata does not match development manifest")
    strict = [column for column in canonical_order if column.startswith("dualcycon_strict_")]
    strict_missing_count = int(detector.groupby("sample_id", sort=False)[strict].first().isna().all(axis=1).sum())
    _require(strict_missing_count == 68, f"Expected 68 strict-detector structural failures, found {strict_missing_count}")
    _require(len(strict) == 36, "Unexpected strict-detector feature count")
    _require(len(frame) == 6972 and len(canonical_order) == 153, "Canonical frame dimensions are not 6972 x 153")
    config_hash = None
    if config and config.get("source_artifacts", {}).get("forensic_config"):
        config_path = Path(config["source_artifacts"]["forensic_config"])
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_hash = file_fingerprint(config_path)["sha256"] if config_path.is_file() else None
    source_hashes = {
        "audit_config_sha256": config_hash or REQUIRED_SOURCE_HASHES["audit_config_sha256"],
        "development_manifest_fingerprint": manifest["manifest_fingerprint"],
        "development_metadata_sha256": file_fingerprint(paths["development_metadata"])["sha256"],
        "detector_features_sha256": file_fingerprint(paths["detector_features"])["sha256"],
        "cwt_features_sha256": file_fingerprint(paths["cwt_features"])["sha256"],
    }
    _require(source_hashes == REQUIRED_SOURCE_HASHES, f"Source hash lock mismatch: {source_hashes}")
    return {
        "audit_root": str(root),
        "markers": {marker: True for marker in REQUIRED_MARKERS},
        "source_hashes": source_hashes,
        "rows": int(len(frame)),
        "measurements": int(frame["id_measurement"].nunique()),
        "train_rows": int((frame["split"] == "train").sum()),
        "validation_rows": int((frame["split"] == "validation").sum()),
        "positive_rows": int(frame["target"].sum()),
        "canonical_feature_count": int(len(canonical_order)),
        "structural_zero_rows": strict_missing_count,
        "holdout_rows": 0,
        "alignment_status": "VALID",
        "data_contract": "PD_RAW_DATA_ROOT/data/raw; forensic artifacts only; no researchdata or Dropbox",
    }


def build_modality_feature_inventory(canonical_feature_order: Sequence[str]) -> pd.DataFrame:
    """Assign every canonical predictor exactly once to S, T, or C."""

    canonical = list(canonical_feature_order)
    if len(canonical) != 153 or len(set(canonical)) != len(canonical):
        raise ModalityDiagnosticError("Canonical feature order must contain 153 unique predictors")
    rows: list[dict[str, str | int]] = []
    for position, column in enumerate(canonical):
        family: str | None = None
        source = "detector_features.parquet"
        rationale = ""
        if column == CWT_EVENT:
            family, source, rationale = "S", "cwt_features.parquet", "CWT event availability context"
        elif column in CWT_SUMMARIES:
            family, source, rationale = "C", "cwt_features.parquet", "Bounded CWT morphology/energy summary"
        else:
            matches = [prefix for prefix in DETECTOR_PREFIXES if column.startswith(prefix + "_")]
            if len(matches) != 1:
                raise ModalityDiagnosticError(f"Unassignable canonical feature: {column}")
            prefix = matches[0]
            suffix = column[len(prefix) + 1:]
            if suffix in STRUCTURAL_SUFFIXES:
                family, rationale = "S", "Detector availability, noise, or fixed event-location context"
            elif suffix in TEMPORAL_SUFFIXES:
                family, rationale = "T", "Detector pulse strength, spacing, polarity, or width summary"
            else:
                raise ModalityDiagnosticError(f"Unassignable detector feature: {column}")
        rows.append({"position": position, "feature": column, "family": family, "source": source, "rationale": rationale})
    inventory = pd.DataFrame(rows)
    counts = inventory["family"].value_counts().to_dict()
    _require(counts == {"S": 113, "T": 32, "C": 8}, f"Unexpected S/T/C counts: {counts}")
    return inventory


def feature_set_columns(
    inventory: pd.DataFrame,
    feature_set: str,
    canonical_order: Sequence[str],
) -> list[str]:
    """Return a feature set in original forensic order."""

    allowed = {"S", "S+T", "S+C", "S+T+C"}
    _require(feature_set in allowed, f"Unsupported modality feature set: {feature_set}")
    families = {"S"} if feature_set == "S" else {"S", "T"} if feature_set == "S+T" else {"S", "C"} if feature_set == "S+C" else {"S", "T", "C"}
    family_map = inventory.set_index("feature")["family"].to_dict()
    result = [column for column in canonical_order if family_map.get(column) in families]
    expected = {"S": 113, "S+T": 145, "S+C": 121, "S+T+C": 153}[feature_set]
    _require(len(result) == expected and len(set(result)) == expected, f"Feature set {feature_set} has wrong dimensions")
    return result


def prediction_overlap_and_oracle(
    labels: np.ndarray,
    temporal_prediction: np.ndarray,
    cwt_prediction: np.ndarray,
) -> dict[str, Any]:
    """Summarize complementary hard decisions and an oracle upper bound."""

    labels = np.asarray(labels, dtype=np.int64)
    temporal_prediction = np.asarray(temporal_prediction, dtype=np.int64)
    cwt_prediction = np.asarray(cwt_prediction, dtype=np.int64)
    _require(len(labels) == len(temporal_prediction) == len(cwt_prediction) > 0, "Overlap arrays must be non-empty and aligned")
    correct_temporal = temporal_prediction == labels
    correct_cwt = cwt_prediction == labels
    rows = []
    strata = [("all", np.ones(len(labels), dtype=bool)), ("PD", labels == 1), ("NonPD", labels == 0)]
    for stratum, mask in strata:
        denominator = int(mask.sum())
        for state, state_mask in (
            ("both_correct", correct_temporal & correct_cwt),
            ("temporal_only", correct_temporal & ~correct_cwt),
            ("cwt_only", ~correct_temporal & correct_cwt),
            ("both_wrong", ~correct_temporal & ~correct_cwt),
        ):
            count = int((mask & state_mask).sum())
            rows.append({"stratum": stratum, "state": state, "count": count, "fraction": count / denominator if denominator else 0.0})
    oracle_prediction = np.where(correct_temporal, temporal_prediction, cwt_prediction)
    stronger = max(fast_mcc(labels, temporal_prediction), fast_mcc(labels, cwt_prediction))
    return {
        "rows": rows,
        "disagreement_rate": float(np.mean(temporal_prediction != cwt_prediction)),
        "oracle_mcc": fast_mcc(labels, oracle_prediction),
        "stronger_individual_mcc": stronger,
        "oracle_headroom": fast_mcc(labels, oracle_prediction) - stronger,
    }


def classify_modality_verdict(summary: dict[str, Any]) -> dict[str, str]:
    """Apply the registered independent-strength and conditional-delta rules."""

    if not all(summary.get(key, False) for key in ("integrity_ok", "regression_ok", "partition_ok", "leakage_ok")):
        verdict = "REPRESENTATION DECOMPOSITION INCONCLUSIVE"
    else:
        temporal = float(summary["temporal_mean_mcc"])
        cwt = float(summary["cwt_mean_mcc"])
        combined = float(summary["combined_mean_mcc"])
        delta_c_t = float(summary["delta_c_given_t_mean"])
        positive = int(summary.get("delta_c_given_t_positive_seeds", 0))
        negative = int(summary.get("delta_c_given_t_negative_seeds", 0))
        temporal_strong = temporal >= 0.60
        cwt_strong = cwt >= 0.60
        cwt_meaningful = cwt >= 0.50
        strongly_complementary = delta_c_t >= 0.010 and positive >= 2 and negative == 0
        weakly_complementary = 0.005 <= delta_c_t < 0.010 and positive >= 2
        redundant = abs(delta_c_t) < 0.005 or positive < 2
        harmful = delta_c_t <= -0.005 and negative >= 2
        combined_improves_both = combined > max(temporal, cwt)
        if temporal_strong and cwt_strong and combined_improves_both and strongly_complementary:
            verdict = "TEMPORAL+CWT SUPPORTED"
        elif temporal_strong and cwt_meaningful and (strongly_complementary or weakly_complementary):
            verdict = "CWT COMPLEMENTARY BUT WEAKER"
        elif temporal >= 0.50 and cwt_meaningful and redundant:
            verdict = "CWT REDUNDANT"
        elif (cwt < 0.50 or harmful or not positive >= 2) and delta_c_t <= 0.005:
            verdict = "CWT NOT SUPPORTED"
        elif temporal < 0.50 and cwt_strong:
            verdict = "TEMPORAL NOT SUPPORTED"
        else:
            verdict = "REPRESENTATION DECOMPOSITION INCONCLUSIVE"
    recommendations = {
        "TEMPORAL+CWT SUPPORTED": "V6 may proceed to a preregistered multimodal study; keep holdouts locked until protocol freeze.",
        "CWT COMPLEMENTARY BUT WEAKER": "V6 may retain CWT as a secondary modality; do not claim adaptive fusion from this diagnostic.",
        "CWT REDUNDANT": "V6 should prioritize temporal/shared context and treat CWT as optional robustness evidence.",
        "CWT NOT SUPPORTED": "V6 should not invest in CWT fusion without a new predeclared representation rationale.",
        "TEMPORAL NOT SUPPORTED": "V6 should revisit temporal representation before making a multimodal claim.",
        "REPRESENTATION DECOMPOSITION INCONCLUSIVE": "Do not launch V6; resolve the failed lock, regression, leakage, or diagnostic evidence first.",
    }
    return {"verdict": verdict, "v6_recommendation": recommendations[verdict]}


__all__ = [
    "BASELINE_REFERENCE", "CWT_SUMMARIES", "ModalityDiagnosticError",
    "build_modality_feature_inventory", "classify_modality_verdict",
    "feature_set_columns", "make_three_phase_features", "prediction_overlap_and_oracle",
    "validate_modality_source_artifacts",
]
