import numpy as np
import pandas as pd
import pytest

from partial_discharge_adaptive_fusion.evaluation import grouped_paired_bootstrap_delta
from partial_discharge_adaptive_fusion.vsb_forensics import make_three_phase_features
from partial_discharge_adaptive_fusion.vsb_modality import (
    CWT_EVENT,
    CWT_SUMMARIES,
    DETECTOR_PREFIXES,
    STRUCTURAL_SUFFIXES,
    TEMPORAL_SUFFIXES,
    BASELINE_REFERENCE,
    build_modality_feature_inventory,
    classify_modality_verdict,
    feature_set_columns,
    prediction_overlap_and_oracle,
)


def _canonical_columns():
    columns = []
    for prefix in DETECTOR_PREFIXES:
        columns.extend(f"{prefix}_{suffix}" for suffix in STRUCTURAL_SUFFIXES)
        columns.extend(f"{prefix}_{suffix}" for suffix in TEMPORAL_SUFFIXES)
    return columns + [CWT_EVENT] + list(CWT_SUMMARIES)


def test_feature_inventory_is_exclusive_and_complete():
    canonical = _canonical_columns()
    inventory = build_modality_feature_inventory(canonical)
    assert len(inventory) == 153
    assert inventory["feature"].is_unique
    assert inventory["family"].value_counts().to_dict() == {"S": 113, "T": 32, "C": 8}
    assert inventory.sort_values("position")["feature"].tolist() == canonical
    for feature_set, expected in {"S": 113, "S+T": 145, "S+C": 121, "S+T+C": 153}.items():
        assert len(feature_set_columns(inventory, feature_set, canonical)) == expected


def test_three_phase_features_preserve_mixed_signal_targets_and_phase_order():
    rows = []
    for measurement, labels in [("m1", [0, 1, 0]), ("m2", [0, 0, 0])]:
        for phase, label in enumerate(labels):
            rows.append({
                "sample_id": f"{measurement}-{phase}", "id_measurement": measurement,
                "phase": str(phase), "target": label, "split": "train", "oof_fold": 0,
                "x": float(phase + 10 * (measurement == "m2")),
            })
    frame = pd.DataFrame(rows)
    result = make_three_phase_features(frame, ["x"], phase_order=("0", "1", "2"))
    assert len(result) == 6
    assert result.loc[result["sample_id"] == "m1-1", "target"].iloc[0] == 1
    assert result.loc[result["sample_id"] == "m1-1", "phase_0_x"].iloc[0] == 0.0
    assert result.loc[result["sample_id"] == "m1-1", "phase_2_x"].iloc[0] == 2.0


def test_three_phase_features_reject_missing_phase():
    frame = pd.DataFrame({
        "sample_id": ["s0", "s1"], "id_measurement": ["m", "m"],
        "phase": ["0", "1"], "target": [0, 0], "split": ["train", "train"],
        "oof_fold": [0, 0], "x": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="does not contain exactly phases"):
        make_three_phase_features(frame, ["x"])


def test_grouped_bootstrap_is_deterministic_and_reports_group_unit():
    labels = np.repeat([0, 0, 1, 1], 3)
    prediction_a = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0])
    prediction_b = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    groups = np.repeat(["m0", "m1", "m2", "m3"], 3)
    first = grouped_paired_bootstrap_delta(labels, prediction_a, prediction_b, groups, iterations=200, seed=42)
    second = grouped_paired_bootstrap_delta(labels, prediction_a, prediction_b, groups, iterations=200, seed=42)
    assert first == second
    assert first["group_count"] == 4


def test_overlap_reports_oracle_headroom_and_strata():
    labels = np.array([0, 0, 1, 1])
    temporal = np.array([0, 1, 1, 0])
    cwt = np.array([0, 0, 0, 1])
    result = prediction_overlap_and_oracle(labels, temporal, cwt)
    assert {row["state"] for row in result["rows"]} == {"both_correct", "temporal_only", "cwt_only", "both_wrong"}
    assert 0.0 <= result["disagreement_rate"] <= 1.0
    assert result["oracle_mcc"] >= result["stronger_individual_mcc"]


def test_verdict_rules_are_registered():
    verdict = classify_modality_verdict({
        "integrity_ok": True, "regression_ok": True, "partition_ok": True, "leakage_ok": True,
        "temporal_mean_mcc": 0.65, "cwt_mean_mcc": 0.61, "combined_mean_mcc": 0.68,
        "delta_c_given_t_mean": 0.012, "delta_c_given_t_positive_seeds": 3,
        "delta_c_given_t_negative_seeds": 0,
    })
    assert verdict["verdict"] == "TEMPORAL+CWT SUPPORTED"
    assert "V6" in verdict["v6_recommendation"]


@pytest.mark.integration
def test_historical_combined_baseline_regression():
    from pathlib import Path
    from partial_discharge_adaptive_fusion.vsb_forensics import (
        evaluate_feature_baseline, prepare_forensic_baseline_frame,
    )
    root = Path(__file__).parents[2]
    audit_root = root / "results/audits/vsb-literature-forensic"
    if not (audit_root / "detector_features.parquet").is_file():
        pytest.skip("forensic artifacts are not available")
    frame, canonical = prepare_forensic_baseline_frame(audit_root)
    inventory = build_modality_feature_inventory(canonical)
    for mode, references in BASELINE_REFERENCE.items():
        columns = feature_set_columns(inventory, "S+T+C", canonical)
        for seed, reference in references.items():
            result = evaluate_feature_baseline(frame, columns, classifier="hist_gradient_boosting", seed=seed, grouped=True, mode=mode)
            assert abs(result["metrics"]["mcc"] - reference["mcc"]) <= 1e-9
            assert abs(result["threshold_from_train_oof"] - reference["threshold"]) <= 1e-12
