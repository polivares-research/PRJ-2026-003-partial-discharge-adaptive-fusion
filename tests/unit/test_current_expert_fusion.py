from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partial_discharge_adaptive_fusion.current_expert_fusion import (
    ALL_METHODS,
    canonical_pair_frame,
    component_thresholds,
    cross_fitted_meta_oof,
    expert_gate,
    fusion_gate,
    meta_fold_assignments,
    reliability_features,
    validate_prediction_source,
)


def _source() -> pd.DataFrame:
    rows = []
    for seed in (42, 43, 44):
        for split, offset in (("train_oof", 0), ("validation", 100)):
            for index in range(30):
                group = f"m{(index + offset) // 3:03d}"
                target = int((index + seed + offset) % 4 == 0)
                base = 0.82 if target else 0.18
                for method, adjustment in (("temporal", 0.0), ("global_spectrogram", -0.08)):
                    probability = np.clip(base + adjustment + ((index + seed) % 5 - 2) * 0.01, 0.01, 0.99)
                    rows.append({
                        "sample_id": f"{split[:1]}-{seed}-{index}",
                        "id_measurement": group,
                        "phase": str(index % 3),
                        "target": target,
                        "dataset": "vsb",
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "probability": probability,
                        "threshold": 0.5,
                        "prediction": int(probability >= 0.5),
                    })
    return pd.DataFrame(rows)


def test_source_validation_and_pair_alignment():
    source = _source()
    result = validate_prediction_source(source)
    assert result["status"] == "PASS"
    pair = canonical_pair_frame(source, dataset="vsb", seed=42, split="validation")
    assert len(pair) == 30
    assert pair["sample_id"].is_unique
    assert {"probability_temporal", "probability_global_spectrogram"} <= set(pair.columns)


def test_source_rejects_holdouts_duplicates_and_nonfinite_values():
    source = _source()
    holdout = source.copy()
    holdout.loc[0, "split"] = "test_grouped_holdout"
    with pytest.raises(ValueError, match="Holdout"):
        validate_prediction_source(holdout)

    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prediction_source(duplicate)

    nonfinite = source.copy()
    nonfinite.loc[0, "probability"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_prediction_source(nonfinite)


def test_meta_folds_are_complete_and_group_isolated():
    source = _source()
    pair = canonical_pair_frame(source, dataset="vsb", seed=42, split="train_oof")
    folds = meta_fold_assignments(
        pair["target"].to_numpy(), pair["id_measurement"].to_numpy(), dataset="vsb", seed=42
    )
    assert set(folds) == set(range(5))
    assignments = pd.DataFrame({"group": pair["id_measurement"], "fold": folds})
    assert assignments.groupby("group")["fold"].nunique().max() == 1


def test_cross_fitted_fusion_is_finite_and_thresholds_are_oof_derived():
    source = _source()
    pair = canonical_pair_frame(source, dataset="vsb", seed=42, split="train_oof")
    oof, components, folds = cross_fitted_meta_oof(pair, dataset="vsb", seed=42)
    assert len(oof) == len(pair)
    assert len(folds) == len(pair)
    for method in ALL_METHODS:
        values = oof[f"probability_{method}"].to_numpy()
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
    thresholds = component_thresholds(components)
    assert set(thresholds) >= {"temporal", "global_spectrogram", "best_fixed", "adaptive", "best_individual"}
    assert all(0.05 <= value <= 0.95 for value in thresholds.values())


def test_reliability_features_are_fixed_and_finite():
    features = reliability_features(np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.6, 0.8]))
    assert features.shape == (3, 11)
    assert np.isfinite(features).all()


def test_gates_are_predeclared_and_dataset_specific():
    metrics = pd.DataFrame([
        {"dataset": "vsb", "split": "validation", "method": method, "mcc": value}
        for method, values in {
            "temporal": [0.63, 0.62, 0.65],
            "global_spectrogram": [0.56, 0.60, 0.55],
        }.items() for value in values
    ])
    assert expert_gate(metrics, dataset="vsb")["eligible"]
    summary = {
        "best_fixed_minus_best_individual": {"seed_deltas": [0.01, 0.02, 0.01], "hierarchical_ci": {"ci_95": [0.001, 0.03]}},
    }
    assert fusion_gate(summary)["eligible"]
