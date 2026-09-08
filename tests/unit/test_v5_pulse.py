import numpy as np
import pytest
import torch

from partial_discharge_adaptive_fusion.modeling.pulse_models import PulseBagExpert
from partial_discharge_adaptive_fusion.modeling.pulse_preprocessing import (
    IndexedPulseView,
    PulseMaskView,
    PulseRepresentationView,
    StandardizedPulseView,
    fit_pulse_standardizer,
)
from partial_discharge_adaptive_fusion.modeling.pulse_data import PulseBagDataset
from partial_discharge_adaptive_fusion.pulse import PulsePolicy, prepare_pulse_bag, pulse_coverage_report
from partial_discharge_adaptive_fusion.pulse_cache import (
    CacheContractError, CacheMetadata, open_verified_memmap, write_memmap_cache,
)
from partial_discharge_adaptive_fusion.v5_protocol import gate_decision, select_v5_candidate


def _synthetic_signal(length=8000):
    time = np.arange(length, dtype=np.float32)
    signal = np.sin(2 * np.pi * (time - 1000) / length).astype(np.float32)
    for center in (1200, 1600, 2200, 4600, 5200, 6200):
        signal[center] += 4.0
    return signal


def test_v5_pulse_bag_has_two_halves_and_signal_level_mask():
    policy = PulsePolicy(
        signal_length=8000, moving_average_samples=101, minimum_peak_distance=20,
        n_pulses_per_half=8, temporal_length=32, cwt_length=64,
    )
    bag = prepare_pulse_bag(_synthetic_signal(), policy)
    assert bag.temporal.shape == (2, 8, 32)
    assert bag.cwt_segments.shape == (2, 8, 64)
    assert bag.valid_mask.shape == (2, 8)
    assert np.all(bag.peak_indices[bag.valid_mask] >= 0)
    report = pulse_coverage_report(bag.valid_mask[None, ...])
    assert report["coverage_ok"]


def test_v5_temporal_and_cwt_experts_return_one_parent_logit():
    temporal = PulseBagExpert("temporal", cycle_consistency=True).eval()
    cwt = PulseBagExpert("cwt").eval()
    mask = torch.ones(2, 2, 4, dtype=torch.bool)
    with torch.inference_mode():
        temporal_logits = temporal(torch.randn(2, 2, 4, 32), mask)
        cwt_logits = cwt(torch.randn(2, 2, 4, 1, 8, 16), mask)
    assert temporal_logits.shape == cwt_logits.shape == (2,)
    assert torch.isfinite(temporal_logits).all()
    assert torch.isfinite(cwt_logits).all()


def test_v5_lazy_candidate_views_preserve_parent_indexing_and_masks():
    values = np.arange(4 * 2 * 8 * 1 * 16, dtype=np.float32).reshape(4, 2, 8, 1, 16)
    valid = np.ones((4, 2, 8), dtype=bool)
    valid[2, :, 4:] = False
    candidate = PulseRepresentationView(values, representation="temporal", n_pulses=4)
    masks = PulseMaskView(valid, 4)
    subset = IndexedPulseView(candidate, np.array([3, 1]))
    subset_masks = IndexedPulseView(masks, np.array([3, 1]))
    assert subset.shape == (2, 2, 4, 1, 16)
    assert np.array_equal(subset[0], candidate[3])
    assert np.array_equal(subset_masks[1], masks[1])
    standardizer = fit_pulse_standardizer(candidate, masks, np.array([0, 1]))
    normalized = StandardizedPulseView(candidate, masks, standardizer)
    dataset = PulseBagDataset(normalized, masks, np.zeros(4), np.array([0, 1]))
    batch_values, batch_mask, batch_labels = dataset[0]
    assert batch_values.shape == (2, 4, 1, 16)
    assert batch_mask.shape == (2, 4)
    assert batch_labels.item() == 0.0


def test_v5_cache_rejects_partial_and_incompatible_artifacts(tmp_path):
    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    metadata = CacheMetadata(
        dataset_id="test", dataset_version="v1", source_fingerprint="source",
        split_manifest_hash="split", pulse_policy={"n": 8}, transform={"kind": "temporal"},
        dtype="float16", preprocessing_version="test", library_versions={}, shape=values.shape,
    )
    destination = tmp_path / "values.dat"
    write_memmap_cache(destination, values, metadata)
    observed, _ = open_verified_memmap(destination, metadata)
    assert np.array_equal(observed, values)
    bad = CacheMetadata(
        dataset_id="test", dataset_version="v1", source_fingerprint="changed",
        split_manifest_hash="split", pulse_policy={"n": 8}, transform={"kind": "temporal"},
        dtype="float16", preprocessing_version="test", library_versions={}, shape=values.shape,
    )
    with pytest.raises(CacheContractError):
        open_verified_memmap(destination, bad)
    destination.with_suffix(destination.suffix + ".complete").unlink()
    with pytest.raises(CacheContractError):
        open_verified_memmap(destination, metadata)


def test_v5_gate_stop_review_pass_and_selection():
    stop = gate_decision(
        "stop", [
            {"temporal_validation_mcc": 0.61, "cwt_validation_mcc": 0.59},
            {"temporal_validation_mcc": 0.62, "cwt_validation_mcc": 0.58},
            {"temporal_validation_mcc": 0.61, "cwt_validation_mcc": 0.60},
        ], compute_cost=1,
    )
    review = gate_decision(
        "review", [
            {"temporal_validation_mcc": 0.68, "cwt_validation_mcc": 0.64},
            {"temporal_validation_mcc": 0.69, "cwt_validation_mcc": 0.65},
            {"temporal_validation_mcc": 0.70, "cwt_validation_mcc": 0.66},
        ], compute_cost=1,
    )
    passed = gate_decision(
        "pass", [
            {"temporal_validation_mcc": 0.76, "cwt_validation_mcc": 0.68},
            {"temporal_validation_mcc": 0.74, "cwt_validation_mcc": 0.67},
            {"temporal_validation_mcc": 0.75, "cwt_validation_mcc": 0.66},
        ], compute_cost=2,
    )
    cheaper = gate_decision(
        "cheaper", [
            {"temporal_validation_mcc": 0.75, "cwt_validation_mcc": 0.67},
            {"temporal_validation_mcc": 0.75, "cwt_validation_mcc": 0.67},
            {"temporal_validation_mcc": 0.75, "cwt_validation_mcc": 0.67},
        ], compute_cost=1,
    )
    assert stop.status == "STOP"
    assert review.status == "REVIEW_DO_NOT_FREEZE"
    assert passed.status == "PASS"
    assert select_v5_candidate([passed, cheaper]).candidate_id == "cheaper"
