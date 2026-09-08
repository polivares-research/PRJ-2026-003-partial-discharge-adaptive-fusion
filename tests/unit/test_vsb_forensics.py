import numpy as np
import pandas as pd

from partial_discharge_adaptive_fusion.vsb_forensics import (
    DetectorOutput,
    canonical_fingerprint,
    cwt_log_power_summary,
    detector_match,
    morphology_table,
    phase_pattern_table,
)


def test_forensic_fingerprint_is_deterministic():
    assert canonical_fingerprint({"b": 2, "a": 1}) == canonical_fingerprint({"a": 1, "b": 2})


def test_phase_pattern_table_preserves_all_three_phase_patterns():
    metadata = pd.DataFrame({
        "id_measurement": ["m1", "m1", "m1", "m2", "m2", "m2"],
        "phase": [0, 1, 2, 0, 1, 2],
        "target": [0, 0, 1, 1, 1, 1],
    })
    result = phase_pattern_table(metadata)
    assert int(result.loc[result["pattern"] == "001", "measurements"].iloc[0]) == 1
    assert int(result.loc[result["pattern"] == "111", "measurements"].iloc[0]) == 1


def _output(peaks):
    peaks = np.asarray(peaks, dtype=np.int64)
    return DetectorOutput(
        detector_id="test", peaks=peaks, scores=np.ones(len(peaks), dtype=np.float32),
        signs=np.ones(len(peaks), dtype=np.int8), raw_candidate_count=len(peaks),
        noise_scale=1.0, origin=-1, alignment_status="VALID", boundary_count=0,
        transformed=np.zeros(128, dtype=np.float32),
    )


def test_detector_matching_uses_fixed_tolerance():
    result = detector_match(_output([10, 100]), _output([12, 150]), tolerance=3)
    assert result["matched_count"] == 1
    assert result["median_abs_offset_samples"] == 2


def test_cwt_summary_has_finite_compact_features():
    values = np.sin(np.linspace(0, 8 * np.pi, 512)).astype(np.float32)
    summary = cwt_log_power_summary(values, scales=np.geomspace(1.5, 64.0, 32), time_bins=128)
    assert len(summary) >= 7
    assert all(np.isfinite(list(summary.values())))


def test_morphology_table_preserves_parent_and_signal_metadata():
    waveforms = np.vstack([
        np.sin(np.linspace(0, np.pi, 32)),
        np.cos(np.linspace(0, np.pi, 32)),
        np.sin(np.linspace(0, 2 * np.pi, 32)),
    ]).astype(np.float32)
    metadata = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3"], "id_measurement": ["m1", "m1", "m2"],
        "phase": ["0", "1", "0"], "target": [0, 1, 0], "split": ["train"] * 3,
    })
    assignments, summary = morphology_table(waveforms, metadata, clusters=2)
    assert len(assignments) == 3
    assert set(assignments["sample_id"]) == {"s1", "s2", "s3"}
    assert summary["n_clusters"] == 2
