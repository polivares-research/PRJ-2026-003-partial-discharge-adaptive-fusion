import numpy as np

from partial_discharge_adaptive_fusion.pulse import PulsePolicy, detect_cycle_reference


def test_v5_infers_missing_circular_crossing():
    policy = PulsePolicy(signal_length=8000, moving_average_samples=101)
    signal = np.concatenate([
        -np.ones(policy.half_length, dtype=np.float32),
        np.ones(policy.half_length, dtype=np.float32),
    ])
    reference = detect_cycle_reference(signal, policy)
    assert reference.inferred_missing_crossing
    assert reference.origin == policy.half_length
    assert reference.opposite == 0
