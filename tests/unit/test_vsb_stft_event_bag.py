import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from partial_discharge_adaptive_fusion import pulse
from partial_discharge_adaptive_fusion.pulse_impl import CycleReference, PeakDetection, PulsePolicy


def test_event_bag_discards_boundary_events_without_wrapping(monkeypatch):
    policy = PulsePolicy()
    signal = np.zeros(policy.signal_length, dtype=np.float32)
    flattened = np.arange(policy.signal_length, dtype=np.float32)
    fake = PeakDetection(
        aligned_signal=signal,
        flattened_signal=flattened,
        reference=CycleReference(0, policy.half_length, policy.half_length, np.array([1]), "rising"),
        peak_indices=(np.array([1, 1000]), np.array([policy.half_length + 1000, policy.signal_length - 1])),
        peak_amplitudes=(np.array([2, 1], dtype=np.float32), np.array([2, 1], dtype=np.float32)),
        selected_counts=(2, 2),
    )
    monkeypatch.setattr(pulse, "detect_pulses", lambda _signal, _policy: fake)
    bag = pulse.prepare_v5_stft_event_bag(signal, policy, segment_length=512, capacity=2)
    assert bag.segments.shape == (2, 2, 512)
    assert bag.valid_mask.tolist() == [[True, False], [True, False]]
    assert bag.peak_indices.tolist() == [[1000, -1], [policy.half_length + 1000, -1]]
    assert np.array_equal(bag.segments[0, 0], flattened[744:1256])
