import numpy as np
import torch

from partial_discharge_adaptive_fusion.modeling.models import MultiInstanceExpert, make_expert
from partial_discharge_adaptive_fusion.v4_protocol import gate_decision, select_v4_candidate
from partial_discharge_adaptive_fusion.windowed import (
    load_or_fit_window_standardizer, standardizer_fingerprint,
)
from partial_discharge_adaptive_fusion.windowing import WindowSpec


def test_v4_models_return_signal_level_shapes():
    temporal = MultiInstanceExpert(
        "geometry_adapted_temporal", aggregation="top_k_mean", instance_microbatch_size=2,
    ).eval()
    cwt = MultiInstanceExpert(
        "geometry_adapted_cwt", aggregation="top_k_mean", instance_microbatch_size=2,
    ).eval()
    with torch.inference_mode():
        temporal_logits = temporal(torch.randn(2, 3, 1, 32))
        cwt_logits = cwt(torch.randn(2, 3, 1, 8, 16))
    assert tuple(temporal_logits.shape) == (2,)
    assert tuple(cwt_logits.shape) == (2,)


def test_v4_standardizer_cache_reuses_and_invalidates(tmp_path):
    raw = np.arange(32, dtype=np.float32).reshape(4, 8)
    factory = lambda: iter([raw])
    spec = WindowSpec(window_length=4, stride=4, aggregation="max")
    destination = tmp_path / "temporal.npz"
    parameters = {"version": "v4", "fit_partition": "train", "window": spec.as_dict()}
    first, hit = load_or_fit_window_standardizer(
        factory, spec, representation="temporal", destination=str(destination),
        expected_parameters=parameters,
    )
    assert not hit
    second, hit = load_or_fit_window_standardizer(
        factory, spec, representation="temporal", destination=str(destination),
        expected_parameters=parameters,
    )
    assert hit
    assert standardizer_fingerprint(first) == standardizer_fingerprint(second)
    changed, hit = load_or_fit_window_standardizer(
        factory, spec, representation="temporal", destination=str(destination),
        expected_parameters={**parameters, "changed": True},
    )
    assert not hit
    assert standardizer_fingerprint(changed) == standardizer_fingerprint(first)


def test_v4_gate_stop_weak_and_pass_states():
    stop = gate_decision("stop", [{"best_individual_mcc": 0.60, "minimum_expert_mcc": 0.59}])
    weak = gate_decision("weak", [{"best_individual_mcc": 0.65, "minimum_expert_mcc": 0.62}])
    passed = gate_decision(
        "pass", [
            {"best_individual_mcc": 0.72, "minimum_expert_mcc": 0.68},
            {"best_individual_mcc": 0.75, "minimum_expert_mcc": 0.70},
        ],
    )
    assert stop.status == "STOP"
    assert weak.status == "WEAK_DO_NOT_FREEZE"
    assert passed.status == "PASS"


def test_v4_candidate_selection_uses_mcc_then_expert_then_cost():
    low_cost = gate_decision("low", [{"best_individual_mcc": 0.75, "minimum_expert_mcc": 0.70}], compute_cost=1)
    high_cost = gate_decision("high", [{"best_individual_mcc": 0.75, "minimum_expert_mcc": 0.70}], compute_cost=2)
    assert select_v4_candidate([low_cost, high_cost]).candidate_id == "low"
