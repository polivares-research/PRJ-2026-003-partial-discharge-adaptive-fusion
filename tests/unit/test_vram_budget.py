from scripts.estimate_vram import estimate_all


def test_vram_budget_exposes_vsb_temporal_bottleneck_without_cuda_work():
    payload = estimate_all(batch_size=128)
    cases = {case["case"]: case for case in payload["cases"]}
    assert payload["training"] is False
    assert payload["cuda_allocation"] is False
    assert cases["vsb_temporal_native"]["largest_tensor_gib"] > 6.0
    assert cases["vsb_temporal_native"]["model_payload_plus_three_x_activation_gib"] > 70.0
    assert cases["matlab_temporal"]["model_payload_plus_three_x_activation_gib"] < 0.1
    assert cases["vsb_cwt_native_policy"]["model_payload_plus_three_x_activation_gib"] < 0.5


def test_batch4_budget_is_the_active_low_memory_variant_without_cuda_work():
    payload = estimate_all(batch_size=4)
    cases = {case["case"]: case for case in payload["cases"]}
    assert payload["training"] is False
    assert payload["cuda_allocation"] is False
    assert cases["vsb_temporal_native"]["model_payload_plus_three_x_activation_gib"] < 2.5
    assert cases["matlab_temporal"]["batch_size"] == 4
    assert cases["matlab_cwt"]["batch_size"] == 4
