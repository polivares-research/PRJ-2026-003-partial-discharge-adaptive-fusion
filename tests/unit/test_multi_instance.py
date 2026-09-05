import numpy as np
import torch

from partial_discharge_adaptive_fusion.modeling.data import MultiInstanceDataset
from partial_discharge_adaptive_fusion.modeling.models import MultiInstanceExpert, make_expert


def test_multi_instance_dataset_keeps_one_label_per_parent_signal():
    values = np.zeros((3, 5, 1, 8), dtype=np.float32)
    dataset = MultiInstanceDataset(values, np.array([0, 1, 0]))
    bag, label = dataset[1]
    assert tuple(bag.shape) == (5, 1, 8)
    assert label.item() == 1.0


def test_k_one_matches_the_unchanged_encoder_at_signal_level():
    torch.manual_seed(42)
    base = make_expert("temporal").eval()
    bag_model = MultiInstanceExpert("temporal", aggregation="max", instance_microbatch_size=1).eval()
    bag_model.encoder.load_state_dict(base.state_dict())
    values = torch.randn(2, 1, 1, 32)
    with torch.inference_mode():
        expected = base(values[:, 0])
        observed = bag_model(values)
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)


def test_multi_instance_outputs_signal_logits_and_window_summaries():
    torch.manual_seed(7)
    model = MultiInstanceExpert("temporal", aggregation="top_k_mean", instance_microbatch_size=2).eval()
    with torch.inference_mode():
        logits, summary = model(torch.randn(3, 5, 1, 32), return_window_summary=True)
    assert tuple(logits.shape) == (3,)
    assert set(summary) == {
        "max_probability", "mean_probability", "top_k_mean_probability",
        "std_probability", "fraction_high_confidence", "top1_top2_gap",
    }
    assert all(torch.all((value >= 0) & (value <= 1)) for value in summary.values())
