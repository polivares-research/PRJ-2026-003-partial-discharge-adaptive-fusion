import numpy as np
import pytest

from partial_discharge_adaptive_fusion.config import InfrastructureError, runtime_info
from partial_discharge_adaptive_fusion.modeling.train import cross_fitted_logits


def test_cross_fit_is_blocked_before_model_construction_without_cuda():
    if runtime_info().cuda_available:
        pytest.skip("This assertion covers the managed host's current CUDA stop condition.")
    with pytest.raises(InfrastructureError, match="CUDA is unavailable"):
        cross_fitted_logits(
            np.zeros((10, 1, 4), dtype=np.float32), np.array([0, 1] * 5),
            kind="temporal", folds=np.array([0, 1] * 5), seed=42, epochs=1,
        )
