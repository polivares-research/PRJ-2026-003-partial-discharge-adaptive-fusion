import pytest

from partial_discharge_adaptive_fusion.config import InfrastructureError, runtime_info, require_cuda


def test_runtime_records_torch_and_cuda_state():
    info = runtime_info()
    assert info.torch
    assert isinstance(info.cuda_available, bool)


def test_cuda_requirement_never_falls_back_to_cpu():
    info = runtime_info()
    if info.cuda_available:
        assert require_cuda().type == "cuda"
    else:
        with pytest.raises(InfrastructureError, match="CUDA is unavailable"):
            require_cuda()
