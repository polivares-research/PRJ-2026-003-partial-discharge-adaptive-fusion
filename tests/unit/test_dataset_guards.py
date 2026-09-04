import pytest

from partial_discharge_adaptive_fusion.dataset import DatasetAccessError, load_mat_partition


def test_te2_is_closed_to_the_audit_loader():
    with pytest.raises(DatasetAccessError, match="Te2.mat is protected"):
        load_mat_partition(None, "Te2.mat")
