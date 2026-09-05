import numpy as np

import pytest

from partial_discharge_adaptive_fusion.cache import load_cache, load_cache_verified, write_array_cache, write_cwt_cache


def test_array_cache_is_reopenable_and_records_provenance(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    record = write_array_cache(
        values, root=tmp_path, dataset_id="dataset", dataset_version="v1",
        partition="train", representation="temporal", parameters={"standardized": True},
    )
    reopened = load_cache(record)
    assert reopened.shape == (3, 4)
    assert np.array_equal(reopened, values)
    assert record.metadata["dataset_version"] == "v1"
    assert np.array_equal(load_cache_verified(record, expected_parameters={"standardized": True}), values)
    with pytest.raises(ValueError, match="Incompatible"):
        load_cache_verified(record, expected_parameters={"standardized": False})


def test_cwt_cache_has_expected_shape(tmp_path):
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    record = write_cwt_cache(
        values, root=tmp_path, dataset_id="dataset", dataset_version="v1",
        partition="train", scales=np.array([1.5, 3.0]), time_bins=4, batch_size=1,
    )
    assert load_cache(record).shape == (2, 2, 4)
