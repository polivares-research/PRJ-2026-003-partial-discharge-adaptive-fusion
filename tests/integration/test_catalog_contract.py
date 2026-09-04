import os

import pytest

from partial_discharge_adaptive_fusion.dataset import expected_raw_files, resolve_dataset


@pytest.mark.integration
def test_both_local_partial_discharge_datasets_resolve():
    root = os.environ.get("PD_RAW_DATA_ROOT")
    if not root:
        pytest.skip("PD_RAW_DATA_ROOT is not configured")
    matlab = resolve_dataset("engineering-partial-discharge-noise-signals", "v1", raw_root=root)
    vsb = resolve_dataset("engineering-vsb-power-line-fault-detection", "2018-kaggle-snapshot", raw_root=root)
    assert matlab.handle.available()
    assert vsb.handle.available()
    for dataset_id in (matlab.dataset_id, vsb.dataset_id):
        assert expected_raw_files(dataset_id)
