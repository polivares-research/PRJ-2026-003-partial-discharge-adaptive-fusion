from pathlib import Path

import pytest

from partial_discharge_adaptive_fusion.dataset import (
    DatasetAccessError,
    expected_raw_files,
    resolve_dataset,
)


def test_expected_local_layout_is_portable():
    assert expected_raw_files("engineering-partial-discharge-noise-signals") == [
        "dataset-pd-noise/Tr1.mat",
        "dataset-pd-noise/Va1.mat",
        "dataset-pd-noise/Te1.mat",
        "dataset-pd-noise/Te2.mat",
    ]
    assert expected_raw_files("engineering-vsb-power-line-fault-detection") == [
        "dataset-vsb-power-line-fault-detection/train.parquet",
        "dataset-vsb-power-line-fault-detection/metadata_train.csv",
    ]


def test_local_resolver_fails_with_a_clear_message_when_data_is_missing(tmp_path: Path):
    with pytest.raises(DatasetAccessError, match="Local dataset directory is missing"):
        resolve_dataset(
            "engineering-partial-discharge-noise-signals", "v1", raw_root=tmp_path,
        )


def test_local_resolver_rejects_unknown_versions(tmp_path: Path):
    (tmp_path / "dataset-pd-noise").mkdir()
    with pytest.raises(DatasetAccessError, match="Unsupported local version"):
        resolve_dataset(
            "engineering-partial-discharge-noise-signals", "wrong", raw_root=tmp_path,
        )
