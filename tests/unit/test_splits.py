import numpy as np
import pandas as pd

from partial_discharge_adaptive_fusion.splits import (
    assert_complete_oof,
    matlab_manifest,
    vsb_grouped_manifest,
)


def test_matlab_manifest_preserves_source_partitions_and_oof_coverage():
    manifest = matlab_manifest(
        {"Tr1.mat": np.array([0, 1] * 10), "Va1.mat": np.array([0, 1] * 3), "Te1.mat": np.array([0, 1] * 4)},
        dataset_id="engineering-partial-discharge-noise-signals", dataset_version="v1",
    )
    manifest.validate()
    assert set(manifest.frame.loc[manifest.frame["split"] == "train", "oof_fold"]) == {0, 1, 2, 3, 4}
    assert_complete_oof(manifest)
    assert set(manifest.frame["split"]) == {"train", "validation", "test_historical"}


def test_vsb_manifest_keeps_measurements_together():
    rows = []
    for measurement in range(30):
        target = measurement % 2
        for phase in range(3):
            rows.append({
                "signal_id": str(measurement * 3 + phase),
                "id_measurement": str(measurement), "phase": str(phase), "target": target,
            })
    metadata = pd.DataFrame(rows)
    manifest = vsb_grouped_manifest(
        metadata, dataset_id="engineering-vsb-power-line-fault-detection",
        dataset_version="2018-kaggle-snapshot", split_seed=42,
    )
    manifest.validate()
    assert_complete_oof(manifest)
    split_by_group = manifest.frame.groupby("group_id")["split"].nunique()
    assert split_by_group.max() == 1
    assert set(manifest.frame["split"]) == {"train", "validation", "test"}
