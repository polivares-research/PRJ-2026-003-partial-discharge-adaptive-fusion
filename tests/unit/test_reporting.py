import numpy as np

from partial_discharge_adaptive_fusion.reporting import prediction_frame


def test_prediction_frame_records_config_version_for_variant_isolation():
    frame = prediction_frame(
        np.array(["s0", "s1"]),
        np.array([0, 1]),
        dataset_id="dataset",
        split="validation",
        seed=42,
        config_version="two-dataset-confirmatory-v2-batch4",
        temporal=np.array([0.1, 0.9]),
        cwt=np.array([0.2, 0.8]),
    )
    assert frame["config_version"].unique().tolist() == ["two-dataset-confirmatory-v2-batch4"]
