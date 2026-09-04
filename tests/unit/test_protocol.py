from pathlib import Path

import pytest

from partial_discharge_adaptive_fusion.config import ConfigurationError
from partial_discharge_adaptive_fusion.protocol import (
    MATLAB_DATASET_ID,
    VSB_DATASET_ID,
    assert_protocol_frozen,
    load_experiment_config,
)


def test_draft_protocol_keeps_vsb_unresolved_and_rejects_confirmatory_use():
    config = load_experiment_config(Path("configs/experiments/two-dataset-confirmatory-v1.yaml"))
    assert config["datasets"][VSB_DATASET_ID]["input_policy"]["status"] == "unresolved"
    with pytest.raises(ConfigurationError, match="not frozen"):
        assert_protocol_frozen(config)


def test_canonical_dataset_constants_are_the_configured_ids():
    config = load_experiment_config(Path("configs/experiments/two-dataset-confirmatory-v1.yaml"))
    assert set((MATLAB_DATASET_ID, VSB_DATASET_ID)) == set(config["datasets"])


def test_batch4_amendment_is_frozen_and_explicit():
    config = load_experiment_config(Path("configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml"))
    assert_protocol_frozen(config)
    assert config["experts"]["batch_size"] == 4
    assert config["experts"]["inference_batch_size"] == 4
    assert config["experts"]["gradient_accumulation_steps"] == 1
    assert config["data_source"] == {
        "type": "local_raw",
        "root_environment_variable": "PD_RAW_DATA_ROOT",
        "default_relative_root": "data/raw",
    }
