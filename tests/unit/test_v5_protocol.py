from pathlib import Path

import pytest

from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.v5_protocol import (
    assert_v5_frozen, assert_v5_holdouts_closed, validate_v5_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"


def test_v5_draft_config_keeps_holdouts_closed():
    config = load_experiment_config(CONFIG)
    validate_v5_config(config)
    assert_v5_holdouts_closed(config)
    with pytest.raises(Exception):
        assert_v5_frozen(config)
