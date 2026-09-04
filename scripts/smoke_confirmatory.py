"""Dependency-light smoke checks for the confirmatory infrastructure.

This is not a scientific run and never trains a model or opens a test
partition. It is useful on managed environments where pytest is not installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from partial_discharge_adaptive_fusion.cache import load_cache, write_array_cache, write_cwt_cache
from partial_discharge_adaptive_fusion.config import InfrastructureError, runtime_info, require_cuda
from partial_discharge_adaptive_fusion.fusion import adaptive_probability, conservative_probability
from partial_discharge_adaptive_fusion.modeling.models import make_expert
from partial_discharge_adaptive_fusion.protocol import assert_protocol_frozen, load_experiment_config
from partial_discharge_adaptive_fusion.splits import assert_complete_oof, matlab_manifest, vsb_grouped_manifest


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        record = write_array_cache(
            values, root=root, dataset_id="dataset", dataset_version="v1",
            partition="train", representation="temporal", parameters={"standardized": True},
        )
        assert np.array_equal(load_cache(record), values)
        cwt = write_cwt_cache(
            np.arange(16, dtype=np.float32).reshape(2, 8), root=root, dataset_id="dataset",
            dataset_version="v1", partition="train", scales=np.array([1.5, 3.0]),
            time_bins=4, batch_size=1,
        )
        assert load_cache(cwt).shape == (2, 2, 4)

    _, weights = adaptive_probability(
        np.array([0.9, 0.2]), np.array([0.1, 0.8]),
        np.array([0.9, 0.2]), np.array([0.1, 0.8]),
    )
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(
        conservative_probability(
            np.array([0.4, 0.6]), np.array([0.2, 0.8]),
            np.array([0.50, 0.9]), np.array([0.52, 0.1]), 0.05,
        ), [0.4, 0.8],
    )

    matlab = matlab_manifest(
        {"Tr1.mat": np.array([0, 1] * 10), "Va1.mat": np.array([0, 1] * 3),
         "Te1.mat": np.array([0, 1] * 4)},
        dataset_id="engineering-partial-discharge-noise-signals", dataset_version="v1",
    )
    assert_complete_oof(matlab)
    rows = [
        {"signal_id": str(group * 3 + phase), "id_measurement": str(group),
         "phase": str(phase), "target": group % 2}
        for group in range(30) for phase in range(3)
    ]
    vsb = vsb_grouped_manifest(
        pd.DataFrame(rows), dataset_id="engineering-vsb-power-line-fault-detection",
        dataset_version="2018-kaggle-snapshot",
    )
    assert_complete_oof(vsb)
    assert vsb.frame.groupby("group_id")["split"].nunique().max() == 1

    draft = load_experiment_config("configs/experiments/two-dataset-confirmatory-v1.yaml")
    frozen = load_experiment_config("configs/experiments/two-dataset-confirmatory-v1-frozen.yaml")
    batch4 = load_experiment_config("configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml")
    assert_protocol_frozen(frozen)
    assert_protocol_frozen(batch4)
    assert make_expert("temporal")(torch.zeros(2, 1, 400)).shape == (2,)
    assert make_expert("cwt")(torch.zeros(2, 1, 32, 120)).shape == (2,)
    if not runtime_info().cuda_available:
        try:
            require_cuda()
        except InfrastructureError:
            pass
        else:  # pragma: no cover - only reached on an unexpected runtime state
            raise AssertionError("CUDA guard did not reject an unavailable device.")
    assert draft["datasets"]["engineering-vsb-power-line-fault-detection"]["input_policy"]["status"] == "unresolved"
    assert batch4["experts"]["batch_size"] == 4
    assert batch4["experts"]["inference_batch_size"] == 4
    assert batch4["data_source"]["type"] == "local_raw"
    print("Confirmatory smoke checks passed; no training or test evaluation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
