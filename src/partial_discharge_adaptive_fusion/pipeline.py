"""Execution gates and provenance for the confirmatory pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import InfrastructureError, require_cuda
from .protocol import (
    MATLAB_DATASET_ID, VSB_DATASET_ID, assert_development_selection_ready,
    assert_protocol_frozen, load_experiment_config,
)


def git_commit() -> str | None:
    """Return the current commit when the repository is available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_provenance(
    *,
    config_path: str | Path,
    dataset_id: str,
    split: str,
    seed: int,
    imbalance_strategy: str,
) -> dict[str, Any]:
    """Create the minimum provenance record required for every result file."""

    from .config import runtime_info

    config = load_experiment_config(config_path)
    if dataset_id not in (MATLAB_DATASET_ID, VSB_DATASET_ID):
        raise ValueError(f"Unknown confirmatory dataset: {dataset_id}")
    return {
        "config_path": str(Path(config_path)),
        "config_version": config["config_version"],
        "data_source": config.get("data_source", {"type": "historical_catalog"}),
        "dataset_id": dataset_id,
        "split": split,
        "seed": int(seed),
        "imbalance_strategy": imbalance_strategy,
        "git_commit": git_commit(),
        "runtime": runtime_info().as_dict(),
    }


def require_training_ready(config_path: str | Path, *, dataset_id: str, confirmatory: bool = False) -> dict[str, Any]:
    """Stop before model construction unless all scientific gates are satisfied."""

    config = load_experiment_config(config_path)
    if dataset_id == VSB_DATASET_ID:
        from .config import require_frozen_input_policy

        require_frozen_input_policy(config, VSB_DATASET_ID)
    if confirmatory:
        assert_protocol_frozen(config)
    try:
        require_cuda()
    except InfrastructureError:
        raise
    return config


def require_development_ready(config_path: str | Path, *, dataset_id: str) -> dict[str, Any]:
    """Gate representation selection to development data and CUDA only."""

    config = load_experiment_config(config_path)
    assert_development_selection_ready(config)
    if dataset_id not in (MATLAB_DATASET_ID, VSB_DATASET_ID):
        raise ValueError(f"Unknown confirmatory dataset: {dataset_id}")
    require_cuda()
    return config
