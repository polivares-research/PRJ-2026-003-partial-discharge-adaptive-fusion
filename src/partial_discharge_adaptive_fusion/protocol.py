"""Configuration loading, validation, and protocol-freeze guards.

The confirmatory stage has two deliberately different states:

* a draft protocol, in which the VSB native signal audit is still allowed to
  determine whether an input policy can be defended; and
* a frozen protocol, in which all preprocessing and evaluation decisions are
  locked before opening MATLAB ``Te2`` or the VSB holdout.

This module contains the state transition but does not decide a VSB signal
length, resampling policy, or window.  Those values must be supplied by the
data-audit stage and recorded in the resulting YAML.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigurationError, require_frozen_input_policy


MATLAB_DATASET_ID = "engineering-partial-discharge-noise-signals"
VSB_DATASET_ID = "engineering-vsb-power-line-fault-detection"
EXPECTED_DATASET_VERSIONS = {
    MATLAB_DATASET_ID: "v1",
    VSB_DATASET_ID: "2018-kaggle-snapshot",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load one experiment YAML and reject an empty or malformed document."""

    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(f"Experiment configuration does not exist: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ConfigurationError(f"Experiment configuration must be a mapping: {source}")
    validate_experiment_config(document)
    return document


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate invariant parts of the confirmatory configuration."""

    required = {"config_version", "datasets", "splits", "seeds", "experts", "fusion", "evaluation", "protection"}
    missing = required - set(config)
    if missing:
        raise ConfigurationError(f"Experiment configuration missing keys: {sorted(missing)}")
    data_source = config.get("data_source")
    if data_source is not None:
        if not isinstance(data_source, dict) or data_source.get("type") != "local_raw":
            raise ConfigurationError("data_source must declare type 'local_raw' when present.")
        if data_source.get("root_environment_variable") != "PD_RAW_DATA_ROOT":
            raise ConfigurationError("local_raw configurations must use PD_RAW_DATA_ROOT.")
        if data_source.get("default_relative_root") != "data/raw":
            raise ConfigurationError("local_raw configurations must default to data/raw.")
    datasets = config["datasets"]
    for dataset_id, version in EXPECTED_DATASET_VERSIONS.items():
        if dataset_id not in datasets:
            raise ConfigurationError(f"Missing canonical dataset in configuration: {dataset_id}")
        observed_version = datasets[dataset_id].get("version")
        if observed_version != version:
            raise ConfigurationError(
                f"Unexpected version for {dataset_id}: {observed_version!r}; expected {version!r}"
            )
    if config["seeds"] != [42, 43, 44, 45, 46]:
        raise ConfigurationError("The confirmatory protocol requires seeds [42, 43, 44, 45, 46].")
    if config["splits"].get("oof_folds") != 5:
        raise ConfigurationError("The confirmatory protocol requires five OOF folds.")
    experts = config["experts"]
    batch_size = experts.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigurationError("Expert batch_size must be a positive integer.")
    inference_batch_size = experts.get("inference_batch_size")
    if inference_batch_size is not None and (
        not isinstance(inference_batch_size, int) or inference_batch_size < 1
    ):
        raise ConfigurationError("Expert inference_batch_size must be a positive integer when present.")
    accumulation = experts.get("gradient_accumulation_steps", 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ConfigurationError("gradient_accumulation_steps must be a positive integer.")
    protection = config["protection"]
    if protection.get("cuda_required_for_training") is not True:
        raise ConfigurationError("CUDA-only training protection must remain enabled.")
    if protection.get("cpu_fallback") not in (None, "forbidden"):
        raise ConfigurationError("CPU fallback is not permitted by the protocol.")
    if protection.get("test_tuning") is not True and protection.get("test_tuning") != "forbidden":
        raise ConfigurationError("Test tuning protection must be explicit.")


def assert_protocol_frozen(config: dict[str, Any]) -> None:
    """Allow confirmatory evaluation only after the complete protocol is frozen."""

    validate_experiment_config(config)
    if config.get("protocol_status") != "frozen":
        raise ConfigurationError("The confirmatory protocol is not frozen.")
    require_frozen_input_policy(config, VSB_DATASET_ID)
    require_frozen_input_policy(config, MATLAB_DATASET_ID)
    if config["protection"].get("te2_before_protocol_freeze") != "forbidden":
        raise ConfigurationError("Te2 protection is missing from the frozen configuration.")


def freeze_protocol(
    config: dict[str, Any],
    *,
    vsb_input_policy: dict[str, Any],
    audit_record: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a new frozen protocol after an explicit VSB audit decision.

    The function never mutates the input mapping.  It requires the caller to
    provide a policy already marked ``frozen`` and refuses to infer any signal
    transformation from test results.
    """

    validate_experiment_config(config)
    if vsb_input_policy.get("status") != "frozen":
        raise ConfigurationError("VSB input policy must be marked frozen before protocol freeze.")
    native_audit_completed = audit_record.get("native_signal_audit_completed") or audit_record.get(
        "native_signal_sample", {}
    ).get("native_signal_audit_completed")
    if not native_audit_completed:
        raise ConfigurationError("A completed native VSB signal audit is required before freeze.")
    frozen = deepcopy(config)
    frozen["protocol_status"] = "frozen"
    frozen["datasets"][VSB_DATASET_ID]["input_policy"] = deepcopy(vsb_input_policy)
    frozen["datasets"][MATLAB_DATASET_ID]["input_policy"]["status"] = "frozen"
    schema = audit_record.get("parquet_schema", {})
    frozen["freeze_record"] = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_record_summary": {
            "dataset": deepcopy(audit_record.get("dataset")),
            "metadata_audit": deepcopy(audit_record.get("metadata_audit")),
            "native_signal_sample": deepcopy(audit_record.get("native_signal_sample")),
            "parquet_schema_summary": {
                key: deepcopy(schema.get(key))
                for key in ("num_rows", "num_columns", "num_row_groups")
                if key in schema
            },
        },
    }
    validate_experiment_config(frozen)
    assert_protocol_frozen(frozen)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    return frozen
