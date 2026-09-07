"""Scientific gates and provenance guards for the V5 pulse-aware study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import ConfigurationError


V5_PROTOCOL_VERSION = "v5-pulse-aware-v1"
V5_DEVELOPMENT_SEEDS = (42, 43, 44)
V5_CONFIRMATORY_SEEDS = (42, 43, 44, 45, 46)
V5_CANDIDATE_IDS = (
    "v4_uniform_control",
    "pulse_np86_cwt32",
    "pulse_np86_cwt64",
    "pulse_np257_cwt64",
)


@dataclass(frozen=True)
class V5GateDecision:
    candidate_id: str
    status: str
    mean_best_individual_mcc: float
    mean_second_expert_mcc: float
    minimum_seed_best_individual_mcc: float
    minimum_seed_second_expert_mcc: float
    compute_cost: float
    reason: str


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_v5_config(config: dict[str, Any]) -> None:
    """Reject protocol configurations that could expose a holdout early."""

    if config.get("config_version") != "two-dataset-confirmatory-v5-pulse-aware-localraw":
        raise ConfigurationError("V5 config_version is not the registered pulse-aware protocol")
    data_source = config.get("data_source", {})
    if data_source.get("type") != "local_raw" or data_source.get("root_environment_variable") != "PD_RAW_DATA_ROOT":
        raise ConfigurationError("V5 must use local raw data through PD_RAW_DATA_ROOT")
    if data_source.get("default_relative_root") != "data/raw":
        raise ConfigurationError("V5 local raw data must default to data/raw")
    if tuple(config.get("seeds", ())) != V5_CONFIRMATORY_SEEDS:
        raise ConfigurationError("V5 confirmatory seeds must be [42, 43, 44, 45, 46]")
    if tuple(config.get("selection", {}).get("development_seeds", ())) != V5_DEVELOPMENT_SEEDS:
        raise ConfigurationError("V5 development seeds must be [42, 43, 44]")
    protection = config.get("protection", {})
    required_protection = {
        "cuda_required_for_training": True,
        "cpu_fallback": "forbidden",
        "te2_before_protocol_freeze": "forbidden",
        "vsb_holdout_before_gate": "forbidden",
        "official_unlabeled_test_access": "forbidden",
        "test_tuning": "forbidden",
        "external_vsb_weights": "forbidden",
    }
    for key, expected in required_protection.items():
        if protection.get(key) != expected:
            raise ConfigurationError(f"V5 protection {key!r} must be {expected!r}")
    candidates = config.get("selection", {}).get("model_candidates", {})
    if set(candidates) != set(V5_CANDIDATE_IDS):
        raise ConfigurationError("V5 candidate set does not match the predeclared bounded set")
    for candidate_id, candidate in candidates.items():
        if candidate.get("holdout_access") is not False:
            raise ConfigurationError(f"Candidate {candidate_id} does not explicitly block holdouts")
    if config.get("datasets", {}).get(
        "engineering-partial-discharge-noise-signals", {},
    ).get("confirmatory_test") != "Te2.mat":
        raise ConfigurationError("V5 must protect MATLAB Te2")
    if config.get("datasets", {}).get(
        "engineering-vsb-power-line-fault-detection", {},
    ).get("official_test") is None:
        raise ConfigurationError("V5 must declare the unlabeled VSB test as excluded")


def assert_v5_holdouts_closed(config: dict[str, Any]) -> None:
    validate_v5_config(config)
    if config.get("protocol_status") not in {"draft", "development_selection"}:
        raise ConfigurationError("V5 development gate requires a draft/development-selection config")


def assert_v5_frozen(config: dict[str, Any]) -> None:
    validate_v5_config(config)
    if config.get("protocol_status") != "frozen":
        raise ConfigurationError("V5 confirmatory evaluation requires a frozen config")
    for dataset_id, dataset in config.get("datasets", {}).items():
        policy = dataset.get("input_policy", {})
        if policy.get("status") != "frozen":
            raise ConfigurationError(f"Input policy for {dataset_id} is not frozen")
    if config.get("freeze_record", {}).get("holdouts_opened_before_freeze") is True:
        raise ConfigurationError("V5 freeze record reports that a holdout was opened early")


def gate_decision(
    candidate_id: str,
    records: Iterable[dict[str, float]],
    *,
    compute_cost: float,
) -> V5GateDecision:
    """Apply the locked V5 gate to validation MCC values from seeds 42--44."""

    rows = list(records)
    if not rows:
        raise ValueError("V5 gate requires at least one development seed")
    temporal = np.asarray([row["temporal_validation_mcc"] for row in rows], dtype=float)
    cwt = np.asarray([row["cwt_validation_mcc"] for row in rows], dtype=float)
    best = np.maximum(temporal, cwt)
    second = np.minimum(temporal, cwt)
    mean_best = float(best.mean())
    mean_second = float(second.mean())
    min_best = float(best.min())
    min_second = float(second.min())
    if not np.isfinite(np.r_[temporal, cwt]).all():
        status, reason = "STOP", "non-finite validation MCC"
    elif min_best <= 0.60 or mean_best < 0.60 or mean_second < 0.60:
        status, reason = "STOP", "an expert or seed failed the minimum MCC safety floor"
    elif mean_best < 0.70 or mean_second < 0.65:
        status, reason = "REVIEW_DO_NOT_FREEZE", "development MCC is below the V5 freeze gate"
    else:
        status, reason = "PASS", "best and second expert satisfy the V5 freeze gate"
    return V5GateDecision(
        candidate_id=candidate_id, status=status,
        mean_best_individual_mcc=mean_best, mean_second_expert_mcc=mean_second,
        minimum_seed_best_individual_mcc=min_best,
        minimum_seed_second_expert_mcc=min_second,
        compute_cost=float(compute_cost), reason=reason,
    )


def select_v5_candidate(decisions: Iterable[V5GateDecision]) -> V5GateDecision:
    eligible = [decision for decision in decisions if decision.status == "PASS"]
    if not eligible:
        raise ConfigurationError("No V5 candidate passed the development gate")
    return max(
        eligible,
        key=lambda decision: (
            decision.mean_best_individual_mcc,
            decision.mean_second_expert_mcc,
            -decision.compute_cost,
        ),
    )
