"""V4 development gates and deterministic candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateDecision:
    """Aggregate VSB development decision for one candidate."""

    candidate_id: str
    mean_best_individual_mcc: float
    minimum_best_individual_mcc: float
    minimum_expert_mcc: float
    status: str
    compute_cost: float


def gate_decision(
    candidate_id: str,
    records: list[dict[str, Any]],
    *,
    compute_cost: float = 0.0,
) -> GateDecision:
    """Apply the frozen V4 gate to validation-only seed records."""

    if not records:
        raise ValueError("At least one development seed record is required.")
    best = [float(row["best_individual_mcc"]) for row in records]
    expert = [float(row["minimum_expert_mcc"]) for row in records]
    mean_best = sum(best) / len(best)
    minimum_best = min(best)
    minimum_expert = min(expert)
    if mean_best <= 0.60:
        status = "STOP"
    elif mean_best < 0.70 or minimum_best <= 0.60:
        status = "WEAK_DO_NOT_FREEZE"
    else:
        status = "PASS"
    return GateDecision(
        candidate_id=candidate_id, mean_best_individual_mcc=mean_best,
        minimum_best_individual_mcc=minimum_best, minimum_expert_mcc=minimum_expert,
        status=status, compute_cost=float(compute_cost),
    )


def select_v4_candidate(decisions: list[GateDecision]) -> GateDecision:
    """Select the best passing candidate using the registered tie-breaks."""

    passing = [item for item in decisions if item.status == "PASS"]
    if not passing:
        raise ValueError("No V4 candidate passed the development expert gate.")
    return max(
        passing,
        key=lambda item: (
            item.mean_best_individual_mcc,
            item.minimum_expert_mcc,
            -item.compute_cost,
        ),
    )


def assert_gate_passed(summary: dict[str, Any]) -> None:
    """Reject protocol freeze unless the selected candidate passed."""

    selected = summary.get("selected_candidate")
    decisions = summary.get("candidate_decisions", {})
    if not selected or decisions.get(selected, {}).get("status") != "PASS":
        raise ValueError("The V4 development gate has not passed; holdouts remain blocked.")
