"""Evaluate the locked V5 development gate from parent-signal seed records.

Training scripts write one record per candidate and development seed. This
script deliberately consumes only those records and never opens Te2 or the VSB
holdout. It is separated from training so a stopped run can be audited and
resumed without changing the gate definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.v5_protocol import (
    V5_DEVELOPMENT_SEEDS,
    V5GateDecision,
    assert_v5_holdouts_closed,
    gate_decision,
    select_v5_candidate,
)


def evaluate_gate(config_path: Path, records_path: Path) -> dict:
    config = load_experiment_config(config_path)
    assert_v5_holdouts_closed(config)
    document = json.loads(records_path.read_text(encoding="utf-8"))
    if document.get("holdouts_opened") is not False:
        raise RuntimeError("Input records do not prove that V5 holdouts stayed closed")
    decisions: dict[str, V5GateDecision] = {}
    seed_records: dict[str, list[dict]] = {}
    for candidate_id, payload in document.get("candidates", {}).items():
        rows = payload.get("seed_records", [])
        observed_seeds = tuple(sorted(int(row["seed"]) for row in rows))
        if observed_seeds != V5_DEVELOPMENT_SEEDS:
            raise RuntimeError(f"{candidate_id}: expected development seeds {V5_DEVELOPMENT_SEEDS}, got {observed_seeds}")
        seed_records[candidate_id] = rows
        decisions[candidate_id] = gate_decision(
            candidate_id, rows, compute_cost=float(payload.get("compute_cost", 0.0)),
        )
    if not decisions:
        raise RuntimeError("No candidate records were supplied")
    selected = None
    if any(decision.status == "PASS" for decision in decisions.values()):
        selected = select_v5_candidate(decisions.values()).candidate_id
    return {
        "protocol_version": "v5-pulse-aware-v1",
        "holdouts_opened": False,
        "candidate_decisions": {key: value.__dict__ for key, value in decisions.items()},
        "seed_records": seed_records,
        "selected_candidate": selected,
        "gate_status": "PASS" if selected else "STOP_OR_REVIEW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/v5-pulse-aware/development_gate.json"))
    args = parser.parse_args(argv)
    summary = evaluate_gate(args.config, args.records)
    write_json(summary, args.output)
    print(summary["gate_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
