"""Evaluate the locked V5 development gate from parent-signal seed records."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from partial_discharge_adaptive_fusion.logging_utils import configure_progress_logging
from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.v5_protocol import (
    V5_DEVELOPMENT_SEEDS, V5GateDecision, assert_v5_holdouts_closed,
    gate_decision, select_v5_candidate,
)


def evaluate_gate(config_path: Path, records_path: Path, logger) -> dict:
    started = time.perf_counter()
    logger.info(f"gate started: config={config_path}, records={records_path}")
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
        logger.info(f"candidate started: {candidate_id}, seeds={observed_seeds}")
        for row in sorted(rows, key=lambda value: int(value["seed"])):
            logger.info(f"candidate={candidate_id} seed={row['seed']} temporal_mcc={row['temporal_validation_mcc']:.4f} cwt_mcc={row['cwt_validation_mcc']:.4f}")
        seed_records[candidate_id] = rows
        decisions[candidate_id] = gate_decision(candidate_id, rows, compute_cost=float(payload.get("compute_cost", 0.0)))
        logger.info(f"candidate finished: {candidate_id}, status={decisions[candidate_id].status}")
    if not decisions:
        raise RuntimeError("No candidate records were supplied")
    selected = None
    if any(decision.status == "PASS" for decision in decisions.values()):
        selected = select_v5_candidate(decisions.values()).candidate_id
        logger.info(f"candidate selected: {selected}")
    else:
        logger.info("no candidate passed; holdouts remain closed")
    logger.info(f"gate finished in {time.perf_counter() - started:.1f}s")
    return {
        "protocol_version": "v5-pulse-aware-v1", "holdouts_opened": False,
        "candidate_decisions": {key: value.__dict__ for key, value in decisions.items()},
        "seed_records": seed_records, "selected_candidate": selected,
        "gate_status": "PASS" if selected else "STOP_OR_REVIEW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/v5-pulse-aware/development_gate.json"))
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    logger = configure_progress_logging("v5.gate", args.log_file)
    summary = evaluate_gate(args.config, args.records, logger)
    write_json(summary, args.output)
    logger.info(f"gate report written: {args.output}")
    print(summary["gate_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
