"""Freeze V5 only after a passing development gate."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import yaml

from partial_discharge_adaptive_fusion.logging_utils import configure_progress_logging
from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.v5_protocol import (
    V5_CANDIDATE_IDS, V5GateDecision, assert_v5_holdouts_closed,
    select_v5_candidate, validate_v5_config,
)


def freeze(config_path: Path, gate_path: Path, output_path: Path, logger) -> dict:
    started = time.perf_counter()
    logger.info(f"freeze started: config={config_path}, gate={gate_path}")
    config = load_experiment_config(config_path)
    assert_v5_holdouts_closed(config)
    summary = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if summary.get("holdouts_opened") is not False:
        raise RuntimeError("The gate summary does not prove that holdouts stayed closed")
    decisions = []
    for candidate_id, payload in summary.get("candidate_decisions", {}).items():
        if candidate_id not in V5_CANDIDATE_IDS:
            raise RuntimeError(f"Unexpected candidate in gate summary: {candidate_id}")
        decisions.append(V5GateDecision(**payload))
    selected = select_v5_candidate(decisions)
    logger.info(f"freeze candidate selected: {selected.candidate_id}")
    frozen = deepcopy(config)
    frozen["protocol_status"] = "frozen"
    frozen["datasets"]["engineering-partial-discharge-noise-signals"]["input_policy"]["status"] = "frozen"
    vsb_policy = frozen["datasets"]["engineering-vsb-power-line-fault-detection"]["input_policy"]
    vsb_policy["status"] = "frozen"
    vsb_policy["selected_candidate"] = selected.candidate_id
    vsb_policy["selection_metrics"] = {
        "mean_best_individual_mcc": selected.mean_best_individual_mcc,
        "mean_second_expert_mcc": selected.mean_second_expert_mcc,
        "minimum_seed_best_individual_mcc": selected.minimum_seed_best_individual_mcc,
        "minimum_seed_second_expert_mcc": selected.minimum_seed_second_expert_mcc,
    }
    frozen["freeze_record"] = {
        "v5_protocol_version": "v5-pulse-aware-v1", "selected_candidate": selected.candidate_id,
        "holdouts_opened_before_freeze": False, "gate_summary": str(gate_path),
        "candidate_decision": selected.__dict__,
    }
    validate_v5_config(frozen)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    logger.info(f"freeze finished in {time.perf_counter() - started:.1f}s: output={output_path}")
    return frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"))
    parser.add_argument("--gate", type=Path, default=Path("reports/metrics/v5-pulse-aware/development_gate.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw-frozen.yaml"))
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    logger = configure_progress_logging("v5.freeze", args.log_file)
    freeze(args.config, args.gate, args.output, logger)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
