"""Freeze V4 only after the development gate and cycle audit pass."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from partial_discharge_adaptive_fusion.protocol import VSB_DATASET_ID, freeze_protocol, load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.v4_protocol import assert_gate_passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v4-windowed-localraw.yaml"))
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--cycle-audit", type=Path, default=Path("results/manifests/v4_vsb_cycle_reference_audit.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    gate_path = args.gate or Path(config["outputs"]["development_gate_summary"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert_gate_passed(gate)
    cycle_audit = json.loads(args.cycle_audit.read_text(encoding="utf-8"))
    if not cycle_audit.get("native_signal_audit_completed"):
        raise ValueError("The VSB cycle/native signal audit is incomplete.")
    selected_id = str(gate["selected_candidate"])
    candidate = config["selection"]["model_candidates"][selected_id]
    draft = copy.deepcopy(config)
    draft["experts"]["temporal_kind"] = candidate["temporal_kind"]
    draft["experts"]["cwt_kind"] = candidate["cwt_kind"]
    draft["selection"]["selected_candidate"] = selected_id
    draft["selection"]["gate_record"] = gate
    draft["cycle_reference"]["audit_record"] = cycle_audit
    policy = copy.deepcopy(draft["datasets"][VSB_DATASET_ID]["input_policy"])
    policy["status"] = "frozen"
    policy["selected_candidate"] = selected_id
    policy["cycle_consistency_enabled"] = bool(cycle_audit.get("cycle_consistency_enabled", False))
    output = args.output or Path(config["outputs"]["frozen_config"])
    frozen = freeze_protocol(
        draft, vsb_input_policy=policy, audit_record=cycle_audit, output_path=output,
    )
    write_json({
        "config_version": frozen["config_version"], "selected_candidate": selected_id,
        "gate_summary": gate_path.as_posix(), "cycle_audit": args.cycle_audit.as_posix(),
        "holdouts_opened": False, "frozen_config": output.as_posix(),
    }, output.with_name("v4_protocol_freeze_record.json"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
