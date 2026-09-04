"""Freeze the confirmatory YAML after a completed, pre-registered VSB audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from partial_discharge_adaptive_fusion.protocol import freeze_protocol, load_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v1.yaml"))
    parser.add_argument("--vsb-policy", type=Path, required=True, help="JSON policy produced by the audit decision")
    parser.add_argument("--audit-record", type=Path, required=True, help="JSON native-signal audit record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    policy = json.loads(args.vsb_policy.read_text(encoding="utf-8"))
    audit_record = json.loads(args.audit_record.read_text(encoding="utf-8"))
    frozen = freeze_protocol(
        config, vsb_input_policy=policy, audit_record=audit_record, output_path=args.output,
    )
    print(f"Frozen protocol written to {args.output} ({frozen['config_version']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
