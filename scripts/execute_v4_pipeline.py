"""Execute the gated V4 pipeline with stage-level timing and provenance."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from partial_discharge_adaptive_fusion.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/two-dataset-confirmatory-v4-windowed-localraw.yaml"
FROZEN_CONFIG = ROOT / "configs/experiments/two-dataset-confirmatory-v4-windowed-localraw-frozen.yaml"
AUDIT = ROOT / "results/manifests/v4_vsb_cycle_reference_audit.json"


def run_stage(name: str, command: list[str], env: dict[str, str]) -> dict[str, object]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=True)
    return {
        "name": name, "command": command, "returncode": completed.returncode,
        "started_at_utc": started_at, "duration_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--dataset", choices=("matlab", "vsb", "both"), default="both")
    args = parser.parse_args(argv)
    raw_root = (args.raw_root or ROOT / "data/raw").resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PD_RAW_DATA_ROOT"] = str(raw_root)
    python = sys.executable
    stages = []
    stages.append(run_stage("cycle_reference_audit", [
        python, "scripts/audit_v4_vsb_cycle_reference.py", "--raw-root", str(raw_root),
        "--output", str(AUDIT),
    ], env))
    stages.append(run_stage("development_gate", [
        python, "scripts/run_v4_development_gate.py", "--config", str(CONFIG),
        "--raw-root", str(raw_root), "--io-batch-size", str(args.io_batch_size),
    ], env))
    stages.append(run_stage("protocol_freeze", [
        python, "scripts/freeze_v4_protocol.py", "--config", str(CONFIG),
        "--cycle-audit", str(AUDIT), "--output", str(FROZEN_CONFIG),
    ], env))
    expert = [
        python, "scripts/run_representation_aware_experts.py", "--dataset", args.dataset,
        "--config", str(FROZEN_CONFIG), "--raw-root", str(raw_root),
        "--io-batch-size", str(args.io_batch_size),
    ]
    fusion = [
        python, "scripts/run_representation_aware_fusion.py", "--dataset", args.dataset,
        "--config", str(FROZEN_CONFIG),
    ]
    if args.seeds:
        expert.extend(["--seeds", *[str(seed) for seed in args.seeds]])
        fusion.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    stages.append(run_stage("confirmatory_experts", expert, env))
    stages.append(run_stage("signal_level_fusion_and_evaluation", fusion, env))
    write_json({
        "pipeline": "v4-windowed", "config": str(FROZEN_CONFIG),
        "raw_root_environment_variable": "PD_RAW_DATA_ROOT", "holdouts_opened_after_gate": True,
        "stages": stages, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, ROOT / "results/manifests/v4_execution.json")
    print("V4 pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
