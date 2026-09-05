"""Execute selection, revised experts and signal-level fusion in order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from partial_discharge_adaptive_fusion.reporting import write_json


ROOT = Path(__file__).resolve().parents[1]
SELECTION_CONFIG = ROOT / "configs/experiments/two-dataset-vsb-window-selection-localraw.yaml"
FROZEN_CONFIG = ROOT / "configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml"


def _run(command: list[str], env: dict[str, str]) -> dict[str, object]:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=True)
    return {"command": command, "returncode": completed.returncode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("matlab", "vsb", "both"), default="both")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--io-batch-size", type=int, default=4)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--execute-notebooks", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PD_RAW_DATA_ROOT"] = str((args.raw_root or ROOT / "data/raw").resolve())
    records = []
    if not args.skip_selection:
        records.append(_run([
            sys.executable, "scripts/select_vsb_window_protocol.py",
            "--config", str(SELECTION_CONFIG), "--raw-root", env["PD_RAW_DATA_ROOT"],
            "--io-batch-size", str(args.io_batch_size),
        ], env))
    if not FROZEN_CONFIG.is_file():
        raise FileNotFoundError(
            f"Frozen v3 config is missing: {FROZEN_CONFIG}. Run selection first or remove --skip-selection."
        )
    expert_command = [
        sys.executable, "scripts/run_representation_aware_experts.py", "--dataset", args.dataset,
        "--config", str(FROZEN_CONFIG), "--raw-root", env["PD_RAW_DATA_ROOT"],
        "--io-batch-size", str(args.io_batch_size),
    ]
    if args.seeds:
        expert_command.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    records.append(_run(expert_command, env))
    fusion_command = [
        sys.executable, "scripts/run_representation_aware_fusion.py", "--dataset", args.dataset,
        "--config", str(FROZEN_CONFIG),
    ]
    if args.seeds:
        fusion_command.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    records.append(_run(fusion_command, env))
    if args.execute_notebooks:
        notebook_env = {**env, "PD_RUN_EXPERIMENT": "0"}
        for notebook in sorted((ROOT / "notebooks").glob("**/*.ipynb")):
            records.append(_run([
                "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(notebook),
            ], notebook_env))
    write_json({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(FROZEN_CONFIG), "dataset": args.dataset,
        "raw_root_environment_variable": "PD_RAW_DATA_ROOT",
        "commands": records, "notebook_execution_requested": args.execute_notebooks,
    }, ROOT / "results/manifests/representation_aware_execution.json")
    print("Representation-aware pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
