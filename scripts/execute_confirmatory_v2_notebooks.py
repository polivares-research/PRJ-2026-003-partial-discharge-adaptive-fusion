"""Execute and persist every confirmatory v2 notebook in registered order.

The caller must activate ``partial-discharge``. Raw files are resolved from
``PD_RAW_DATA_ROOT`` or the repository's ``data/raw`` directory. The script
executes notebooks in-place so their cell outputs are auditable, continues
after a notebook error, and records failures in an English summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOKS = (
    "notebooks/00-infrastructure/0.0-por-gpu-memory-budget.ipynb",
    "notebooks/01-data/1.0-por-dataset-inventory.ipynb",
    "notebooks/01-data/1.1-por-matlab-data-audit.ipynb",
    "notebooks/01-data/1.2-por-vsb-data-audit.ipynb",
    "notebooks/01-data/1.3-por-experimental-protocol.ipynb",
    "notebooks/02-experts/2.0-por-matlab-experts.ipynb",
    "notebooks/02-experts/2.1-por-vsb-experts.ipynb",
    "notebooks/02-experts/2.2-por-expert-complementarity.ipynb",
    "notebooks/03-fusion/3.0-por-static-fusion.ipynb",
    "notebooks/03-fusion/3.1-por-reliability-aware-fusion.ipynb",
    "notebooks/03-fusion/3.2-por-conservative-adaptive-fusion.ipynb",
    "notebooks/04-evaluation/4.0-por-multiseed-evaluation.ipynb",
    "notebooks/04-evaluation/4.1-por-statistical-analysis.ipynb",
    "notebooks/04-evaluation/4.2-por-cross-dataset-analysis.ipynb",
    "notebooks/05-results/5.0-por-final-results.ipynb",
    "notebooks/05-results/5.1-por-publication-figures.ipynb",
)
CONFIG_VERSION = "two-dataset-confirmatory-v2-batch4-localraw"


def _errors(notebook: nbformat.NotebookNode) -> list[str]:
    messages: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                messages.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}".strip())
    return messages


def _write_summary(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated,
        "config_version": CONFIG_VERSION,
        "environment": "partial-discharge",
        "execution_mode": "in_place_with_cell_outputs",
        "notebooks": results,
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Confirmatory v2 Notebook Execution Summary",
        "",
        f"- Generated (UTC): {generated}",
        f"- Configuration: `{CONFIG_VERSION}`",
        "- Environment: `partial-discharge`",
        "- Neural training policy: CUDA required; physical and inference batch 4; gradient accumulation 1",
        "- Notebook outputs: persisted in the corresponding notebook files",
        "",
        "## Notebook status",
        "",
        "| Notebook | Status | Duration (s) | Errors |",
        "|---|---:|---:|---|",
    ]
    for item in results:
        errors = "<br>".join(str(error) for error in item.get("errors", [])) or "—"
        lines.append(
            f"| `{item['notebook']}` | {item['status']} | {float(item['duration_seconds']):.1f} | {errors} |"
        )
    lines += [
        "",
        "## Scientific interpretation",
        "",
        "The execution log is an audit trail, not evidence of publication readiness by itself. "
        "Publication claims require complete expert predictions for all five seeds, locked MATLAB Te2 "
        "and VSB grouped-holdout outputs, paired statistics, and the pre-specified MCC criterion. "
        "Any failed or incomplete expert notebook makes the confirmatory result incomplete.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/confirmatory_v2_notebook_execution_summary.md"),
    )
    parser.add_argument("--kernel-name", default=os.environ.get("PD_KERNEL_NAME", "partial-discharge"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if Path(sys.executable).resolve().parent.parent.name != "partial-discharge":
        raise SystemExit("This runner must use the mamba environment named partial-discharge.")
    os.environ.setdefault("PD_RAW_DATA_ROOT", str(root / "data" / "raw"))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    os.environ.setdefault("PD_RUN_EXPERIMENT", "1")

    results: list[dict[str, object]] = []
    for relative in NOTEBOOKS:
        path = root / relative
        print(f"Executing {relative}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        started = time.monotonic()
        status = "passed"
        errors: list[str] = []
        try:
            client = NotebookClient(
                notebook,
                timeout=0,
                kernel_name=args.kernel_name,
                allow_errors=True,
                resources={"metadata": {"path": str(root)}},
            )
            client.execute()
            errors = _errors(notebook)
            if errors:
                status = "failed"
        except Exception as exc:  # persist partial notebook state before continuing
            status = "failed"
            errors = [f"{type(exc).__name__}: {exc}"]
        duration = time.monotonic() - started
        notebook.metadata["execution_summary"] = {
            "config_version": CONFIG_VERSION,
            "environment": "partial-discharge",
            "status": status,
        }
        nbformat.write(notebook, path)
        item = {
            "notebook": relative,
            "status": status,
            "duration_seconds": round(duration, 3),
            "errors": errors,
        }
        results.append(item)
        print(f"{status}: {relative} ({duration:.1f}s)", flush=True)

    _write_summary(results, root / args.summary)
    failed = sum(item["status"] == "failed" for item in results)
    print(f"Completed {len(results)} notebooks; failed={failed}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
