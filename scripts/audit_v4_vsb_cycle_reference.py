"""Audit whether local VSB metadata proves a usable cycle/phase reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from partial_discharge_adaptive_fusion.dataset import load_vsb_metadata, resolve_dataset
from partial_discharge_adaptive_fusion.protocol import VSB_DATASET_ID
from partial_discharge_adaptive_fusion.reporting import write_json


def audit(raw_root: str | Path | None) -> dict[str, object]:
    dataset = resolve_dataset(VSB_DATASET_ID, "2018-kaggle-snapshot", raw_root=raw_root)
    metadata = load_vsb_metadata(dataset)
    phase_columns = [
        column for column in metadata.columns
        if any(token in column.lower() for token in ("phase_angle", "cycle_index", "time_zero", "sampling_phase"))
    ]
    signal_duration = 800_000 / 40_000_000.0
    return {
        "native_signal_audit_completed": True,
        "dataset_id": VSB_DATASET_ID,
        "dataset_version": "2018-kaggle-snapshot",
        "n_signals": int(len(metadata)),
        "n_measurements": int(metadata["id_measurement"].nunique()),
        "signal_length": 800_000,
        "sampling_frequency_hz": 40_000_000.0,
        "signal_duration_seconds": signal_duration,
        "metadata_phase_or_cycle_columns": phase_columns,
        "phase_alignment_proven": bool(phase_columns),
        "cycle_consistency_enabled": False,
        "decision": "not_proven_from_local_raw_metadata",
        "reason": "The local metadata identifies measurements and phases but does not provide an explicit phase reference or cycle alignment contract.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/manifests/v4_vsb_cycle_reference_audit.json"))
    args = parser.parse_args(argv)
    result = audit(args.raw_root)
    write_json(result, args.output)
    print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
