"""Validate the managed environment and local raw data before confirmatory runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from partial_discharge_adaptive_fusion.config import (
    ConfigurationError,
    raw_data_root,
    runtime_info,
)
from partial_discharge_adaptive_fusion.dataset import (
    DatasetAccessError,
    dataset_provenance,
    expected_raw_files,
    resolve_dataset,
)
from partial_discharge_adaptive_fusion.reporting import write_json


DATASETS = (
    ("engineering-partial-discharge-noise-signals", "v1"),
    ("engineering-vsb-power-line-fault-detection", "2018-kaggle-snapshot"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Do not require CUDA for data-only audits.")
    parser.add_argument("--raw-root", type=Path, help="Optional local raw-data root; defaults to PD_RAW_DATA_ROOT or data/raw.")
    parser.add_argument("--output", type=Path, help="Optional JSON record of the preflight result.")
    args = parser.parse_args(argv)
    info = runtime_info()
    print(json.dumps(info.as_dict(), indent=2, sort_keys=True))
    failures: list[str] = []
    resolved: list[str] = []
    if info.pyarrow is None:
        failures.append("pyarrow is unavailable; VSB Parquet audit and prediction serialization are blocked.")
    try:
        root = raw_data_root(args.raw_root)
    except Exception as exc:
        failures.append(str(exc))
        root = None
    if root is not None:
        for dataset_id, version in DATASETS:
            try:
                dataset = resolve_dataset(dataset_id, version, raw_root=root)
                missing = [
                    relative for relative in expected_raw_files(dataset_id)
                    if not (root / relative).is_file()
                ]
                if missing:
                    raise DatasetAccessError(
                        f"Missing required local files for {dataset_id}: {missing}"
                    )
                print(f"resolved {dataset.dataset_id}@{dataset.version}")
                resolved.append(f"{dataset.dataset_id}@{dataset.version}")
            except (DatasetAccessError, ConfigurationError) as exc:
                failures.append(str(exc))
    if not args.audit_only and not info.cuda_available:
        failures.append("CUDA is unavailable; neural-network training is blocked.")
    payload = {
        "runtime": info.as_dict(),
        "data_source": "local_raw",
        "raw_data_root_configured": root is not None,
        "resolved_datasets": resolved,
        "audit_only": args.audit_only,
        "dataset_provenance": [],
        "failures": failures,
        "passed": not failures,
    }
    if root is not None:
        for dataset_id, version in DATASETS:
            try:
                payload["dataset_provenance"].append(
                    dataset_provenance(resolve_dataset(dataset_id, version, raw_root=root))
                )
            except (DatasetAccessError, ConfigurationError):
                pass
    if args.output:
        write_json(payload, args.output)
    if failures:
        print("Preflight FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    print("Preflight PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
