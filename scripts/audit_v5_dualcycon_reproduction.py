"""Audit V5's Dual-CyCon-inspired assumptions without opening holdouts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from partial_discharge_adaptive_fusion.config import raw_data_root, runtime_info
from partial_discharge_adaptive_fusion.dataset import (
    VSB_DATASET_ID,
    audit_vsb_metadata,
    audit_vsb_signal_sample,
    inspect_vsb_parquet_schema,
    load_vsb_metadata,
    resolve_dataset,
)
from partial_discharge_adaptive_fusion.logging_utils import configure_progress_logging
from partial_discharge_adaptive_fusion.protocol import load_experiment_config
from partial_discharge_adaptive_fusion.reporting import write_json
from partial_discharge_adaptive_fusion.v5_protocol import validate_v5_config


def build_audit(config_path: Path, raw_root: Path | None, logger) -> dict:
    logger.info(f"audit started: config={config_path}")
    config = load_experiment_config(config_path)
    validate_v5_config(config)
    dataset = resolve_dataset(VSB_DATASET_ID, "2018-kaggle-snapshot", raw_root=raw_data_root(raw_root))
    logger.info(f"dataset resolved: {dataset.dataset_id}@{dataset.version}")
    metadata = load_vsb_metadata(dataset)
    logger.info(f"metadata loaded: signals={len(metadata)}, measurements={metadata['id_measurement'].nunique()}")
    external_workspace = Path(config["data_source"]["unavailable_external_workspace"])
    researchdata_available = importlib.util.find_spec("researchdata") is not None
    result = {
        "audit_version": "v5-dual-cycon-audit-v1",
        "dataset": {"id": dataset.dataset_id, "version": dataset.version, "data_contract": "local_raw/PD_RAW_DATA_ROOT"},
        "metadata_audit": audit_vsb_metadata(metadata),
        "native_signal_sample": audit_vsb_signal_sample(dataset, metadata, sample_size=3),
        "parquet_schema": inspect_vsb_parquet_schema(dataset),
        "external_workspace": {
            "path": str(external_workspace), "available": external_workspace.is_dir(),
            "status": "UNRESOLVED_NOT_MOUNTED" if not external_workspace.is_dir() else "AVAILABLE",
        },
        "researchdata": {
            "available_in_partial_discharge": researchdata_available,
            "status": "UNRESOLVED_NOT_INSTALLED" if not researchdata_available else "AVAILABLE",
        },
        "literature_items": [
            {"item": "moving average 10000 and zero-crossing phase alignment", "status": "VERIFIED_FROM_PAPER", "v5_status": "ADAPTED_TO_SIGNAL_LEVEL"},
            {"item": "alpha=100 beta=1 high-pass flattening", "status": "VERIFIED_FROM_PAPER", "v5_status": "ADAPTED_TO_SIGNAL_LEVEL"},
            {"item": "Np=257, wt=128, wf=512", "status": "VERIFIED_FROM_PAPER", "v5_status": "CANDIDATE_POLICY_NOT_YET_FROZEN"},
            {"item": "public reference implementation and exact training source", "status": "UNAVAILABLE_UNTIL_PROVENANCE_LICENSE_REVIEW", "v5_status": "NO_EXTERNAL_CODE_OR_WEIGHTS_USED"},
            {"item": "measurement-level any-phase label aggregation", "status": "NOT_COMPATIBLE_WITH_V5_SIGNAL_LEVEL_UNIT", "v5_status": "LITERATURE-INSPIRED_SIGNAL-LEVEL-ADAPTATION"},
        ],
        "cycle_consistency": {"default": "disabled_until_phase_alignment_is_proven", "holdouts_opened": False},
        "runtime": runtime_info().as_dict(),
    }
    logger.info("metadata and native signal audit completed; holdouts remain closed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/v5-pulse-aware/dual_cycon_audit.json"))
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    logger = configure_progress_logging("v5.audit", args.log_file)
    audit = build_audit(args.config, args.raw_root, logger)
    write_json(audit, args.output)
    logger.info(f"audit finished: report={args.output}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
