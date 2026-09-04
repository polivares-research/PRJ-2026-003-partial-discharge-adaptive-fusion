"""Generate non-neural split manifests from repository-local raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

from partial_discharge_adaptive_fusion.config import raw_data_root
from partial_discharge_adaptive_fusion.dataset import load_mat_partition, load_vsb_metadata, resolve_dataset
from partial_discharge_adaptive_fusion.splits import matlab_manifest, vsb_grouped_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/manifests"))
    parser.add_argument("--raw-root", type=Path, help="Optional local raw-data root.")
    args = parser.parse_args()
    root = raw_data_root(args.raw_root)
    matlab = resolve_dataset("engineering-partial-discharge-noise-signals", "v1", raw_root=root)
    partitions = {name: load_mat_partition(matlab, name) for name in ("Tr1.mat", "Va1.mat", "Te1.mat")}
    mat_manifest = matlab_manifest(
        {name: batch.label for name, batch in partitions.items()},
        dataset_id=matlab.dataset_id, dataset_version=matlab.version,
    )
    mat_manifest.write_csv(args.output / "matlab_manifest.csv")
    vsb = resolve_dataset(
        "engineering-vsb-power-line-fault-detection", "2018-kaggle-snapshot", raw_root=root,
    )
    vsb_manifest = vsb_grouped_manifest(
        load_vsb_metadata(vsb), dataset_id=vsb.dataset_id, dataset_version=vsb.version,
    )
    vsb_manifest.write_csv(args.output / "vsb_manifest.csv")
    print(f"Wrote manifests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
