"""Stage catalog files into the repository-local raw-data layout.

This is an optional source-machine utility.  The destination runtime does
not import ``researchdata``: it reads only the files produced under
``data/raw``.  Raw files are ignored by Git and must be transferred through a
separate approved data channel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from partial_discharge_adaptive_fusion.dataset import (
    LOCAL_DATASET_VERSIONS,
    LOCAL_FILE_MAP,
    MATLAB_DATASET_ID,
    REQUIRED_FILE_IDS,
    VSB_DATASET_ID,
)


def _catalog_get_dataset():
    try:
        module = importlib.import_module("researchdata")
        return module.get_dataset
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The optional staging utility requires the canonical researchdata "
            "package on the source machine; install/use it only there."
        ) from exc


def _selected_file_ids(dataset_id: str, *, include_official_test: bool) -> list[str]:
    file_ids = list(LOCAL_FILE_MAP[dataset_id])
    if dataset_id == MATLAB_DATASET_ID:
        return file_ids
    excluded = {
        "vsb-test-parquet", "vsb-test-parquet-zip", "vsb-metadata-test", "vsb-sample-submission",
    }
    if include_official_test:
        excluded = set()
    return [file_id for file_id in file_ids if file_id not in excluded]


def _sha256(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> str | None:
    if path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root", type=Path,
        default=Path(os.environ["PD_DATASETS_CATALOG_ROOT"]) if os.environ.get("PD_DATASETS_CATALOG_ROOT") else None,
        required=False,
        help="Catalog root used only on the source machine (or PD_DATASETS_CATALOG_ROOT).",
    )
    parser.add_argument(
        "--raw-root", type=Path,
        default=Path(os.environ.get("PD_RAW_DATA_ROOT", "data/raw")),
        help="Destination raw-data root; defaults to data/raw.",
    )
    parser.add_argument(
        "--include-official-test", action="store_true",
        help="Also stage unlabeled VSB test artifacts; excluded by default.",
    )
    parser.add_argument(
        "--inventory", type=Path, default=Path("results/manifests/raw_data_inventory.json"),
        help="Portable inventory path without absolute paths.",
    )
    args = parser.parse_args(argv)
    if args.catalog_root is None:
        parser.error("--catalog-root or PD_DATASETS_CATALOG_ROOT is required on the source machine")
    catalog_root = args.catalog_root.expanduser().resolve()
    if not catalog_root.is_dir():
        parser.error(f"Catalog root does not exist: {catalog_root}")
    raw_root = args.raw_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    get_dataset = _catalog_get_dataset()
    inventory: dict[str, Any] = {
        "data_source": "staged_from_catalog",
        "catalog_root_used": True,
        "datasets": [],
        "official_vsb_test_staged": bool(args.include_official_test),
    }
    for dataset_id, version in LOCAL_DATASET_VERSIONS.items():
        try:
            dataset = get_dataset(dataset_id, version=version, catalog_root=catalog_root)
        except Exception as exc:
            raise RuntimeError(f"Could not resolve {dataset_id}@{version}: {exc}") from exc
        dataset_record: dict[str, Any] = {
            "dataset_id": dataset_id,
            "version": version,
            "files": [],
        }
        for file_id in _selected_file_ids(
            dataset_id, include_official_test=args.include_official_test,
        ):
            source = Path(dataset.path(file_id))
            if not source.is_file():
                if file_id in REQUIRED_FILE_IDS[dataset_id]:
                    raise RuntimeError(f"Required catalog file is missing: {dataset_id}/{file_id}")
                continue
            relative = LOCAL_FILE_MAP[dataset_id][file_id]
            destination = raw_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Staging {dataset_id}/{file_id} -> {relative}", flush=True)
            shutil.copy2(source, destination)
            item: dict[str, Any] = {
                "file_id": file_id,
                "relative_path": relative,
                "size_bytes": destination.stat().st_size,
            }
            digest = _sha256(destination)
            if digest:
                item["sha256"] = digest
            dataset_record["files"].append(item)
        inventory["datasets"].append(dataset_record)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Staging complete. Inventory written to {args.inventory}")
    print("Raw files are intentionally ignored by Git; transfer them separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
