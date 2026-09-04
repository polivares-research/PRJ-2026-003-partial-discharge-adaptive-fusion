"""Dataset adapters backed exclusively by the repository-local ``data/raw``."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import ConfigurationError, raw_data_root


MAT_FILE_IDS = {
    "Tr0.mat": "pd-noise-tr0",
    "Va0.mat": "pd-noise-va0",
    "Te0.mat": "pd-noise-te0",
    "Tr1.mat": "pd-noise-tr1",
    "Va1.mat": "pd-noise-va1",
    "Te1.mat": "pd-noise-te1",
    "Te2.mat": "pd-noise-te2",
}

MATLAB_DATASET_ID = "engineering-partial-discharge-noise-signals"
MATLAB_VERSION = "v1"
VSB_DATASET_ID = "engineering-vsb-power-line-fault-detection"
VSB_VERSION = "2018-kaggle-snapshot"

LOCAL_FILE_MAP: dict[str, dict[str, str]] = {
    MATLAB_DATASET_ID: {
        "pd-noise-tr0": "dataset-pd-noise/Tr0.mat",
        "pd-noise-va0": "dataset-pd-noise/Va0.mat",
        "pd-noise-te0": "dataset-pd-noise/Te0.mat",
        "pd-noise-tr1": "dataset-pd-noise/Tr1.mat",
        "pd-noise-va1": "dataset-pd-noise/Va1.mat",
        "pd-noise-te1": "dataset-pd-noise/Te1.mat",
        "pd-noise-te2": "dataset-pd-noise/Te2.mat",
        "pd-noise-readme": "dataset-pd-noise/readme.txt",
        "pd-noise-figshare-metadata": "dataset-pd-noise/24033225.xml",
        "pd-noise-paper": "dataset-pd-noise/1-s2.0-S095219762400232X-main.pdf",
    },
    VSB_DATASET_ID: {
        "vsb-train-parquet": "dataset-vsb-power-line-fault-detection/train.parquet",
        "vsb-test-parquet": "dataset-vsb-power-line-fault-detection/test.parquet",
        "vsb-train-parquet-zip": "dataset-vsb-power-line-fault-detection/train.parquet.zip",
        "vsb-test-parquet-zip": "dataset-vsb-power-line-fault-detection/test.parquet.zip",
        "vsb-metadata-train": "dataset-vsb-power-line-fault-detection/metadata_train.csv",
        "vsb-metadata-test": "dataset-vsb-power-line-fault-detection/metadata_test.csv",
        "vsb-sample-submission": "dataset-vsb-power-line-fault-detection/sample_submission.csv",
        "vsb-source-metadata": "dataset-vsb-power-line-fault-detection/source_metadata.yaml",
    },
}

LOCAL_DATASET_VERSIONS = {
    MATLAB_DATASET_ID: MATLAB_VERSION,
    VSB_DATASET_ID: VSB_VERSION,
}

REQUIRED_FILE_IDS = {
    MATLAB_DATASET_ID: ("pd-noise-tr1", "pd-noise-va1", "pd-noise-te1", "pd-noise-te2"),
    VSB_DATASET_ID: ("vsb-train-parquet", "vsb-metadata-train"),
}


class DatasetAccessError(RuntimeError):
    """Raised when local data resolution or schema validation fails."""


@dataclass(frozen=True)
class SignalBatch:
    """Common sample-level representation used by downstream code."""

    sample_id: np.ndarray
    group_id: np.ndarray
    signal: np.ndarray
    label: np.ndarray | None
    dataset_id: str
    dataset_version: str
    partition: str
    metadata: pd.DataFrame


@dataclass(frozen=True)
class LocalDatasetSource:
    """File-backed source implementing the small API used by the adapters."""

    root: Path
    dataset_id: str
    version: str
    file_map: dict[str, str]

    def path(self, file_id: str) -> Path:
        try:
            relative_path = self.file_map[file_id]
        except KeyError as exc:
            raise DatasetAccessError(f"Unknown local file ID: {file_id}") from exc
        return self.root / relative_path

    def available(self) -> bool:
        return (self.root / self.dataset_id_directory()).is_dir()

    def dataset_id_directory(self) -> str:
        return "dataset-pd-noise" if self.dataset_id == MATLAB_DATASET_ID else "dataset-vsb-power-line-fault-detection"

    def load(self, file_id: str) -> list[dict[str, Any]]:
        path = self.path(file_id)
        if not path.is_file():
            raise DatasetAccessError(f"Required local data file is missing: {path}")
        if path.suffix.lower() != ".csv":
            raise DatasetAccessError(f"Local tabular loading is only supported for CSV files: {path.name}")
        return pd.read_csv(path).to_dict(orient="records")


@dataclass(frozen=True)
class DatasetHandle:
    """Resolved local dataset handle plus detached provenance metadata."""

    handle: Any
    metadata: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.metadata["dataset_id"])

    @property
    def version(self) -> str:
        versions = self.metadata.get("versions", {})
        return str(versions.get("current_version"))


def resolve_dataset(
    dataset_id: str,
    version: str,
    *,
    raw_root: str | Path | None = None,
) -> DatasetHandle:
    """Resolve one dataset from the repository-local raw-data layout."""

    if dataset_id not in LOCAL_FILE_MAP:
        raise DatasetAccessError(f"Unsupported local dataset identifier: {dataset_id}")
    expected_version = LOCAL_DATASET_VERSIONS[dataset_id]
    if version != expected_version:
        raise DatasetAccessError(
            f"Unsupported local version for {dataset_id}: requested {version!r}, "
            f"expected {expected_version!r}."
        )
    root = raw_data_root(raw_root)
    handle = LocalDatasetSource(
        root=root, dataset_id=dataset_id, version=version, file_map=LOCAL_FILE_MAP[dataset_id],
    )
    if not handle.available():
        raise DatasetAccessError(
            f"Local dataset directory is missing for {dataset_id}@{version}: "
            f"{root / handle.dataset_id_directory()}"
        )
    files = [
        {
            "file_id": file_id,
            "relative_path": relative_path,
            "required": file_id in REQUIRED_FILE_IDS[dataset_id],
        }
        for file_id, relative_path in LOCAL_FILE_MAP[dataset_id].items()
    ]
    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "versions": {"current_version": version},
        "source": {"type": "local_raw", "root_environment_variable": "PD_RAW_DATA_ROOT"},
        "files": files,
    }
    if dataset_id == VSB_DATASET_ID:
        metadata["scope"] = {"temporal_scope": {"sampling_frequency": 40_000_000}}
    return DatasetHandle(handle=handle, metadata=metadata)


def expected_raw_files(dataset_id: str, *, include_optional: bool = False) -> list[str]:
    """Return portable relative paths required for a local dataset."""

    if dataset_id not in LOCAL_FILE_MAP:
        raise DatasetAccessError(f"Unsupported local dataset identifier: {dataset_id}")
    required = set(REQUIRED_FILE_IDS[dataset_id])
    return [
        relative_path
        for file_id, relative_path in LOCAL_FILE_MAP[dataset_id].items()
        if include_optional or file_id in required
    ]


def dataset_provenance(dataset: DatasetHandle) -> dict[str, Any]:
    """Return portable local-file metadata without exposing absolute paths."""

    files = []
    for record in dataset.metadata.get("files", []):
        item = dict(record)
        path = Path(dataset.handle.path(str(record["file_id"])))
        item["resolved_exists"] = path.is_file()
        if item["resolved_exists"]:
            item["size_bytes"] = path.stat().st_size
        if item["resolved_exists"] and path.stat().st_size <= 32 * 1024 * 1024:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            item["observed_sha256"] = digest
        files.append(item)
    return {
        "dataset_id": dataset.dataset_id,
        "version": dataset.version,
        "files": files,
    }


def _decode_matlab_categorical(workspace: Any, n_samples: int) -> np.ndarray:
    """Decode the observed MATLAB categorical payload (NonPD=1, PD=2)."""

    raw = np.asarray(workspace, dtype=np.uint8).reshape(-1).tobytes()
    marker = bytes([16, 0, 2, 0]) + b"PD"
    if b"NonPD" not in raw or marker not in raw:
        raise DatasetAccessError("MATLAB categorical labels were not found in the payload.")
    shape_marker = struct.pack("<II", 2, n_samples)
    candidates: list[np.ndarray] = []
    offset = 0
    while True:
        position = raw.find(shape_marker, offset)
        if position < 0:
            break
        start = position + len(shape_marker)
        codes = np.frombuffer(raw, dtype=np.uint8, count=n_samples, offset=start).copy()
        if codes.size == n_samples and np.all(np.isin(codes, [1, 2])):
            candidates.append(codes)
        offset = position + 1
    if len(candidates) != 1:
        raise DatasetAccessError(f"Expected one categorical label vector, found {len(candidates)}.")
    return (candidates[0] == 2).astype(np.int64)


def _load_mat_partition_unlocked(dataset: DatasetHandle, partition: str) -> SignalBatch:
    """Load a MATLAB partition after the caller has passed the policy guard."""

    from scipy.io import loadmat

    file_id = MAT_FILE_IDS[partition]
    table = Path(partition).stem
    path = dataset.handle.path(file_id)
    mat = loadmat(path, squeeze_me=False, struct_as_record=False, variable_names=[table, "__function_workspace__"])
    if table not in mat or "__function_workspace__" not in mat:
        raise DatasetAccessError(f"{partition}: expected MATLAB variables are missing.")
    record = mat[table][0, 0]
    signal_cells = np.asarray(record.signals, dtype=object).reshape(-1)
    signals = np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in signal_cells])
    if signals.ndim != 2 or signals.shape[1] != 400 or not np.isfinite(signals).all():
        raise DatasetAccessError(f"{partition}: unexpected signal array {signals.shape}.")
    labels = _decode_matlab_categorical(mat["__function_workspace__"], len(signals))
    metadata = pd.DataFrame({
        "sample_id": [f"{partition}:{i:08d}" for i in range(len(signals))],
        "group_id": [None] * len(signals),
        "label": labels,
        "partition": partition,
    })
    return SignalBatch(
        sample_id=metadata["sample_id"].to_numpy(), group_id=metadata["group_id"].to_numpy(object),
        signal=signals, label=labels, dataset_id=dataset.dataset_id,
        dataset_version=dataset.version, partition=partition, metadata=metadata,
    )


def load_mat_partition(dataset: DatasetHandle, partition: str) -> SignalBatch:
    """Load a development or historical MATLAB partition.

    ``Te2`` is intentionally unavailable through this audit/development entry
    point.  The separate confirmatory function below requires a frozen YAML.
    """

    if partition not in MAT_FILE_IDS:
        raise DatasetAccessError(f"Unknown MATLAB partition: {partition}")
    if partition == "Te2.mat":
        raise DatasetAccessError("Te2.mat is protected; use the post-freeze confirmatory loader.")
    return _load_mat_partition_unlocked(dataset, partition)


def load_mat_confirmatory_partition(dataset: DatasetHandle, config: dict[str, Any]) -> SignalBatch:
    """Open MATLAB ``Te2`` only with a frozen protocol and frozen policy."""

    from .config import require_frozen_input_policy

    policy = config.get("datasets", {}).get(dataset.dataset_id, {}).get("input_policy", {})
    if config.get("protocol_status") != "frozen":
        raise DatasetAccessError("Te2 is unavailable until protocol_status is 'frozen'.")
    try:
        require_frozen_input_policy(config, dataset.dataset_id)
    except ConfigurationError as exc:
        raise DatasetAccessError(str(exc)) from exc
    expected = config["datasets"][dataset.dataset_id].get("confirmatory_test")
    if expected != "Te2.mat":
        raise DatasetAccessError("Frozen configuration does not identify Te2.mat as the confirmatory test.")
    if policy.get("temporal") != "standardized_400_sample_signal":
        raise DatasetAccessError("Frozen MATLAB temporal policy does not match the inherited 400-sample input.")
    if policy.get("cwt") != "morlet_log_power_32_scales_120_time_bins":
        raise DatasetAccessError("Frozen MATLAB CWT policy does not match the inherited PoC4 representation.")
    return _load_mat_partition_unlocked(dataset, "Te2.mat")


def load_vsb_metadata(dataset: DatasetHandle, *, train: bool = True) -> pd.DataFrame:
    """Load VSB metadata using the catalog's CSV artifact."""

    file_id = "vsb-metadata-train" if train else "vsb-metadata-test"
    rows = dataset.handle.load(file_id)
    frame = pd.DataFrame(rows)
    required = {"signal_id", "id_measurement", "phase"}
    if train:
        required.add("target")
    missing = required - set(frame.columns)
    if missing:
        raise DatasetAccessError(f"VSB metadata missing columns: {sorted(missing)}")
    for column in ["signal_id", "id_measurement", "phase"]:
        frame[column] = frame[column].astype(str)
    if train:
        frame["target"] = frame["target"].astype(int)
    return frame


def audit_vsb_metadata(metadata: pd.DataFrame) -> dict[str, Any]:
    """Summarize VSB label/group integrity without reading signal values."""

    required = {"signal_id", "id_measurement", "phase"}
    if "target" in metadata.columns:
        required.add("target")
    missing = required - set(metadata.columns)
    if missing:
        raise DatasetAccessError(f"VSB metadata missing columns: {sorted(missing)}")
    if metadata["signal_id"].duplicated().any():
        raise DatasetAccessError("VSB metadata contains duplicate signal_id values.")
    group_sizes = metadata.groupby("id_measurement")["phase"].nunique()
    phase_counts = metadata.groupby("id_measurement").size()
    result: dict[str, Any] = {
        "n_signals": int(len(metadata)),
        "n_measurements": int(metadata["id_measurement"].nunique()),
        "phase_values": sorted(metadata["phase"].astype(str).unique().tolist()),
        "groups_with_three_distinct_phases": int((group_sizes == 3).sum()),
        "groups_with_unexpected_phase_count": int((group_sizes != 3).sum()),
        "groups_with_unexpected_row_count": int((phase_counts != 3).sum()),
    }
    if "target" in metadata.columns:
        mixed = metadata.groupby("id_measurement")["target"].nunique()
        result.update({
            "label_counts": {str(k): int(v) for k, v in metadata["target"].value_counts().sort_index().items()},
            "groups_with_mixed_labels": int((mixed > 1).sum()),
            "mixed_label_groups": sorted(mixed[mixed > 1].index.astype(str).tolist()),
        })
    return result


def audit_vsb_signal_sample(
    dataset: DatasetHandle,
    metadata: pd.DataFrame,
    *,
    sample_size: int = 3,
) -> dict[str, Any]:
    """Inspect native Parquet signal shape/range on a fixed metadata sample.

    This function reports observations only.  It does not choose a resampling,
    window, or input length, so the VSB policy remains unresolved until a human
    or pre-registered protocol decision records those values.
    """

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    sample = metadata.head(sample_size).copy()
    batches = list(iter_vsb_signal_batches(dataset, sample, batch_size=sample_size))
    if len(batches) != 1:
        raise DatasetAccessError("Unexpected number of batches during VSB signal audit.")
    signals = batches[0].signal
    temporal_scope = dataset.metadata.get("scope", {}).get("temporal_scope", {})
    sampling_frequency = temporal_scope.get("sampling_frequency")
    duration = signals.shape[1] / float(sampling_frequency) if sampling_frequency else None
    return {
        "native_signal_audit_completed": True,
        "sample_signal_ids": sample["signal_id"].astype(str).tolist(),
        "sample_shape": list(signals.shape),
        "native_length": int(signals.shape[1]),
        "finite": bool(np.isfinite(signals).all()),
        "minimum": float(np.min(signals)),
        "maximum": float(np.max(signals)),
        "mean": float(np.mean(signals)),
        "std": float(np.std(signals)),
        "sampling_frequency_hz": float(sampling_frequency) if sampling_frequency else None,
        "duration_seconds": float(duration) if duration else None,
        "input_policy_decision": "unresolved",
    }


def inspect_vsb_parquet_schema(dataset: DatasetHandle) -> dict[str, Any]:
    """Inspect Parquet metadata without materializing the full 3.8 GB matrix."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetAccessError("pyarrow is required for VSB Parquet schema inspection.") from exc
    parquet_path = dataset.handle.path("vsb-train-parquet")
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    return {
        "path_name": Path(parquet_path).name,
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": parquet_file.metadata.num_columns,
        "num_row_groups": parquet_file.metadata.num_row_groups,
        "columns": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
    }


def iter_vsb_signal_batches(
    dataset: DatasetHandle,
    metadata: pd.DataFrame,
    *,
    batch_size: int = 8,
) -> Iterator[SignalBatch]:
    """Yield complete VSB signals in bounded column batches."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetAccessError("pyarrow is required for VSB signal loading.") from exc
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet_file = pq.ParquetFile(dataset.handle.path("vsb-train-parquet"))
    available = set(parquet_file.schema_arrow.names)
    signal_ids = metadata["signal_id"].astype(str).tolist()
    missing = sorted(set(signal_ids) - available)
    if missing:
        raise DatasetAccessError(f"VSB signal IDs absent from Parquet schema: {missing[:5]}")
    for start in range(0, len(metadata), batch_size):
        frame = metadata.iloc[start : start + batch_size].reset_index(drop=True)
        requested_columns = frame["signal_id"].tolist()
        table = parquet_file.read(columns=requested_columns)
        if set(table.column_names) != set(requested_columns):
            raise DatasetAccessError("VSB Parquet returned a column set different from metadata signal_id.")
        table = table.select(requested_columns)
        signals = np.asarray(table.to_pandas(), dtype=np.float32).T
        if signals.shape[0] != len(frame) or signals.ndim != 2:
            raise DatasetAccessError(f"Unexpected VSB batch shape: {signals.shape}")
        yield SignalBatch(
            sample_id=frame["signal_id"].to_numpy(),
            group_id=frame["id_measurement"].to_numpy(),
            signal=signals,
            label=frame["target"].to_numpy(np.int64) if "target" in frame else None,
            dataset_id=dataset.dataset_id, dataset_version=dataset.version,
            partition="train" if "target" in frame else "test", metadata=frame,
        )
