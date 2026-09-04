"""Portable serialization for frozen predictions, metrics, and provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def write_json(payload: Any, path: str | Path) -> Path:
    """Write JSON with NumPy scalar/array support."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return destination


def write_yaml(payload: Any, path: str | Path) -> Path:
    """Write a human-readable configuration or summary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def write_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write aligned sample-level predictions to Parquet."""

    required = {"sample_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Prediction frame contains duplicate sample IDs.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(destination, index=False)
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to serialize prediction Parquet files.") from exc
    return destination


def prediction_frame(
    sample_id: np.ndarray,
    labels: np.ndarray,
    *,
    dataset_id: str,
    split: str,
    seed: int,
    config_version: str | None = None,
    **probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build one aligned prediction table for multiple methods."""

    sample_id = np.asarray(sample_id).astype(str)
    labels = np.asarray(labels, dtype=np.int64)
    frame = pd.DataFrame({
        "sample_id": sample_id, "label": labels, "dataset_id": dataset_id,
        "split": split, "seed": int(seed),
    })
    if config_version is not None:
        frame["config_version"] = str(config_version)
    for method, values in probabilities.items():
        values = np.asarray(values, dtype=np.float64)
        if len(values) != len(frame):
            raise ValueError(f"Probability length mismatch for {method}.")
        frame[f"probability_{method}"] = values
    return frame


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")
