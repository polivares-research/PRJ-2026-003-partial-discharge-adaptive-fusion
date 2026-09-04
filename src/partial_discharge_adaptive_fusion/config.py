"""Portable configuration and runtime preflight helpers.

The scientific pipeline reads raw data from the repository-local ``data/raw``
layout (or the optional ``PD_RAW_DATA_ROOT`` override). Expensive neural
network stages call :func:`require_cuda` before constructing a model.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(RuntimeError):
    """Raised when a run configuration is incomplete or non-portable."""


class InfrastructureError(RuntimeError):
    """Raised when a required runtime capability is unavailable."""


@dataclass(frozen=True)
class RuntimeInfo:
    """Versions and hardware recorded with every experiment."""

    python: str
    platform: str
    torch: str
    torch_cuda: str | None
    cuda_available: bool
    gpu_name: str | None
    numpy: str | None
    pandas: str | None
    scipy: str | None
    sklearn: str | None
    pyarrow: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_root() -> Path:
    """Find the repository root from the current working directory."""

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise ConfigurationError("Run from the repository root or set PD_RAW_DATA_ROOT explicitly.")


def raw_data_root(value: str | Path | None = None) -> Path:
    """Resolve the local raw-data root without requiring an external catalog."""

    configured = value or os.environ.get("PD_RAW_DATA_ROOT")
    root = Path(configured).expanduser().resolve() if configured else project_root() / "data" / "raw"
    if not root.is_dir():
        raise ConfigurationError(f"Raw-data root does not exist: {root}")
    return root


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_info() -> RuntimeInfo:
    """Collect runtime details without starting training."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by environment checks
        raise InfrastructureError("PyTorch is required for the expert pipeline.") from exc
    return RuntimeInfo(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=bool(torch.cuda.is_available()),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        numpy=_version("numpy"),
        pandas=_version("pandas"),
        scipy=_version("scipy"),
        sklearn=_version("scikit-learn"),
        pyarrow=_version("pyarrow"),
    )


def require_cuda() -> Any:
    """Return a CUDA device or fail before any neural-network work."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InfrastructureError("PyTorch is required and must run in partial-discharge.") from exc
    if not torch.cuda.is_available():
        raise InfrastructureError(
            "CUDA is unavailable. Neural-network training is intentionally blocked; "
            "the project does not fall back to CPU."
        )
    return torch.device("cuda")


def require_frozen_input_policy(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Return a dataset input policy only when it is explicitly frozen."""

    datasets = config.get("datasets", {})
    policy = datasets.get(dataset_id, {}).get("input_policy")
    if not isinstance(policy, dict) or policy.get("status") != "frozen":
        raise ConfigurationError(
            f"Input policy for {dataset_id} is not frozen; audit and protocol freeze are required."
        )
    return policy
