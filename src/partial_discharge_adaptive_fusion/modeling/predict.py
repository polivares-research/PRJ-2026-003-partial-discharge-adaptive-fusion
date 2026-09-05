"""Inference entry points."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import require_cuda
from .data import MultiInstanceDataset, NumpyDataset


SUMMARY_NAMES = (
    "max_probability",
    "mean_probability",
    "top_k_mean_probability",
    "std_probability",
    "fraction_high_confidence",
    "top1_top2_gap",
)


def _prediction_loader(
    values: np.ndarray,
    indices: np.ndarray | None,
    *,
    batch_size: int,
    multi_instance: bool,
) -> DataLoader:
    dataset_class = MultiInstanceDataset if multi_instance else NumpyDataset
    dataset = dataset_class(values, indices=indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def _collect_logits_and_summaries(
    model: nn.Module,
    loader: DataLoader,
    *,
    return_window_summary: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    device = require_cuda()
    model.eval()
    logits: list[np.ndarray] = []
    summaries: dict[str, list[np.ndarray]] = {name: [] for name in SUMMARY_NAMES}
    with torch.inference_mode():
        for batch in loader:
            batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
            output = model(batch_x.to(device, non_blocking=True), return_window_summary=return_window_summary)
            if return_window_summary:
                batch_logits, batch_summary = output
            else:
                batch_logits, batch_summary = output, {}
            logits.append(batch_logits.detach().cpu().numpy())
            for name in SUMMARY_NAMES:
                if name in batch_summary:
                    summaries[name].append(batch_summary[name].detach().cpu().numpy())
    result = np.concatenate(logits).astype(np.float64)
    return result, {
        name: np.concatenate(chunks).astype(np.float64)
        for name, chunks in summaries.items()
        if chunks
    }


def logits_for_array(
    model: nn.Module,
    values: np.ndarray,
    *,
    batch_size: int = 512,
    multi_instance: bool = False,
) -> np.ndarray:
    """Run bounded inference on CUDA and return CPU NumPy logits."""

    device = require_cuda()
    loader = _prediction_loader(np.asarray(values), None, batch_size=batch_size, multi_instance=multi_instance)
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            output.append(model(batch.to(device, non_blocking=True)).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def logits_for_indices(
    model: nn.Module,
    values: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 512,
    multi_instance: bool = False,
) -> np.ndarray:
    """Run inference on selected rows without materializing a memmap subset."""

    device = require_cuda()
    loader = _prediction_loader(
        values, np.asarray(indices, dtype=np.int64), batch_size=batch_size, multi_instance=multi_instance,
    )
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            output.append(model(batch.to(device, non_blocking=True)).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def logits_and_window_summaries_for_array(
    model: nn.Module,
    values: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return signal-level logits and diagnostic window statistics."""

    require_cuda()
    loader = _prediction_loader(np.asarray(values), None, batch_size=batch_size, multi_instance=True)
    return _collect_logits_and_summaries(model, loader, return_window_summary=True)


def logits_and_window_summaries_for_indices(
    model: nn.Module,
    values: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict selected parent signals while retaining one summary per signal."""

    loader = _prediction_loader(
        values, np.asarray(indices, dtype=np.int64), batch_size=batch_size, multi_instance=True,
    )
    return _collect_logits_and_summaries(model, loader, return_window_summary=True)
