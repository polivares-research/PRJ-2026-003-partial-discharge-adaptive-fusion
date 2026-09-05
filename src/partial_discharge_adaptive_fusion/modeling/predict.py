"""Inference entry points."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import require_cuda
from .data import NumpyDataset


def logits_for_array(model: nn.Module, values: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    """Run bounded inference on CUDA and return CPU NumPy logits."""

    device = require_cuda()
    loader = DataLoader(
        NumpyDataset(np.asarray(values)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
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
) -> np.ndarray:
    """Run inference on selected rows without materializing a memmap subset."""

    device = require_cuda()
    loader = DataLoader(
        NumpyDataset(values, indices=np.asarray(indices, dtype=np.int64)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            output.append(model(batch.to(device, non_blocking=True)).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64)
