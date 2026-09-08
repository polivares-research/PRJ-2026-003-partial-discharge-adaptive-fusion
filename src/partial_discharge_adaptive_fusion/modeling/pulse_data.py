"""Torch datasets for V5 signal-level pulse bags."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PulseBagDataset(Dataset):
    """Return one label and one validity mask for each original signal."""

    def __init__(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray,
        labels: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        shape = getattr(values, "shape", None)
        mask_shape = getattr(valid_mask, "shape", None)
        if shape is None or mask_shape is None or len(shape) < 4 or shape[0] != mask_shape[0]:
            raise ValueError("Pulse values must be [signals, halves, pulses, ...] and align with masks")
        if len(mask_shape) != 3 or tuple(mask_shape[:2]) != tuple(shape[:2]):
            raise ValueError("Pulse validity masks must be [signals, halves, pulses]")
        if labels is not None and len(labels) != shape[0]:
            raise ValueError("Pulse labels must align with parent signals")
        self.values = values
        self.valid_mask = valid_mask
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.indices = np.arange(shape[0], dtype=np.int64) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        values = torch.as_tensor(np.asarray(self.values[index], dtype=np.float32))
        mask = torch.as_tensor(self.valid_mask[index], dtype=torch.bool)
        if self.labels is None:
            return values, mask
        return values, mask, torch.tensor(self.labels[index], dtype=torch.float32)
