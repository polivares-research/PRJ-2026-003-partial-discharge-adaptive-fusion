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
        values = np.asarray(values)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if values.ndim < 4 or values.shape[0] != len(valid_mask):
            raise ValueError("Pulse values must be [signals, halves, pulses, ...] and align with masks")
        if valid_mask.ndim != 3 or valid_mask.shape[:2] != values.shape[:2]:
            raise ValueError("Pulse validity masks must be [signals, halves, pulses]")
        if labels is not None and len(labels) != len(values):
            raise ValueError("Pulse labels must align with parent signals")
        self.values = values
        self.valid_mask = valid_mask
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.indices = np.arange(len(values), dtype=np.int64) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        values = torch.as_tensor(np.asarray(self.values[index], dtype=np.float32))
        mask = torch.as_tensor(self.valid_mask[index], dtype=torch.bool)
        if self.labels is None:
            return values, mask
        return values, mask, torch.tensor(self.labels[index], dtype=torch.float32)
