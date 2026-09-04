"""Batch datasets that preserve memory-mapped representation stores."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """Read selected rows and cast only each batch item to float32."""

    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        self.values = values
        self.labels = None if labels is None else np.asarray(labels)
        self.indices = np.arange(len(values), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if self.labels is not None and len(self.labels) != len(values):
            raise ValueError("Labels must align with the complete values array.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Any:
        index = int(self.indices[position])
        value = np.array(self.values[index], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(value)
        if self.labels is None:
            return tensor
        return tensor, torch.tensor(float(self.labels[index]), dtype=torch.float32)
