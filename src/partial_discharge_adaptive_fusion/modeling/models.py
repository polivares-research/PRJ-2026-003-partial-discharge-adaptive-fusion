"""PoC4-compatible expert architectures."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class TinyTemporalCNN(nn.Module):
    """The unchanged PoC1/PoC3 temporal expert."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x, return_embedding: bool = False):
        embedding = self.features(x).squeeze(-1)
        logits = self.classifier(embedding).squeeze(-1)
        return (logits, embedding) if return_embedding else logits


class SmallCWT2DCNN(nn.Module):
    """The fixed small CWT expert validated in PoC3/PoC4."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x, return_embedding: bool = False):
        embedding = self.features(x).flatten(1)
        logits = self.classifier(embedding).squeeze(-1)
        return (logits, embedding) if return_embedding else logits


def make_expert(kind: str) -> nn.Module:
    """Construct an expert by its frozen representation name."""

    if kind == "temporal":
        return TinyTemporalCNN()
    if kind == "cwt":
        return SmallCWT2DCNN()
    raise ValueError(f"Unknown expert kind: {kind}")


class MultiInstanceExpert(nn.Module):
    """Apply one unchanged expert to windows and pool at signal level.

    The model receives ``[batch, windows, channels, ...]`` and returns one
    bag-level logit per original signal. Window labels are never used: the
    loss is evaluated only after aggregation.
    """

    def __init__(
        self,
        kind: str,
        *,
        aggregation: str,
        top_k_fraction: float = 0.10,
        instance_microbatch_size: int = 32,
    ) -> None:
        super().__init__()
        if aggregation not in {"max", "top_k_mean"}:
            raise ValueError("aggregation must be 'max' or 'top_k_mean'")
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if instance_microbatch_size < 1:
            raise ValueError("instance_microbatch_size must be positive")
        self.kind = kind
        self.aggregation = aggregation
        self.top_k_fraction = float(top_k_fraction)
        self.instance_microbatch_size = int(instance_microbatch_size)
        self.encoder = make_expert(kind)

    def _top_k(self, n_windows: int) -> int:
        if self.aggregation == "max":
            return 1
        return max(1, int(np.ceil(self.top_k_fraction * n_windows)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_window_summary: bool = False,
    ) -> Any:
        if x.ndim < 4:
            raise ValueError(f"Expected [batch, windows, channels, ...], got {tuple(x.shape)}")
        batch_size, n_windows = x.shape[:2]
        if n_windows < 1:
            raise ValueError("Each bag must contain at least one window")
        k = self._top_k(n_windows)
        selected: torch.Tensor | None = None
        summary_chunks: list[torch.Tensor] = []
        sum_probability = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        sum_square = torch.zeros_like(sum_probability)
        high_confidence = torch.zeros_like(sum_probability)
        single_window_logit: torch.Tensor | None = None
        seen = 0
        for start in range(0, n_windows, self.instance_microbatch_size):
            stop = min(start + self.instance_microbatch_size, n_windows)
            chunk = x[:, start:stop].reshape(batch_size * (stop - start), *x.shape[2:])
            logits = self.encoder(chunk).reshape(batch_size, stop - start)
            if n_windows == 1:
                single_window_logit = logits[:, 0]
            probabilities = torch.sigmoid(logits)
            detached = probabilities.detach()
            summary_chunks.append(detached)
            sum_probability = sum_probability + detached.sum(dim=1)
            sum_square = sum_square + torch.square(detached).sum(dim=1)
            high_confidence = high_confidence + (detached >= 0.5).sum(dim=1)
            candidate = probabilities if selected is None else torch.cat((selected, probabilities), dim=1)
            selected = torch.topk(candidate, k=min(k, seen + stop - start), dim=1).values
            seen += stop - start
        if selected is None:
            raise RuntimeError("No window predictions were produced")
        aggregated = selected.mean(dim=1)
        bag_logit = (
            single_window_logit
            if single_window_logit is not None
            else torch.logit(torch.clamp(aggregated, 1e-6, 1.0 - 1e-6))
        )
        if not return_window_summary:
            return bag_logit
        all_probabilities = torch.cat(summary_chunks, dim=1)
        ordered = torch.sort(all_probabilities, dim=1).values
        top1 = ordered[:, -1]
        top2 = ordered[:, -2] if n_windows > 1 else top1
        mean = sum_probability / n_windows
        summary = {
            "max_probability": all_probabilities.max(dim=1).values,
            "mean_probability": mean,
            "top_k_mean_probability": aggregated.detach(),
            "std_probability": torch.sqrt(
                torch.clamp(sum_square / n_windows - torch.square(mean), min=0.0)
            ),
            "fraction_high_confidence": high_confidence / n_windows,
            "top1_top2_gap": top1 - top2,
        }
        return bag_logit, {key: value.detach() for key, value in summary.items()}


def make_multi_instance_expert(
    kind: str,
    *,
    aggregation: str,
    top_k_fraction: float = 0.10,
    instance_microbatch_size: int = 32,
) -> MultiInstanceExpert:
    """Construct a shared-window expert with a signal-level aggregator."""

    return MultiInstanceExpert(
        kind,
        aggregation=aggregation,
        top_k_fraction=top_k_fraction,
        instance_microbatch_size=instance_microbatch_size,
    )
