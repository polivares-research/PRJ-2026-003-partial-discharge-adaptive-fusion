"""Pulse-bag temporal and CWT experts for the V5 VSB protocol."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _Residual1D(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.block(values))


class PulseTemporalEncoder(nn.Module):
    """Shared local temporal encoder with short and dilated context."""

    def __init__(self, embedding_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            _Residual1D(32, 1), _Residual1D(32, 2), _Residual1D(32, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(nn.Linear(32, embedding_size), nn.GELU())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(values).squeeze(-1))


class _Residual2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.block(values))


class PulseCWTEncoder(nn.Module):
    """Compact CWT encoder independent of the number of scale bins."""

    def __init__(self, embedding_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.GELU(), nn.MaxPool2d(2),
            _Residual2D(24), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.GELU(),
            _Residual2D(48), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Linear(48, embedding_size), nn.GELU())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(values).flatten(1))


def _masked_pool(
    logits: torch.Tensor,
    mask: torch.Tensor,
    aggregation: str,
    top_k_fraction: float,
) -> torch.Tensor:
    """Pool pulse probabilities within each half while ignoring bag padding."""

    probabilities = torch.sigmoid(logits)
    mask = mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Every half-cycle must contain at least one valid pulse")
    if aggregation == "max":
        return probabilities.masked_fill(~mask, -torch.inf).max(dim=-1).values
    if aggregation != "top_k_mean":
        raise ValueError("aggregation must be 'max' or 'top_k_mean'")
    k = max(1, int(torch.ceil(torch.tensor(probabilities.shape[-1] * top_k_fraction)).item()))
    k = min(k, probabilities.shape[-1])
    values = probabilities.masked_fill(~mask, -torch.inf).topk(k, dim=-1).values
    valid_count = mask.sum(dim=-1).clamp_min(1)
    values = values.masked_fill(~torch.isfinite(values), 0.0)
    denominator = torch.minimum(valid_count, torch.tensor(k, device=mask.device))
    return values.sum(dim=-1) / denominator


def _symmetric_embedding_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Symmetric KL over normalized half-cycle embeddings."""

    p = torch.softmax(first, dim=-1).clamp_min(1e-7)
    q = torch.softmax(second, dim=-1).clamp_min(1e-7)
    return 0.5 * ((p * (p.log() - q.log())).sum(-1) + (q * (q.log() - p.log())).sum(-1)).mean()


def _masked_embedding_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class PulseBagExpert(nn.Module):
    """Encode pulses and produce exactly one logit per parent signal."""

    def __init__(
        self,
        representation: str,
        *,
        aggregation: str = "top_k_mean",
        top_k_fraction: float = 0.10,
        cycle_consistency: bool = False,
        cycle_weight: float = 0.10,
    ) -> None:
        super().__init__()
        if representation not in {"temporal", "cwt"}:
            raise ValueError("representation must be 'temporal' or 'cwt'")
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if cycle_weight < 0:
            raise ValueError("cycle_weight must be non-negative")
        self.representation = representation
        self.aggregation = aggregation
        self.top_k_fraction = float(top_k_fraction)
        self.cycle_consistency = bool(cycle_consistency)
        self.cycle_weight = float(cycle_weight)
        self.encoder = PulseTemporalEncoder() if representation == "temporal" else PulseCWTEncoder()
        self.classifier = nn.Linear(64, 1)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> Any:
        if values.ndim not in {4, 5, 6}:
            raise ValueError("Pulse bags must have [batch, halves, pulses, ...] dimensions")
        if valid_mask.ndim != 3 or tuple(valid_mask.shape[:2]) != tuple(values.shape[:2]):
            raise ValueError("Pulse masks must align with [batch, halves, pulses]")
        batch, halves, pulses = valid_mask.shape
        if halves != 2 or pulses < 1:
            raise ValueError("Pulse bags must contain two non-empty half-cycle axes")
        if self.representation == "temporal":
            if values.ndim == 4:
                values = values.unsqueeze(3)
            if values.ndim != 5 or values.shape[3] != 1:
                raise ValueError("Temporal pulse bags must be [batch, halves, pulses, 1, time]")
        else:
            if values.ndim != 6 or values.shape[3] != 1:
                raise ValueError("CWT pulse bags must be [batch, halves, pulses, 1, scales, time]")
        flat = values.reshape(batch * halves * pulses, *values.shape[3:])
        embeddings = self.encoder(flat).reshape(batch, halves, pulses, -1)
        pulse_logits = self.classifier(embeddings).squeeze(-1)
        half_logits = _masked_pool(pulse_logits, valid_mask, self.aggregation, self.top_k_fraction)
        signal_probability = torch.sigmoid(half_logits).mean(dim=1)
        signal_logit = torch.logit(signal_probability.clamp(1e-6, 1 - 1e-6))
        cycle_loss = signal_logit.new_zeros(())
        if self.cycle_consistency:
            cycle_loss = _symmetric_embedding_kl(
                _masked_embedding_mean(embeddings[:, 0], valid_mask[:, 0]),
                _masked_embedding_mean(embeddings[:, 1], valid_mask[:, 1]),
            )
        if not return_details:
            return signal_logit
        return signal_logit, {
            "pulse_logits": pulse_logits,
            "half_logits": half_logits,
            "cycle_loss": cycle_loss,
        }


def make_pulse_expert(representation: str, **kwargs: Any) -> PulseBagExpert:
    return PulseBagExpert(representation, **kwargs)
