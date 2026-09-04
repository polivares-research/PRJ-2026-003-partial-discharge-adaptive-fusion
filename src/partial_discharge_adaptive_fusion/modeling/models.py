"""PoC4-compatible expert architectures."""

from __future__ import annotations

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
