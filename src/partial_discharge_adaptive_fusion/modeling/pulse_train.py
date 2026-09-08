"""CUDA-only training helpers for V5 signal-level pulse bags."""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from ..config import require_cuda
from .pulse_data import PulseBagDataset
from .pulse_models import PulseBagExpert


@dataclass(frozen=True)
class PulseTrainingResult:
    model: PulseBagExpert
    history: list[dict[str, float]]
    best_epoch: int


@dataclass(frozen=True)
class PulseCrossFitResult:
    logits: np.ndarray
    folds: np.ndarray
    histories: dict[int, list[dict[str, float]]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(
    values: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    dataset = PulseBagDataset(values, valid_mask, labels, indices)
    options = {"num_workers": num_workers, "pin_memory": True}
    if num_workers:
        options.update({"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor})
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **options)


def shutdown_pulse_loader(loader: DataLoader) -> None:
    """Release persistent worker pipes after one short-lived training stage."""

    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def train_pulse_expert(
    values: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    *,
    representation: str,
    seed: int,
    epochs: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    batch_size: int = 4,
    inference_batch_size: int = 16,
    pos_weight: float | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    cycle_consistency: bool = False,
    cycle_weight: float = 0.10,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    mixed_precision: bool = False,
) -> PulseTrainingResult:
    """Fit one pulse expert; validation is always at parent-signal level."""

    device = require_cuda()
    _set_seed(seed)
    labels = np.asarray(labels, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    train_loader = _loader(
        values, valid_mask, labels, train_indices, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )
    validation_loader = _loader(
        values, valid_mask, labels, validation_indices, batch_size=inference_batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )
    model = PulseBagExpert(
        representation, cycle_consistency=cycle_consistency, cycle_weight=cycle_weight,
    ).to(device)
    weight = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    best_state: dict[str, torch.Tensor] | None = None
    best_mcc = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    validation_labels = labels[validation_indices]
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_values, batch_mask, batch_labels in train_loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                logits, details = model(batch_values, batch_mask, return_details=True)
                loss = loss_fn(logits, batch_labels)
                if cycle_consistency:
                    loss = loss + cycle_weight * details["cycle_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_logits = predict_pulse_logits(model, validation_loader, device, mixed_precision=mixed_precision)
        predictions = validation_logits >= 0.0
        from sklearn.metrics import matthews_corrcoef

        validation_mcc = float(matthews_corrcoef(validation_labels, predictions))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_mcc_at_0.5": validation_mcc})
        if validation_mcc > best_mcc + 1e-12:
            best_mcc = validation_mcc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No pulse expert checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval().to(device)
    output = PulseTrainingResult(model=model, history=history, best_epoch=best_epoch)
    shutdown_pulse_loader(train_loader)
    shutdown_pulse_loader(validation_loader)
    return output


def predict_pulse_logits(model: PulseBagExpert, loader: DataLoader, device=None, *, mixed_precision: bool = False) -> np.ndarray:
    device = device or require_cuda()
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch_values, batch_mask = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                logits = model(batch_values, batch_mask)
            values.append(logits.float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def _folds(labels: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: int) -> np.ndarray:
    result = np.full(len(labels), -1, dtype=np.int64)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels, groups)
    for fold, (_, holdout) in enumerate(iterator):
        result[holdout] = fold
    if np.any(result < 0):
        raise ValueError("Pulse OOF folds are incomplete")
    return result


def cross_fitted_pulse_logits(
    values: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    *,
    representation: str,
    folds: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
    epochs: int = 7,
    batch_size: int = 4,
    inference_batch_size: int = 16,
    cycle_consistency: bool = False,
    cycle_weight: float = 0.10,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    mixed_precision: bool = False,
) -> PulseCrossFitResult:
    """Generate leakage-safe parent-signal OOF logits for one pulse expert."""

    require_cuda()
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    if len(values) != len(labels) or len(valid_mask) != len(labels) or len(folds) != len(labels):
        raise ValueError("Pulse values, masks, labels, and folds must align")
    logits = np.full(len(labels), np.nan, dtype=np.float64)
    histories: dict[int, list[dict[str, float]]] = {}
    for fold in np.unique(folds):
        holdout = np.flatnonzero(folds == fold)
        fit = np.flatnonzero(folds != fold)
        fit_groups = None if groups is None else np.asarray(groups)[fit]
        inner_splits = min(5, len(np.unique(fit_groups)) if fit_groups is not None else len(fit))
        inner = _folds(labels[fit], fit_groups, inner_splits, seed + 1009 * int(fold))
        inner_train, inner_validation = fit[inner != 0], fit[inner == 0]
        positives = int(np.sum(labels[inner_train] == 1))
        negatives = int(np.sum(labels[inner_train] == 0))
        result = train_pulse_expert(
            values, valid_mask, labels, representation=representation,
            seed=seed + int(fold), epochs=epochs, train_indices=inner_train,
            validation_indices=inner_validation, batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            pos_weight=(negatives / positives if positives else None),
            cycle_consistency=cycle_consistency, cycle_weight=cycle_weight,
            num_workers=num_workers, persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor, mixed_precision=mixed_precision,
        )
        loader = _loader(
            values, valid_mask, labels, holdout, batch_size=inference_batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        )
        logits[holdout] = predict_pulse_logits(result.model, loader, mixed_precision=mixed_precision)
        histories[int(fold)] = result.history
        shutdown_pulse_loader(loader)
        del result, loader
        gc.collect()
        torch.cuda.empty_cache()
    if not np.isfinite(logits).all():
        raise RuntimeError("Pulse OOF coverage is incomplete")
    return PulseCrossFitResult(logits=logits, folds=folds.copy(), histories=histories)
