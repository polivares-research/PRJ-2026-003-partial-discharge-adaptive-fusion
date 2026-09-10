"""CUDA-only training and leakage-safe prediction helpers."""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from ..config import require_cuda
from .data import MultiInstanceDataset, NumpyDataset
from .models import make_expert, make_multi_instance_expert


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int


@dataclass(frozen=True)
class CrossFitResult:
    """OOF logits plus fold-local training histories for one expert/seed."""

    logits: np.ndarray
    folds: np.ndarray
    histories: dict[int, list[dict[str, float]]]
    window_summaries: dict[str, np.ndarray] | None = None


def set_seed(seed: int) -> None:
    """Set deterministic seeds for a single independent run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_expert(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    kind: str,
    seed: int,
    epochs: int,
    batch_size: int = 128,
    pos_weight: float | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    train_indices: np.ndarray | None = None,
    validation_indices: np.ndarray | None = None,
    validation_batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    multi_instance: bool = False,
    aggregation: str = "top_k_mean",
    top_k_fraction: float = 0.10,
    instance_microbatch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    mixed_precision: bool = False,
    epoch_selection: str = "validation",
) -> TrainingResult:
    """Train one expert on CUDA with validation or registered fixed epochs."""

    device = require_cuda()
    if batch_size < 1:
        raise ValueError("Training batch size must be positive.")
    if validation_batch_size is not None and validation_batch_size < 1:
        raise ValueError("Validation batch size must be positive.")
    if gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be non-negative and prefetch_factor must be positive.")
    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0.")
    if epoch_selection not in {"validation", "fixed"}:
        raise ValueError("epoch_selection must be 'validation' or 'fixed'")
    set_seed(seed)
    model = (
        make_multi_instance_expert(
            kind,
            aggregation=aggregation,
            top_k_fraction=top_k_fraction,
            instance_microbatch_size=instance_microbatch_size,
        )
        if multi_instance else make_expert(kind)
    ).to(device)
    dataset_class = MultiInstanceDataset if multi_instance else NumpyDataset
    train_dataset = dataset_class(train_x, train_y, train_indices)
    validation_dataset = None
    selection_labels = None
    if epoch_selection == "validation":
        validation_dataset = dataset_class(validation_x, validation_y, validation_indices)
        selection_labels = (
            np.asarray(validation_y) if validation_indices is None
            else np.asarray(validation_y)[np.asarray(validation_indices, dtype=np.int64)]
        )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {"num_workers": num_workers, "pin_memory": True}
    validation_options = {"num_workers": num_workers, "pin_memory": True}
    if num_workers:
        loader_options.update({"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor})
        validation_options.update({"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor})
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, **loader_options,
    )
    validation_loader = None if validation_dataset is None else DataLoader(
        validation_dataset, batch_size=validation_batch_size or max(batch_size, 512), shuffle=False,
        **validation_options,
    )
    weight = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_mcc = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_number, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                loss = loss_fn(model(batch_x), batch_y) / gradient_accumulation_steps
            loss.backward()
            if batch_number % gradient_accumulation_steps == 0 or batch_number == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if epoch_selection == "fixed":
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_mcc_at_0.5": float("nan")})
        else:
            validation_logits = predict_logits(model, validation_loader, device, mixed_precision=mixed_precision)
            validation_mcc = _mcc(selection_labels, validation_logits >= 0.0)
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
        raise RuntimeError("No valid model state was produced.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return TrainingResult(model=model, history=history, best_epoch=best_epoch)


def predict_logits(model: nn.Module, loader: DataLoader, device=None, *, mixed_precision: bool = False) -> np.ndarray:
    """Predict logits with explicit CUDA placement."""

    device = device or require_cuda()
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                output = model(batch_x.to(device, non_blocking=True))
            values.append(output.detach().float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def fold_assignments(
    labels: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return deterministic, complete OOF fold IDs."""

    labels = np.asarray(labels)
    folds = np.full(len(labels), -1, dtype=np.int64)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels, np.asarray(groups))
    for fold, (_, holdout) in enumerate(iterator):
        folds[holdout] = fold
    if np.any(folds < 0):
        raise RuntimeError("OOF fold assignment is incomplete.")
    return folds


def cross_fitted_logits(
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    kind: str,
    folds: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
    epochs: int,
    batch_size: int = 128,
    pos_weight: float | None = None,
    imbalance_strategy: str = "none",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    inner_splits: int = 5,
    validation_batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    multi_instance: bool = False,
    aggregation: str = "top_k_mean",
    top_k_fraction: float = 0.10,
    instance_microbatch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    mixed_precision: bool = False,
    fold_transform: Callable[[Any, np.ndarray, np.ndarray, int], Any] | None = None,
    epoch_selection: str = "validation",
) -> CrossFitResult:
    """Generate leakage-safe OOF logits with fold-local validation.

    When ``epoch_selection='fixed'``, the full outer-fit fold is used for the
    registered number of epochs and no validation labels are consulted.  A
    fold transform may fit a normalizer on the outer-fit indices and return a
    lazy or materialized transformed view.  ``folds`` is supplied by the
    frozen manifest and is reused across seeds.
    """

    require_cuda()
    # Keep float16/float32 memmaps lazy; NumpyDataset casts one item at a time.
    shape = getattr(inputs, "shape", None)
    if shape is None:
        raise ValueError("inputs must expose a signal-first shape")
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    if shape[0] != len(labels) or len(folds) != len(labels):
        raise ValueError("Inputs, labels, and folds must have the same number of samples.")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2 or np.any(unique_folds < 0):
        raise ValueError("OOF folds must contain at least two non-negative fold IDs.")
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(labels):
            raise ValueError("Groups must have the same number of samples as labels.")
    logits = np.full(len(labels), np.nan, dtype=np.float64)
    histories: dict[int, list[dict[str, float]]] = {}
    window_summaries: dict[str, np.ndarray] = {}
    prediction_batch_size = validation_batch_size or max(batch_size, 512)
    for fold in unique_folds:
        holdout = np.flatnonzero(folds == fold)
        fit = np.flatnonzero(folds != fold)
        fit_groups = groups[fit] if groups is not None else None
        if epoch_selection == "fixed":
            inner_train = fit
            inner_validation = np.empty(0, dtype=np.int64)
        else:
            inner_count = min(inner_splits, len(np.unique(fit_groups)) if fit_groups is not None else len(fit))
            if inner_count < 2:
                raise ValueError("Not enough outer-fit samples/groups for inner validation.")
            inner = fold_assignments(
                labels[fit], groups=fit_groups, n_splits=inner_count, seed=seed + 1009 * int(fold),
            )
            inner_validation = fit[inner == 0]
            inner_train = fit[inner != 0]
        fold_inputs = inputs if fold_transform is None else fold_transform(inputs, fit, holdout, int(fold))
        result = train_expert(
            fold_inputs, labels, fold_inputs, labels,
            kind=kind, seed=seed + int(fold), epochs=epochs, batch_size=batch_size,
            pos_weight=(
                positive_class_weight(labels[inner_train])
                if imbalance_strategy == "fold_local_pos_weight" else pos_weight
            ),
            learning_rate=learning_rate, weight_decay=weight_decay,
            patience=patience,
            train_indices=inner_train, validation_indices=inner_validation,
            validation_batch_size=validation_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            multi_instance=multi_instance,
            aggregation=aggregation,
            top_k_fraction=top_k_fraction,
            instance_microbatch_size=instance_microbatch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            mixed_precision=mixed_precision,
            epoch_selection=epoch_selection,
        )
        if multi_instance:
            from .predict import logits_and_window_summaries_for_indices

            fold_logits, fold_summaries = logits_and_window_summaries_for_indices(
                result.model, fold_inputs, holdout, batch_size=prediction_batch_size,
                mixed_precision=mixed_precision,
            )
            for name, values in fold_summaries.items():
                window_summaries.setdefault(name, np.full(len(labels), np.nan, dtype=np.float64))
                window_summaries[name][holdout] = values
        else:
            from .predict import logits_for_indices

            fold_logits = logits_for_indices(
                result.model, fold_inputs, holdout, batch_size=prediction_batch_size,
                mixed_precision=mixed_precision,
            )
        logits[holdout] = fold_logits
        histories[int(fold)] = result.history
        del result
        cleanup_cuda()
    if not np.isfinite(logits).all():
        raise RuntimeError("OOF prediction coverage is incomplete.")
    return CrossFitResult(
        logits=logits, folds=folds.copy(), histories=histories,
        window_summaries=window_summaries or None,
    )


def _mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(labels, predictions))


def positive_class_weight(labels: np.ndarray) -> float | None:
    """Compute BCE positive weight from the current training subset only."""

    labels = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None
    return float(negatives / positives)


def cleanup_cuda() -> None:
    """Release Python and CUDA allocations between independent fits."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
