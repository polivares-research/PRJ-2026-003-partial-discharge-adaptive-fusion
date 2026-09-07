"""CUDA-gated orchestration for one independent two-expert seed.

Dataset-specific adapters and representations prepare arrays before calling
this module.  This keeps the orchestration reusable while making the input
policy decision explicit and auditable for VSB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import require_cuda
from ..modeling.predict import logits_and_window_summaries_for_array, logits_for_array
from ..modeling.train import (
    CrossFitResult,
    cleanup_cuda,
    cross_fitted_logits,
    positive_class_weight,
    train_expert,
)


@dataclass(frozen=True)
class ExpertSeedOutput:
    """All logits needed for fusion for one dataset and seed."""

    seed: int
    temporal_oof_logits: np.ndarray
    cwt_oof_logits: np.ndarray
    temporal_validation_logits: np.ndarray
    cwt_validation_logits: np.ndarray
    temporal_test_logits: np.ndarray
    cwt_test_logits: np.ndarray
    temporal_oof_folds: np.ndarray
    cwt_oof_folds: np.ndarray
    temporal_histories: dict[int, list[dict[str, float]]]
    cwt_histories: dict[int, list[dict[str, float]]]
    additional_test_logits: dict[str, tuple[np.ndarray, np.ndarray]]
    temporal_oof_window_summaries: dict[str, np.ndarray] | None = None
    cwt_oof_window_summaries: dict[str, np.ndarray] | None = None
    temporal_validation_window_summaries: dict[str, np.ndarray] | None = None
    cwt_validation_window_summaries: dict[str, np.ndarray] | None = None
    temporal_test_window_summaries: dict[str, np.ndarray] | None = None
    cwt_test_window_summaries: dict[str, np.ndarray] | None = None


def train_two_experts_for_seed(
    *,
    train_temporal: np.ndarray,
    train_cwt: np.ndarray,
    train_labels: np.ndarray,
    oof_folds: np.ndarray,
    validation_temporal: np.ndarray,
    validation_cwt: np.ndarray,
    validation_labels: np.ndarray,
    test_temporal: np.ndarray,
    test_cwt: np.ndarray,
    test_labels: np.ndarray,
    additional_tests: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    seed: int,
    temporal_epochs: int,
    cwt_epochs: int,
    temporal_kind: str = "temporal",
    cwt_kind: str = "cwt",
    batch_size: int = 128,
    imbalance_strategy: str = "none",
    groups: np.ndarray | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    inference_batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    multi_instance: bool = False,
    aggregation: str = "top_k_mean",
    top_k_fraction: float = 0.10,
    instance_microbatch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    mixed_precision: bool = False,
) -> ExpertSeedOutput:
    """Train temporal/CWT experts, OOF cross-fit them, and predict held-out data.

    This function calls the CUDA gate before constructing either model.  For
    VSB, ``imbalance_strategy='fold_local_pos_weight'`` computes the positive
    weight inside each inner training subset; no validation/test labels enter
    that calculation.
    """

    require_cuda()
    labels = np.asarray(train_labels, dtype=np.int64)
    validation_labels = np.asarray(validation_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if len(labels) != len(oof_folds):
        raise ValueError("Train labels and OOF folds must have the same length.")
    if imbalance_strategy not in {"none", "fold_local_pos_weight"}:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")
    global_weight = positive_class_weight(labels) if imbalance_strategy == "fold_local_pos_weight" else None
    prediction_batch_size = max(batch_size, 512) if inference_batch_size is None else inference_batch_size
    if prediction_batch_size < 1:
        raise ValueError("Inference batch size must be positive.")

    print(f"seed {seed}: temporal OOF ({len(np.unique(oof_folds))} folds)", flush=True)
    temporal_oof: CrossFitResult = cross_fitted_logits(
        train_temporal, labels, kind=temporal_kind, folds=oof_folds, seed=seed,
        groups=groups, epochs=temporal_epochs, batch_size=batch_size,
        pos_weight=global_weight, imbalance_strategy=imbalance_strategy,
        learning_rate=learning_rate, weight_decay=weight_decay, patience=patience,
        validation_batch_size=prediction_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        multi_instance=multi_instance, aggregation=aggregation,
        top_k_fraction=top_k_fraction, instance_microbatch_size=instance_microbatch_size,
        num_workers=num_workers, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, mixed_precision=mixed_precision,
    )
    print(f"seed {seed}: CWT OOF ({len(np.unique(oof_folds))} folds)", flush=True)
    cwt_oof: CrossFitResult = cross_fitted_logits(
        train_cwt, labels, kind=cwt_kind, folds=oof_folds, seed=seed + 100,
        groups=groups, epochs=cwt_epochs, batch_size=batch_size,
        pos_weight=global_weight, imbalance_strategy=imbalance_strategy,
        learning_rate=learning_rate, weight_decay=weight_decay, patience=patience,
        validation_batch_size=prediction_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        multi_instance=multi_instance, aggregation=aggregation,
        top_k_fraction=top_k_fraction, instance_microbatch_size=instance_microbatch_size,
        num_workers=num_workers, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, mixed_precision=mixed_precision,
    )

    print(f"seed {seed}: final temporal fit", flush=True)
    temporal_result = train_expert(
        train_temporal, labels, validation_temporal, validation_labels,
        kind=temporal_kind, seed=seed, epochs=temporal_epochs, batch_size=batch_size,
        pos_weight=global_weight, learning_rate=learning_rate, weight_decay=weight_decay,
        patience=patience, validation_batch_size=prediction_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        multi_instance=multi_instance, aggregation=aggregation,
        top_k_fraction=top_k_fraction, instance_microbatch_size=instance_microbatch_size,
        num_workers=num_workers, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, mixed_precision=mixed_precision,
    )
    if multi_instance:
        temporal_validation, temporal_validation_summary = logits_and_window_summaries_for_array(
            temporal_result.model, validation_temporal, batch_size=prediction_batch_size, mixed_precision=mixed_precision,
        )
        temporal_test, temporal_test_summary = logits_and_window_summaries_for_array(
            temporal_result.model, test_temporal, batch_size=prediction_batch_size, mixed_precision=mixed_precision,
        )
    else:
        temporal_validation = logits_for_array(temporal_result.model, validation_temporal, batch_size=prediction_batch_size, mixed_precision=mixed_precision)
        temporal_test = logits_for_array(temporal_result.model, test_temporal, batch_size=prediction_batch_size, mixed_precision=mixed_precision)
        temporal_validation_summary = None
        temporal_test_summary = None
    additional_temporal: dict[str, np.ndarray] = {}
    additional_cwt: dict[str, np.ndarray] = {}
    for name, (additional_temporal_values, additional_cwt_values) in (additional_tests or {}).items():
        additional_temporal[name] = logits_for_array(
            temporal_result.model, additional_temporal_values, batch_size=prediction_batch_size,
            multi_instance=multi_instance, mixed_precision=mixed_precision,
        )
    del temporal_result
    cleanup_cuda()

    print(f"seed {seed}: final CWT fit", flush=True)
    cwt_result = train_expert(
        train_cwt, labels, validation_cwt, validation_labels,
        kind=cwt_kind, seed=seed + 100, epochs=cwt_epochs, batch_size=batch_size,
        pos_weight=global_weight, learning_rate=learning_rate, weight_decay=weight_decay,
        patience=patience, validation_batch_size=prediction_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        multi_instance=multi_instance, aggregation=aggregation,
        top_k_fraction=top_k_fraction, instance_microbatch_size=instance_microbatch_size,
        num_workers=num_workers, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, mixed_precision=mixed_precision,
    )
    if multi_instance:
        cwt_validation, cwt_validation_summary = logits_and_window_summaries_for_array(
            cwt_result.model, validation_cwt, batch_size=prediction_batch_size, mixed_precision=mixed_precision,
        )
        cwt_test, cwt_test_summary = logits_and_window_summaries_for_array(
            cwt_result.model, test_cwt, batch_size=prediction_batch_size, mixed_precision=mixed_precision,
        )
    else:
        cwt_validation = logits_for_array(cwt_result.model, validation_cwt, batch_size=prediction_batch_size, mixed_precision=mixed_precision)
        cwt_test = logits_for_array(cwt_result.model, test_cwt, batch_size=prediction_batch_size, mixed_precision=mixed_precision)
        cwt_validation_summary = None
        cwt_test_summary = None
    additional_test_logits = {
        name: (
            additional_temporal[name],
            logits_for_array(cwt_result.model, additional_cwt_values, batch_size=prediction_batch_size,
                             multi_instance=multi_instance, mixed_precision=mixed_precision),
        )
        for name, (additional_temporal_values, additional_cwt_values) in (additional_tests or {}).items()
    }
    del cwt_result
    cleanup_cuda()
    print(f"seed {seed}: expert predictions ready", flush=True)
    return ExpertSeedOutput(
        seed=seed, temporal_oof_logits=temporal_oof.logits, cwt_oof_logits=cwt_oof.logits,
        temporal_validation_logits=temporal_validation, cwt_validation_logits=cwt_validation,
        temporal_test_logits=temporal_test, cwt_test_logits=cwt_test,
        temporal_oof_folds=temporal_oof.folds, cwt_oof_folds=cwt_oof.folds,
        temporal_histories=temporal_oof.histories, cwt_histories=cwt_oof.histories,
        additional_test_logits=additional_test_logits,
        temporal_oof_window_summaries=temporal_oof.window_summaries,
        cwt_oof_window_summaries=cwt_oof.window_summaries,
        temporal_validation_window_summaries=temporal_validation_summary,
        cwt_validation_window_summaries=cwt_validation_summary,
        temporal_test_window_summaries=temporal_test_summary,
        cwt_test_window_summaries=cwt_test_summary,
    )
