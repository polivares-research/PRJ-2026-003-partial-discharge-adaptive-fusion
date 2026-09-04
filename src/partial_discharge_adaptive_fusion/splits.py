"""Leakage-safe split manifests for the two source datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class SplitManifest:
    """A portable sample-to-split assignment."""

    frame: pd.DataFrame
    dataset_id: str
    dataset_version: str
    split_seed: int

    def validate(self) -> None:
        required = {"sample_id", "split", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        if self.frame["sample_id"].duplicated().any():
            raise ValueError("Manifest contains duplicate sample IDs.")
        if self.frame["split"].isna().any() or (self.frame["split"] == "unassigned").any():
            raise ValueError("Manifest contains unassigned samples.")
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            if "group_id" not in self.frame:
                continue
            left_groups = set(self.frame.loc[self.frame["split"] == left, "group_id"].dropna())
            right_groups = set(self.frame.loc[self.frame["split"] == right, "group_id"].dropna())
            overlap = left_groups & right_groups
            if overlap:
                raise ValueError(f"Groups overlap between {left} and {right}: {sorted(overlap)[:5]}")

    def write_csv(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(destination, index=False)


def matlab_manifest(
    partition_labels: dict[str, np.ndarray],
    *,
    dataset_id: str,
    dataset_version: str,
    split_seed: int = 42,
    oof_splits: int = 5,
) -> SplitManifest:
    """Preserve source partitions and create OOF folds within the train partition."""

    rows = []
    for partition, labels in partition_labels.items():
        labels = np.asarray(labels, dtype=np.int64)
        split = {
            "Tr1.mat": "train", "Va1.mat": "validation", "Te1.mat": "test_historical",
            "Te2.mat": "test_confirmatory", "Tr0.mat": "historical_train",
            "Va0.mat": "historical_validation", "Te0.mat": "historical_test",
        }.get(partition, "unclassified")
        folds = np.full(len(labels), -1, dtype=np.int64)
        if partition == "Tr1.mat":
            splitter = StratifiedKFold(n_splits=oof_splits, shuffle=True, random_state=split_seed)
            for fold, (_, holdout) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
                folds[holdout] = fold
        rows.extend({
            "sample_id": f"{partition}:{index:08d}", "group_id": None,
            "partition": partition, "split": split, "oof_fold": int(folds[index]),
            "label": int(label),
        } for index, label in enumerate(labels))
    manifest = SplitManifest(pd.DataFrame(rows), dataset_id, dataset_version, split_seed)
    manifest.validate()
    return manifest


def vsb_grouped_manifest(
    metadata: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_version: str,
    split_seed: int = 42,
    outer_splits: int = 5,
    oof_splits: int = 5,
) -> SplitManifest:
    """Create fixed approximately 60/20/20 splits without breaking measurements."""

    required = {"signal_id", "id_measurement", "target"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"VSB metadata missing columns: {sorted(required - set(metadata.columns))}")
    frame = metadata.copy().reset_index(drop=True)
    labels = frame["target"].astype(int).to_numpy()
    groups = frame["id_measurement"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=split_seed)
    assignments = np.full(len(frame), "unassigned", dtype=object)
    folds = list(outer.split(np.zeros(len(frame)), labels, groups))
    _, test_indices = folds[0]
    _, validation_indices = folds[1]
    train_indices = np.concatenate([folds[i][1] for i in range(2, outer_splits)])
    assignments[train_indices] = "train"
    assignments[validation_indices] = "validation"
    assignments[test_indices] = "test"
    if np.any(assignments == "unassigned"):
        raise RuntimeError("Outer VSB split did not cover every sample.")

    oof_folds = np.full(len(frame), -1, dtype=np.int64)
    inner = StratifiedGroupKFold(n_splits=oof_splits, shuffle=True, random_state=split_seed)
    for fold, (_, holdout_relative) in enumerate(inner.split(
        np.zeros(len(train_indices)), labels[train_indices], groups[train_indices]
    )):
        oof_folds[train_indices[holdout_relative]] = fold
    result = pd.DataFrame({
        "sample_id": frame["signal_id"].astype(str),
        "group_id": groups,
        "partition": "train",
        "split": assignments,
        "oof_fold": oof_folds,
        "phase": frame["phase"].astype(str),
        "label": labels,
    })
    result.loc[result["split"] != "train", "partition"] = "heldout"
    manifest = SplitManifest(result, dataset_id, dataset_version, split_seed)
    manifest.validate()
    return manifest


def assert_complete_oof(manifest: SplitManifest, split: str = "train") -> None:
    values = manifest.frame.loc[manifest.frame["split"] == split, "oof_fold"]
    if values.empty or (values < 0).any() or values.nunique() < 2:
        raise ValueError(f"OOF coverage is incomplete for split {split}.")
