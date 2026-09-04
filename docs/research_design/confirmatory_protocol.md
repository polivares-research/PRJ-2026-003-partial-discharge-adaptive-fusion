# Confirmatory two-dataset protocol

## Objective

Evaluate temporal and continuous-wavelet-transform (CWT) experts, fixed
multimodal fusion, and reliability-aware adaptive fusion on two independent
partial-discharge datasets. Matthews correlation coefficient (MCC) is the
primary metric.

## Data access

The only supported runtime data entry point is the repository-local adapter:

```python
from partial_discharge_adaptive_fusion.dataset import resolve_dataset
dataset = resolve_dataset(dataset_id, version, raw_root="data/raw")
```

The canonical identifiers are:

- `engineering-partial-discharge-noise-signals@v1`;
- `engineering-vsb-power-line-fault-detection@2018-kaggle-snapshot`.

Raw data are transferred separately and are never committed. Set
`PD_RAW_DATA_ROOT` to override the default `data/raw`. The optional
`scripts/stage_local_raw_data.py` utility uses the historical catalog only on
the source machine.

## MATLAB dataset

Use the source partitions unchanged:

- `Tr1.mat`: expert training and OOF source;
- `Va1.mat`: calibration, threshold, model and fixed-weight selection;
- `Te1.mat`: secondary locked evaluation only; it was observed in PoC2;
- `Te2.mat`: primary confirmatory evaluation on previously unused objects.

`Te2.mat` must not be opened before the configuration freeze. It must never be
used for tuning, feature selection, calibration or threshold selection.
The MATLAB source exposes 400-sample signals in `signals` and categorical
labels in `labels`; no per-sample group identifier is available.

## VSB dataset

Use labeled `train.parquet` together with `metadata_train.csv`. The official
test Parquet is not scored because its labels are unavailable.

All three phase rows for one `id_measurement` are one grouping unit. A fixed
approximately 60/20/20 holdout is produced with five-fold
`StratifiedGroupKFold`: folds 0, 1 and 2–4 are test, validation and train,
respectively. Five-fold grouped OOF predictions are generated inside the
train subset. The exact row assignment is saved as a split manifest.

The draft configuration keeps the VSB signal-to-input mapping unresolved until
the Parquet schema and native signal representation are inspected. That audit
has now confirmed an `800000`-sample `int8` native signal, three phases per
measurement, and catalog sampling metadata of 40 MHz over 20 ms. The frozen
configuration therefore uses the native signal without resampling or windows;
normalization remains train-only and the inherited Morlet/log-power CWT uses
32 scales and 120 time bins. This decision was recorded before any model
training or holdout evaluation.

## Experts and fusion

PoC4 expert architectures are inherited without architecture search:

- temporal: 1D CNN with 16/32/64 channels, BatchNorm, ReLU, pooling and
  global-average pooling;
- CWT: 2D CNN with 16/32/64 channels, BatchNorm, ReLU, pooling and
  global-average pooling.

The MATLAB CWT configuration is complex Morlet (`w0=6`), 32 geometric scales
from 1.5 to 64, 120 time bins, log power and a `-10` floor. Scaling is fitted
on training data only. VSB CWT dimensions are dataset-specific but must be
frozen before evaluation.

For every seed, OOF expert predictions are used to train correctness/reliability
models. Candidate reliability estimators are logistic regression, histogram
gradient boosting and a small MLP. Adaptive probabilities use reliability-
proportional weights; conservative fusion falls back to best fixed fusion when
the reliability difference is below the validation-selected delta.

## Reproducibility and statistics

The main seeds are `42, 43, 44, 45, 46`. Every seed independently regenerates
expert training, OOF predictions, learned features, reliability estimators and
fusion outputs. Deterministic representations may be cached per dataset and
configuration.

Report MCC, Accuracy, Precision, PD Recall, Specificity, F1, PR-AUC, ROC-AUC,
Brier score, ECE, confusion counts and errors. Use paired sample bootstrap
within each seed with approximately 10,000 iterations and exact two-sided
McNemar tests. Do not treat repeated predictions on the same test samples
across seeds as independent observations.

The adaptive target inherited from PoC4 is `Delta MCC >= 0.005`, with at least
four of five positive seeds, an interval above zero, net error reduction and
no important PD-recall degradation. The final scientific interpretation is
chosen only after comparing both datasets.

## Batch-4 amendment

The original frozen v1 protocol specifies physical batch 128. Because the
native VSB temporal input makes that batch exceed the available local VRAM, the
new frozen configuration
`configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml` applies physical
batch 4, inference batch 4 and one gradient-accumulation step consistently to
both representations and both datasets. This is a new experimental variant;
its results must not be pooled with v1 results as if the training protocols
were identical. The input representation, model architectures, splits, seeds,
metrics and fusion rules remain unchanged.

## Execution protection

All neural-network stages require CUDA. The runner must record Python,
PyTorch, CUDA, GPU, package versions and repository commit, and must stop
before model construction if CUDA is unavailable. The CPU fallback is
forbidden.
