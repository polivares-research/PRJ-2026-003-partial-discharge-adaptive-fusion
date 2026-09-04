# Adaptive PD/NonPD Fusion PoC — LLM Evaluation Handoff

## Evaluation request

Review the executed PoC and assess whether the evidence supports the adaptive
fusion hypothesis. Then evaluate whether a different model or fusion strategy
is justified, identify the main technical weaknesses, and propose a leakage-safe
next experiment. The human decision remains pending; do not convert the
automatic `REVISE` recommendation into `NO-GO`.

## Research question

Does uncertainty-based adaptive fusion of two heterogeneous neural experts—one
temporal and one spectral—improve MCC over the best individual expert and over
a fixed probability average?

Initial feasibility support requires the adaptive fusion to improve MCC over
all baselines by at least `0.005`.

## Data and protocol

- Dataset: `dataset-pd-noise@v1`.
- Labels: `NonPD=0`, `PD=1`.
- `Tr0`: 6,130 NonPD and 6,130 PD samples.
- `Va0`: 1,533 NonPD and 1,533 PD samples.
- `Te0`: 22,985 NonPD and 22,985 PD samples.
- Signals contain 400 samples each.
- `Tr0` is used for training; `Va0` is used for early stopping,
  calibration, and threshold selection; `Te0` is used only for final
  evaluation.
- `Te1` and `Te2` are reserved and were not loaded.
- Raw MATLAB files are intentionally excluded from this package.

## Models and configuration

- Framework: PyTorch `2.9.1`, CUDA build `12.9`.
- Device used for the recorded run: `cuda` on an NVIDIA GeForce RTX 3060
  Laptop GPU.
- Seed: `42`; batch size: `128`; maximum epochs: `20`; early-stopping
  patience: `5`.
- Both experts use a small 1D CNN: Conv1d/BatchNorm/ReLU/MaxPool blocks with
  widths `16 -> 32 -> 64`, adaptive average pooling, and a linear output.
- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-4`.
- Temporal input: standardized raw signal, 400 features.
- Spectral input: standardized log-magnitude rFFT, 201 features.
- Probability calibration: temperature scaling selected on `Va0`.
- Adaptive fusion: per-sample weights proportional to `1 - normalized entropy`
  of each calibrated probability.
- Thresholds are selected on `Va0` from `0.05` to `0.95` in steps of `0.005`.

## Recorded results on Te0

| Method | MCC | F1 | Accuracy | ROC-AUC | PR-AUC | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Temporal | 0.948191 | 0.973648 | 0.973961 | 0.995869 | 0.996470 | 0.962062 | 0.014140 |
| Spectral | 0.861637 | 0.927876 | 0.930041 | 0.980753 | 0.983565 | 0.900022 | 0.039939 |
| Fixed mean | 0.936821 | 0.968163 | 0.968371 | 0.995066 | 0.995763 | 0.961845 | 0.025103 |
| Adaptive entropy | 0.942515 | 0.970652 | 0.971068 | 0.995094 | 0.995935 | 0.956885 | 0.014749 |

Confusion matrices (`TN, FP, FN, TP`):

- Temporal: `22,660, 325, 872, 22,113`.
- Spectral: `22,067, 918, 2,298, 20,687`.
- Fixed mean: `22,408, 577, 877, 22,108`.
- Adaptive entropy: `22,646, 339, 991, 21,994`.

Calibration temperatures were `0.707107` for temporal and `0.812252` for
spectral. Selected thresholds were `0.245` temporal, `0.505` spectral,
`0.350` fixed mean, and `0.255` adaptive entropy.

## Interpretation boundary

The best baseline is the temporal expert at MCC `0.948191`. The success
threshold for a new adaptive strategy under this protocol is therefore MCC
`>= 0.953191` on `Te0`, after all model and threshold choices have been made
using only `Tr0` and `Va0`.

The recorded adaptive gain is `-0.005676` versus the best baseline. Adaptive
fusion improves over fixed mean by `0.005694`, but does not beat the temporal
expert. The automatic outcome is `REVISE`; the human decision is still pending.

## Package contents

- `notebook/poc.ipynb`: executed notebook with source and outputs.
- `results/summary.json`: complete machine-readable results and provenance.
- `results/poc_summary.png`: generated diagnostic figure.
- `assessment/metrics.json`: assessment copy of the complete results.
- `assessment/evidence.md`: concise evidence record.
- `assessment/decision.md`: human decision placeholder.
- `assessment/limitations.md`: limitations and threats to validity.
- `question.md`: research question and success criterion.
- `research/hypotheses.md`: H1/H0 and rationale.
- `environment/environment.yaml`: recorded environment metadata.
- `data/datasets.yaml`: dataset identifier and catalog metadata.
- `pyproject.toml`: project and PoC dependency declarations.

The dataset files, model weights, credentials, and personal filesystem paths
are not included.
