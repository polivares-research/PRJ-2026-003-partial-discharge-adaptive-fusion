# Experimentos

Each experiment must declare at least:

- dataset and version;
- partition/split;
- representation;
- model;
- hyperparameters;
- seed;
- imbalance strategy;
- threshold-selection rule;
- configuration version.

The primary metric will be MCC. Recall, F1, PR-AUC, ROC-AUC, and Accuracy will
be secondary metrics.

The original batch-128 protocol is `two-dataset-confirmatory-v1-frozen.yaml`.
The GPU-compatible amendment is
`two-dataset-confirmatory-v2-batch4-localraw.yaml`, which applies physical and inference
batch 4 consistently to all four dataset/representation cases. These variants
must be analyzed and reported separately.

The representation-aware amendment is selected by
`two-dataset-vsb-window-selection-localraw.yaml` using development data and
seed 42 only. It writes the frozen
`two-dataset-confirmatory-v3-windowed-localraw.yaml`, which applies the shared
encoder/bag abstraction: MATLAB `K=1`, VSB `K>1`, parent-signal loss and
signal-level evaluation. Its results are stored separately from v1/v2.


## V4 gated execution

`two-dataset-confirmatory-v4-windowed-localraw.yaml` is a development-selection configuration. Run `scripts/execute_v4_pipeline.py` from the repository root with `PD_RAW_DATA_ROOT` or `--raw-root`. The command audits VSB cycle metadata, evaluates seeds 42–44 without holdouts, freezes the selected candidate, and only then runs confirmatory seeds 42–46.
