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
