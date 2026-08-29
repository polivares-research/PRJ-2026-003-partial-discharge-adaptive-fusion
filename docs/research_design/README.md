# Scientific design

The primary task is binary classification of `PD` versus `NonPD`.

The study must separate:

1. representation comparisons;
2. model comparisons by representation;
3. baselines and ablations;
4. noise/SNR experiments;
5. cross-dataset and unseen-object generalization;
6. calibration and uncertainty;
7. reliability-based adaptive fusion.

The primary comparison will use MCC. Recall, F1, PR-AUC, ROC-AUC, and Accuracy
will be secondary metrics. Main results must use multiple seeds and report the
mean, standard deviation, and confidence intervals.
