# Current-Expert Adaptive Fusion Confirmation Plan

## Status and scope

This protocol is registered on `feature/current-expert-adaptive-fusion`, created
from commit `d90c959`. It is a new, self-contained confirmation protocol. V3,
V4, V5, forensic, modality, and historical full-signal reports remain
immutable context and are not active inputs to selection.

The canonical data interface is `PD_RAW_DATA_ROOT=data/raw`. Dropbox,
`researchdata`, public weights, and external test data are not used.

## Scientific decisions

- MATLAB uses `Tr1`/`Va1`; the clean temporal expert uses nine fixed epochs,
  fold-local train-only normalization, and never selects an epoch from `Va1`.
- VSB uses the fixed forensic `S+T` temporal HGB representation and the current
  global spectrogram expert. Signals remain grouped by `id_measurement`.
- Development uses seeds 42--44, five OOF folds, MCC, and thresholds selected
  only from OOF probabilities on the 0.05--0.95 grid with step 0.005.
- The registered reliability model is regularized logistic regression using
  calibrated probabilities, logits, confidence, entropy, and disagreement.
- The primary comparisons are `best_fixed - best_individual`,
  `adaptive - best_fixed`, and `adaptive - best_individual`.
- A dataset opens its holdout only if the expert gate and a predeclared fusion
  gate pass. A failed dataset remains locked.

## Data flow

```text
current expert source
  -> source/schema/hash validation
  -> clean MATLAB temporal regeneration
  -> paired signal-level development tables
  -> cross-fitted calibration/reliability/fusion
  -> validation gates
  -> frozen configuration
  -> dataset-specific confirmatory expert generation
  -> one locked holdout evaluation
```

All calibration, thresholds, fixed weights, reliability models, and
conservative fallback parameters are fitted from training OOF rows. Validation
rows are scored once and do not tune components.

## Gates

VSB requires global spectrogram mean validation MCC at least 0.55 and every
development seed at least 0.50. MATLAB requires both current experts to have
mean MCC at least 0.90 and no seed below 0.85. A fusion method must improve the
OOF-selected best individual by at least 0.005 in mean MCC, in at least two of
three development seeds, with a positive hierarchical paired-bootstrap lower
bound.

After freezing, five seeds (42--46) are used for the eligible dataset only.
The confirmatory verdict requires improvement over both best fixed and best
individual fusion by at least 0.005, at least four positive seeds, a positive
hierarchical 95% interval, and no mean PD-recall reduction greater than 0.02.

## Protection and limitations

VSB grouped holdout, official VSB test, MATLAB Te1, and MATLAB Te2 remain
closed during development. Historical metrics cannot enter active decisions.
The report must identify whether a dataset passed independently and must not
pool raw observations across datasets. Generated predictions, caches,
checkpoints, and large tables remain ignored and outside Git.

