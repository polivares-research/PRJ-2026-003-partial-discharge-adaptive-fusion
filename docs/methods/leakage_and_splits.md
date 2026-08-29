# Leakage and splits

## VSB

- The grouping unit is `id_measurement`.
- All three phases of each measurement must remain in the same split.
- Learnable transformations, normalizers, feature selection, calibrators, and
  thresholds are fitted using training/validation only.
- The public Kaggle test set is not used as a labeled scientific test set.

## Rauscher

- Se conservan las particiones originales.
- `Tr1`, `Va1`, and `Te1` form the main protocol.
- `Tr0`, `Va0`, and `Te0` remain available for the source-described tuning protocol.
- `Te2` is reserved for final generalization to unseen objects.

## General rules

- Splits must be stored as reproducible manifests.
- Each split records the seed, version, grouping, and generation date.
- OOF predictions are produced only within the development set.
- A model is never selected using results from `Te1` or `Te2`.
