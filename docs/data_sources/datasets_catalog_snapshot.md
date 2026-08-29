# Data-source snapshot

Inspection performed on 2026-08-25. The external catalog is:

```text
/home/polivares/Dropbox/Work/Research/datasets_catalog/
```

## VSB Power Line Fault Detection

Ruta relativa:

```text
data/raw/dataset-vsb-power-line-fault-detection/
```

It contains `train.parquet`, `test.parquet`, train/test metadata, a submission
template, and provenance metadata. The catalog records 8,712 training signals,
20,337 test signals, 800,000 samples per signal, and three phases per
`id_measurement`. The training set contains 525 positives and 8,187 negatives.

Mandatory rule: split by `id_measurement`, never by `signal_id`, so that the
three phases remain together.

## Dataset of Partial Discharge and Noise Signals

Ruta relativa:

```text
data/raw/dataset-pd-noise/
```

It contains `Tr0`, `Va0`, `Te0`, `Tr1`, `Va1`, `Te1`, and `Te2` in MATLAB format,
along with the README, Figshare metadata, and associated paper. Each signal has
400 samples. `Te2` is the generalization partition and must not be used for
hyperparameter selection.

## Open items before scientific coding

- Verify licensing and redistribution terms for both datasets.
- Document the internal schema of the MATLAB structures.
- Inspect the VSB Parquet schema.
- Confirm amplitude units.
- Confirm how unseen stators/objects are represented in `Te2`.

Raw data remains outside this repository.
