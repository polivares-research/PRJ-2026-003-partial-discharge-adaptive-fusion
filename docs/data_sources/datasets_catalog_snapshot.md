# Data-source snapshot

Inspection performed on 2026-08-25. This document records the source/catalog
snapshot used to stage the portable files. The runtime no longer requires the
catalog: it reads the manually transferred files from `data/raw` through
`PD_RAW_DATA_ROOT`.

```text
data/raw/
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

## Confirmatory audit status

The structural audits are now recorded in
`results/manifests/matlab_audit.json` and `results/manifests/vsb_audit.json`.
They confirmed the MATLAB `signals`/`labels` structures and 400-sample
partitions, and the VSB Parquet schema, native 800,000-sample `int8` signals,
40 MHz sampling metadata and three-phase measurement grouping. The VSB input
policy is frozen in `results/manifests/vsb_input_policy.json`; it uses the
native signal without resampling or windows.

Remaining non-blocking source-data items are licensing/redistribution review,
amplitude-unit interpretation and the published description of unseen
objects in `Te2`. None of these may be resolved using confirmatory test
results.

Raw data remains outside this repository and is excluded by `.gitignore`.
