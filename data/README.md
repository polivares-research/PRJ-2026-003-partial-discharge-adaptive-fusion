# Data

Raw bytes are never stored in Git. The portable runtime resolves both
datasets from `data/raw`, or from the directory supplied through
`PD_RAW_DATA_ROOT`.

## Required local layout

```text
data/raw/
├── dataset-pd-noise/
│   ├── Tr1.mat
│   ├── Va1.mat
│   ├── Te1.mat
│   └── Te2.mat
└── dataset-vsb-power-line-fault-detection/
    ├── train.parquet
    └── metadata_train.csv
```

The MATLAB auxiliary partitions `Tr0.mat`, `Va0.mat`, and `Te0.mat` are
recommended for the historical audits. VSB `test.parquet` is optional and is
excluded from scientific metrics because the official snapshot has no labels.
See [raw/README.md](raw/README.md) and [raw/manifest.example.yaml](raw/manifest.example.yaml).

## Registered datasets

- `engineering-partial-discharge-noise-signals@v1` — Rauscher et al. MATLAB
  benchmark; source partitions are preserved and `Te2` is confirmatory only.
- `engineering-vsb-power-line-fault-detection@2018-kaggle-snapshot` — VSB
  benchmark; all phases of each `id_measurement` stay in one split.

On the source machine only, `scripts/stage_local_raw_data.py` can copy files
from the catalog into this layout. The destination server does not need
ResearchHub Atlas, `researchdata`, or `PD_DATASETS_CATALOG_ROOT`.
