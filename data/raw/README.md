# Local raw-data drop zone

This directory is intentionally empty in Git. Copy the raw datasets here
manually on the execution server; do not commit or push the data.

The runtime reads only this layout, or the directory supplied through
`PD_RAW_DATA_ROOT`:

```text
data/raw/
├── dataset-pd-noise/
│   ├── Tr0.mat  Va0.mat  Te0.mat
│   ├── Tr1.mat  Va1.mat  Te1.mat  Te2.mat
│   └── auxiliary metadata from the source dataset (optional)
└── dataset-vsb-power-line-fault-detection/
    ├── train.parquet
    ├── metadata_train.csv
    └── optional unlabeled test artifacts
```

Required files are listed in [`manifest.example.yaml`](manifest.example.yaml).
The scientific VSB evaluation uses `train.parquet` and `metadata_train.csv`
only; the official unlabeled `test.parquet` is not required.

From a machine that still has the canonical datasets catalog, the optional
`scripts/stage_local_raw_data.py` utility can copy the files into this layout.
The catalog is not required after staging and is never added to the runtime
path on the destination server.
