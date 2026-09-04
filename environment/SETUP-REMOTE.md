# Recreate `partial-discharge` on a remote GPU server

The repository is executable without ResearchHub Atlas or `researchdata`.
Raw datasets are transferred separately and are never committed to Git.

## 1. Clone and recreate the environment

```bash
git clone git@github.com:polivares-research/PRJ-2026-003-partial-discharge-adaptive-fusion.git
cd PRJ-2026-003-partial-discharge-adaptive-fusion

mamba env create -f environment/partial-discharge.yml
mamba activate partial-discharge
python -m pip install -e .
python -m ipykernel install --user \
  --name partial-discharge --display-name "Python (partial-discharge)"
```

No Atlas checkout or catalog path is required on this server. The optional
`scripts/stage_local_raw_data.py` utility is only for a source machine that
still has access to the original catalog.

## 2. Copy the datasets manually

Copy the files into the following paths, using a separate data-transfer
channel:

```text
data/raw/dataset-pd-noise/{Tr1.mat,Va1.mat,Te1.mat,Te2.mat}
data/raw/dataset-vsb-power-line-fault-detection/{train.parquet,metadata_train.csv}
```

The MATLAB audit also benefits from `Tr0.mat`, `Va0.mat`, and `Te0.mat`.
The unlabeled VSB `test.parquet` is optional and is not used for scientific
metrics. See [../data/raw/manifest.example.yaml](../data/raw/manifest.example.yaml).

Then configure the root (the default is already correct when running from the
repository root):

```bash
export PD_RAW_DATA_ROOT="$PWD/data/raw"
export PD_KERNEL_NAME=partial-discharge
```

## 3. Verify before training

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python scripts/preflight_confirmatory.py --audit-only \
  --output results/manifests/preflight_localraw_audit.json
python scripts/estimate_vram.py --batch-size 4 \
  --output results/manifests/gpu_memory_budget_batch4.json
python scripts/smoke_confirmatory.py
```

The audit-only preflight checks both local dataset directories, all required
files, `pyarrow`, and the portable provenance. A full preflight without
`--audit-only` additionally requires CUDA. Training remains blocked unless
CUDA is visible.

## 4. Execute the confirmatory notebooks

After reviewing the audit and VRAM estimate:

```bash
python scripts/execute_confirmatory_v2_notebooks.py
```

This uses the frozen
`configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml`,
physical batch 4, inference batch 4, and gradient accumulation 1. It executes
the notebooks in their registered order, persists outputs in-place, and writes
an English summary to
`reports/confirmatory_v2_notebook_execution_summary.md` plus JSON.

The expert notebooks are dry wrappers unless `PD_RUN_EXPERIMENT=1` is set.
The execution runner sets that flag for the ordered run. Representation
caches, predictions, checkpoints, raw data, and other large artifacts are
ignored by Git.

## 5. Optional source-machine staging

On a machine where the canonical catalog is already available, stage data
without adding the catalog as a runtime dependency:

```bash
python scripts/stage_local_raw_data.py \
  --catalog-root /path/to/datasets_catalog \
  --raw-root "$PWD/data/raw"
```

Review the resulting inventory, transfer the raw files separately, and run
the destination preflight. Never use `git add -f` for files below `data/`.
