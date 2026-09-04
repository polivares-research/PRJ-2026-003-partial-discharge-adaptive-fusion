# Confirmatory stage execution

The confirmatory stage runs in the mamba environment named
`partial-discharge`. Runtime data are local files under `data/raw`, or under
the directory supplied through `PD_RAW_DATA_ROOT`. ResearchHub Atlas and
`researchdata` are not runtime dependencies.

For a new server, follow [the complete remote setup guide](../../environment/SETUP-REMOTE.md).

## Preflight

From the repository root after manually copying the datasets:

```bash
mamba activate partial-discharge
export PD_RAW_DATA_ROOT="$PWD/data/raw"
python scripts/preflight_confirmatory.py --audit-only \
  --output results/manifests/preflight_localraw_audit.json
python scripts/estimate_vram.py --batch-size 4 \
  --output results/manifests/gpu_memory_budget_batch4.json
python scripts/smoke_confirmatory.py
```

The audit-only gate checks both registered local datasets, required files,
`pyarrow`, and portable provenance. A full preflight additionally requires
visible CUDA:

```bash
python scripts/preflight_confirmatory.py \
  --output results/manifests/preflight_localraw.json
```

Training is intentionally blocked when CUDA is unavailable; there is no CPU
fallback.

## Frozen protocol and execution

The active configuration is
`configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml`.
It fixes physical batch 4, inference batch 4, and gradient accumulation 1 for
both experts and both datasets. The original catalog-backed v1/v2 YAML files
remain historical references and are not required by this runtime.

The ordered runner executes all confirmatory notebooks, writes their cell
outputs in place, and produces an English execution summary:

```bash
python scripts/execute_confirmatory_v2_notebooks.py
```

The expert notebooks are dry wrappers unless `PD_RUN_EXPERIMENT=1` is set.
The ordered runner sets it after the preflight checks. The runner uses the
`partial-discharge` Jupyter kernel and sets `PYTHONPATH` only to this
repository's `src/`.

## Audit and leakage boundary

The data-audit notebooks may inspect local Parquet metadata and bounded native
VSB signals, but they may not use held-out outcomes to select preprocessing.
`Te2.mat` remains inaccessible through the ordinary MATLAB loader and can be
opened only by the frozen confirmatory loader. No test-derived model,
calibration, threshold, feature, or fusion decision is permitted.

## Outputs and provenance

Manifests belong under `results/manifests/`. Per-seed predictions/features are
Parquet, summaries are CSV/JSON/YAML, and figures belong under `reports/`.
Each result records configuration version, data source, dataset/version,
split, seed, imbalance strategy, runtime information, file metadata where
available, and Git commit. Raw data, caches, predictions, and checkpoints are
ignored by Git.
