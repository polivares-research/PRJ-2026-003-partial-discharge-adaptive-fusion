# Scripts

Scripts are pipeline entry points. Scientific logic remains in `src/` and
runtime data are read from `data/raw` through `PD_RAW_DATA_ROOT`.

## Destination-server workflow

From the activated `partial-discharge` environment and repository root:

```bash
export PD_RAW_DATA_ROOT="$PWD/data/raw"
python scripts/preflight_confirmatory.py --audit-only
python scripts/estimate_vram.py --batch-size 4
python scripts/smoke_confirmatory.py
```

After the raw-data audit and protocol review, run the full ordered notebook
execution:

```bash
python scripts/execute_confirmatory_v2_notebooks.py
```

The historical native configuration is
`configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml`.
Both expert scripts use physical batch 4, inference batch 4, and gradient
accumulation 1. The VSB `--io-batch-size` only bounds Parquet reading and does
not alter the neural batch.

Equivalent direct historical expert commands are:

```bash
python scripts/run_matlab_experts.py \
  --config configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml
python scripts/run_vsb_experts.py \
  --config configs/experiments/two-dataset-confirmatory-v2-batch4-localraw.yaml \
  --io-batch-size 4
```

Training requires visible CUDA and is blocked on CPU. Expert execution should
only start after the audit-only preflight and non-training VRAM estimate pass.

## Representation-aware VSB revision

The native VSB v2 outputs are historical diagnostics and are not valid for the
final scientific interpretation. The current workflow is:

```bash
export PYTHONPATH="$PWD/src"
export PD_RAW_DATA_ROOT="$PWD/data/raw"
MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba \
  micromamba run -n partial-discharge \
  python scripts/execute_representation_aware_pipeline.py
```

This executes development-only candidate selection first, freezes the v3 YAML,
then runs the five-seed multi-instance experts and signal-level fusion. To
resume after selection, use `--skip-selection`. The candidate selector never
opens MATLAB `Te2` or the VSB grouped holdout.

## Optional source-machine staging

If the original catalog is available on the source machine, use:

```bash
python scripts/stage_local_raw_data.py \
  --catalog-root /path/to/datasets_catalog \
  --raw-root "$PWD/data/raw"
```

This is the only script that imports `researchdata`, and only on the source
machine. It copies the required MATLAB and VSB files, writes a portable local
inventory, and leaves the raw files ignored by Git. The destination server
does not need Atlas, the catalog, or `researchdata`.
