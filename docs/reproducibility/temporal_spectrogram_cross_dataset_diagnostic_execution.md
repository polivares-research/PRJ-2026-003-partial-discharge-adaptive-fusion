# Temporal vs Spectrogram Diagnostic Execution

Run from the repository root on branch `diagnostic/temporal-spectrogram-cross-dataset`.

```bash
git switch diagnostic/temporal-spectrogram-cross-dataset
export PD_RAW_DATA_ROOT="$PWD/data/raw"
export MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba

/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_temporal_spectrogram_cross_dataset_diagnostic.py \
  --config configs/experiments/temporal-spectrogram-cross-dataset-diagnostic-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage preflight
```

Only after preflight passes, continue with `cache`, `experts`, `evaluation`, and `report`, or use `--stage all`. Completion markers make stages resumable. The runner uses `num_workers=0` and refuses to open forbidden partitions or holdouts.

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_temporal_spectrogram_cross_dataset_diagnostic.py \
  --config configs/experiments/temporal-spectrogram-cross-dataset-diagnostic-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage all

/opt/micromamba/bin/micromamba run -n partial-discharge python -m pytest -q
```

If a stage fails, preserve its log and rerun that stage after the cause is fixed. Do not delete V3/V4/V5 artifacts or use existing prediction files that contain holdout rows.
