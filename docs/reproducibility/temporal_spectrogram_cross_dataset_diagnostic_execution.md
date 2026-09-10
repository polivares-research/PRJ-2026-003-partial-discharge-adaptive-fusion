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
  --stage preflight \
  --log-file logs/temporal_spectrogram_cross_dataset_diagnostic_v2_repaired.log
```

This repaired configuration writes to `results/audits/temporal-spectrogram-cross-dataset-v2-repaired/` and reuses the immutable raw cache at `results/cache/temporal-spectrogram-cross-dataset/`. The prior report and predictions are historical/provisional and are not overwritten. Standardizers are fitted train-only and fold-locally for OOF; final standardizers are fitted on the complete training subset. Epoch selection is fixed to the registered counts; outer validation labels are not used for checkpoint selection.

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_temporal_spectrogram_cross_dataset_diagnostic.py \
  --config configs/experiments/temporal-spectrogram-cross-dataset-diagnostic-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage all \
  --log-file logs/temporal_spectrogram_cross_dataset_diagnostic_v2_repaired.log

/opt/micromamba/bin/micromamba run -n partial-discharge python -m pytest -q
```

The repair adds OOF-only threshold curves, probability diagnostics, registered 50/50 and OOF-selected fixed mixtures, parent-signal VSB event aggregation summaries, and memory-bounded paired bootstrap outputs. A failed temporal regression check leaves the pipeline auditable but forces the scientific verdict to `INCONCLUSIVE`.

If a stage fails, preserve its log and rerun that stage after the cause is fixed. Do not delete V3/V4/V5 artifacts or use existing prediction files that contain holdout rows. The implementation turn that introduced this repair intentionally does not execute the commands above.
