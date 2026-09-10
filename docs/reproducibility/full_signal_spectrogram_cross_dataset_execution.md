# Full-Signal Spectrogram Diagnostic Execution

Run from the repository root on `diagnostic/full-signal-spectrogram-cross-dataset`.
The implementation turn does not execute the expensive cache, training, OOF, or
bootstrap stages.

```bash
git switch diagnostic/full-signal-spectrogram-cross-dataset
export PD_RAW_DATA_ROOT="$PWD/data/raw"
export MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba

/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_full_signal_spectrogram_cross_dataset.py \
  --config configs/experiments/full-signal-spectrogram-cross-dataset-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage preflight \
  --log-file logs/full_signal_spectrogram_cross_dataset.log
```

After preflight passes, run `cache`, then `experts`, `evaluation`, and
`report`, or use `--stage all`. Stages are resumable and write only to the new
`results/cache/full-signal-spectrogram-cross-dataset/` and
`results/audits/full-signal-spectrogram-cross-dataset/` namespaces.

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_full_signal_spectrogram_cross_dataset.py \
  --config configs/experiments/full-signal-spectrogram-cross-dataset-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage all \
  --log-file logs/full_signal_spectrogram_cross_dataset.log
```

The runner must stop before interpretation if CUDA, forensic source locks,
signal shape, group isolation, cache compatibility, or temporal regression
fails. Do not delete old V3/V4/V5 caches and do not stage raw data, generated
arrays, predictions, checkpoints, figures, or ZIP files.
