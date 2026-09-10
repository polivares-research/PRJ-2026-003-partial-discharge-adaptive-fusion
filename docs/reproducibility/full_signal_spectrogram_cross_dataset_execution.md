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

## MATLAB historical-reference repair

The primary global diagnostic intentionally used a fold-local, fixed-epoch
MATLAB temporal path. Its temporal values did not reproduce the historical
reference. To isolate that issue, use the separate reference configuration
below. It reuses the completed VSB/global-spectrogram predictions, opens only
`Tr1.mat` and `Va1.mat`, and writes a new output namespace. It does not rebuild
the global cache and does not open Te1/Te2.

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_full_signal_spectrogram_cross_dataset.py \
  --config configs/experiments/full-signal-spectrogram-cross-dataset-matlab-reference-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage matlab_reference \
  --base-output-root results/audits/full-signal-spectrogram-cross-dataset \
  --log-file logs/full_signal_spectrogram_matlab_reference.log

/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_full_signal_spectrogram_cross_dataset.py \
  --config configs/experiments/full-signal-spectrogram-cross-dataset-matlab-reference-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage evaluation \
  --log-file logs/full_signal_spectrogram_matlab_reference.log

/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_full_signal_spectrogram_cross_dataset.py \
  --config configs/experiments/full-signal-spectrogram-cross-dataset-matlab-reference-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage report \
  --log-file logs/full_signal_spectrogram_matlab_reference.log
```

This reference-only stage records the historical differences explicitly:
nine temporal epochs, full-Tr1 normalization before OOF, and validation-based
epoch selection. Those settings are for regression diagnosis and must not be
silently treated as the fold-local primary protocol.
