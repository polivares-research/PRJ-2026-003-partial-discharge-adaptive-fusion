# Configuraciones

Configuration must separate code from experimental decisions.

- `datasets/`: catalog identifiers, partitions, and relative paths.
- `representations/`: raw waveform, CWT, STFT, wavelet, PSD, and feature parameters.
- `models/`: model families and hyperparameters.
- `experiments/`: reproducible dataset, representation, and model combinations.
- `paths/`: local path configuration and examples without personal paths.
- `runtime/`: seeds, devices, logging, and resource limits.

A future run should be expressible conceptually as:

```bash
run_experiment --config configs/experiments/vsb_raw_baseline.yaml
```

The command is not implemented yet.
