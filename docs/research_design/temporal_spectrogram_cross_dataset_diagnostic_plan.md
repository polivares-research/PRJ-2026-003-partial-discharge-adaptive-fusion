# Cross-Dataset Temporal vs Spectrogram Diagnostic

## Status and scope

**PROPOSED · SCIENTIFIC.** This development-only diagnostic compares the frozen MATLAB temporal reference and the VSB forensic `S+T` baseline against a natural-log-power STFT representation. It is not V6, a neural search, a confirmatory evaluation, or an adaptive-fusion study.

**INHERITED · COMPUTATIONAL.** Raw data are addressed only through `PD_RAW_DATA_ROOT`, defaulting to `data/raw`. Dropbox and `researchdata` remain unavailable and forbidden. The active checkout is the reproducibility boundary.

**INHERITED · SCIENTIFIC.** MATLAB uses `Tr1.mat` for training/OOF and `Va1.mat` for development validation. VSB uses only the fixed forensic train/validation identities. The VSB grouped test, official unlabeled test, and MATLAB `Te1`/`Te2` are locked.

## Registered representations

MATLAB retains 400-sample `TinyTemporalCNN` inputs and adds one-channel Hann log-power STFTs with `n_fft=128`, `win_length=64`, `hop=16`, `center=False`, and shape `[N,1,65,22]`. Because the local source does not provide a verified sampling rate, its frequency axis is normalized cycles/sample.

VSB retains the frozen 145-predictor forensic `S+T` HGB temporal baseline. The spectrogram path uses only the frozen `v5_current` detector, non-wrapping 512-sample segments from the aligned flattened signal, a capacity of 86 events per half-cycle, bag-only zero padding, and validity masks. Each segment uses Hann log-power STFT with `n_fft=128`, `win_length=128`, `hop=32`, shape `[1,65,13]`; two half-cycles form `[2,86,1,65,13]`. The physical frequency interpretation is 0–20 MHz at 312.5 kHz bins.

Standardization is fitted lazily on train-only data for each relevant fold. Unnormalized arrays are cached as float16 with atomic completion markers and fingerprints. Training uses signal-level labels and one parent-signal probability; no window/event labels are introduced.

## Evaluation and interpretation

Seeds are 42, 43, and 44. Thresholds are selected from train OOF probabilities over 0.05–0.95 by 0.005. The primary metric is MCC. Fixed 50/50 and OOF-selected weight-grid combinations are diagnostics only and never create an adaptive pipeline. Paired bootstrap uses individual signals for MATLAB and complete `id_measurement` clusters for VSB, with a hierarchical three-seed summary.

Before scientific interpretation the runner must pass CUDA, artifact/split/coverage checks, prediction alignment, and temporal regression checks. The report separates **OBSERVED**, **INTERPRETATION**, and **UNRESOLVED**. Dual-CyCon frequency concepts are verified only at high level; Erişti and exact reported literature values remain `PROJECT LITERATURE ANCHOR — NOT REVERIFIED`.

The only permitted executive verdicts are `TEMPORAL+SPECTROGRAM SUPPORTED`, `SPECTROGRAM SUPPORTED ONLY ON VSB`, `SPECTROGRAM SUPPORTED ONLY ON MATLAB`, `SPECTROGRAM NOT COMPLEMENTARY`, `SPECTROGRAM NOT SUPPORTED`, and `INCONCLUSIVE`.

## Acceptance

All generated artifacts stay under V5-diagnostic-specific ignored directories. Only the curated Markdown, JSON, and manifest may be committed after execution. A passing implementation must demonstrate deterministic STFT dimensions, non-wrapping pulse extraction, train-only normalization, one parent output, no holdout access, and reproducible source fingerprints.
