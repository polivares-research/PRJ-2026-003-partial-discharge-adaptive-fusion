# VSB Literature Forensic Audit Plan

## Purpose

This audit is a diagnostic research stage, not V6 and not a neural-network
architecture search. It investigates why V3, V4, and V5 produced weak VSB
results and whether the local raw data preserve the pulse, phase, noise, and
three-phase phenomena reported by successful VSB approaches.

## Decision ledger

- **[OBSERVED · SCIENTIFIC]** The local VSB metadata contains 8,712 signals,
  2,904 measurements, 525 positive phase signals, and 38 mixed-label
  measurements.
- **[OBSERVED · SCIENTIFIC]** V4 best-development MCC was approximately
  0.406; V5 best-development MCC was 0.2171. Both gates remained closed.
- **[INHERITED · COMPUTATIONAL]** Raw data are accessed only through
  `PD_RAW_DATA_ROOT`, defaulting to `data/raw`. `researchdata` and Dropbox are
  prohibited runtime dependencies.
- **[INHERITED · SCIENTIFIC]** The V5 train/validation pool is development-only:
  5,229 train signals and 1,743 validation signals. The 1,740-signal grouped
  VSB test split and MATLAB `Te2` remain locked.
- **[PROPOSED · SCIENTIFIC]** Simple diagnostic baselines use fixed
  label-free features and train-only OOF threshold selection. No deep model
  training, fusion search, or holdout tuning is allowed.
- **[UNRESOLVED · DOCUMENTATION]** The 525-versus-575 positive-count
  discrepancy cannot be resolved until the local metadata and literature units
  are compared explicitly.

## Repository and branch safety

The audit branch is `audit/vsb-literature-forensic`. The existing untracked
V5 metrics directory and MATLAB manifest are historical read-only inputs. Their
paths, sizes, and SHA-256 values are recorded in the audit provenance and they
are never staged, overwritten, or used as a write destination.

The audit uses the actual available launcher:

```bash
MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba \
/opt/micromamba/bin/micromamba run -n partial-discharge python ...
```

CUDA is checked at startup. The audit does not silently fall back from an
intended GPU operation to CPU; no PyTorch diagnostic is required for the
baseline stages.

## Protected data boundaries

The audit recomputes and verifies the V5 grouped manifest, then uses only
`split in {train, validation}`. All derived rows retain `signal_id`,
`id_measurement`, `phase`, `target`, `split`, and `oof_fold`. Any overlap of
`id_measurement` between train and validation is a global STOP condition.

The VSB grouped test split, the official unlabeled VSB test, MATLAB `Te2`, and
all locked labels are inaccessible to the audit.

## Diagnostic stages

1. Reconstruct V3, V4, and V5 executable preprocessing and record literature
   provenance.
2. Fingerprint local files, parquet schema, metadata labels, signal shape,
   ranges, finite values, missing IDs, duplicate IDs, and three-phase groups.
3. Reproduce the complete `000`–`111` measurement-label pattern table and
   compare phase-level, any-positive measurement-level, and all-positive
   measurement-level counts.
4. Compare current V5, DFT/sinusoidal, and strict crossing phase references;
   report success, agreement, within-measurement consistency, and pathological
   cases without forcing a 120-degree phase relationship.
5. Compare raw, V5 flattening, Butterworth high-pass, and Savitzky-Golay
   background removal; estimate robust per-signal noise floors.
6. Compare four label-free detectors: current V5, Michau anchor, Chen anchor,
   and strict Dual-CyCon anchor. Exact literature values are used only when
   verified; otherwise they are marked as project anchors or adaptations.
7. Quantify pulse count, amplitude, polarity, width proxy, spacing, centering,
   detector agreement, short-window morphology, and phase-resolved density.
8. Compare phase-independent and measurement-aware three-phase feature tables,
   plus a separately labelled any-positive measurement-level diagnostic.
9. Evaluate fixed logistic and HistGradientBoosting baselines under grouped
   and intentional signal-level random diagnostic splits.
10. Measure train-versus-validation domain shift with group-aware AUC.
11. Compare a V4-compatible uniform descriptive representation with pulse
    descriptors and audit compact CWT summaries without training a CWT CNN.
12. Write the numerical Markdown report, JSON summary, artifact manifest, and
    a small set of referenced diagnostic figures.

## Fixed diagnostic definitions

- V5 current: the repository implementation, including its documented
  fallback behavior, alpha=100/beta=1 flattening, 512-sample spacing, and knee
  selection.
- Michau anchor: 5th-order 20 kHz Butterworth residual, absolute peaks,
  40-sample locality, and 6-MAD prominence. This is an audit adaptation unless
  the exact source value is verified.
- Chen anchor: Savitzky-Golay window 99/order 3 residual, 30-sample locality,
  6-MAD prominence, and local recentering. This is an audit adaptation unless
  verified.
- Strict Dual-CyCon anchor: both crossings required, alpha=100/beta=1
  flattening, minimum distance 512, and at most 257 ranked peaks per half.
- Audit pulse segments never wrap or pad at signal boundaries. V5 circular
  boundary behavior is counted separately.
- Detector matching tolerance is fixed at 32 samples.
- Morphology uses normalized short waveforms, PCA, and
  `MiniBatchKMeans(n_clusters=12, random_state=42)`; cluster labels are not
  interpreted as physical pulse types.
- CWT summaries use 512-sample segments, complex Morlet `w0=6`, 32 scales from
  1.5 to 64 samples, and 128 time bins. Full scalograms are not cached.

## Runtime and failure policy

The full raw pass is streamed over all development signals. Figures and
waveform morphology use a deterministic stratified subset; CWT summaries use a
bounded selected-event set. Each stage has a completion marker, input
fingerprint, elapsed-time log, and resumable output.

Dataset identity, signal integrity, and split leakage failures stop the audit.
An individual literature lookup, alignment method, detector, CWT stage, or
figure may fail independently; the failure, exception, and scientific
consequence are recorded and later stages continue when safe.

## Acceptance

The audit is complete only when the self-contained Markdown report includes
actual numerical tables for dataset integrity, label patterns, noise,
detectors, phase bins, three-phase contribution, grouped/random baselines,
domain AUC, and CWT diagnostics. Every major statement is labelled as
`OBSERVED`, `INTERPRETATION`, or `UNRESOLVED`, and the final decision is one of
`GO`, `PIVOT`, `STOP`, or `INCONCLUSIVE`.
