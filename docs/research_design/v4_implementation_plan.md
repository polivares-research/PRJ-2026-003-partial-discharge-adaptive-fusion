# V4 Stronger Representation-Aware MATLAB/VSB Fusion Study

## Decision ledger

- **[OBSERVED · SCIENTIFIC]** V3 is representation-aware and executable, but VSB adaptive fusion is not confirmed. V3 VSB CWT is approximately 0.506 MCC, best fixed fusion approximately 0.519, and adaptive fusion approximately 0.515.
- **[INHERITED · SCIENTIFIC]** Preserve PD/NonPD, MATLAB source partitions, VSB `id_measurement` grouping, parent-signal supervision, OOF, five confirmatory seeds, MCC, paired bootstrap, McNemar, and separate dataset reporting.
- **[INHERITED · COMPUTATIONAL]** Runtime data is local `data/raw`, optionally overridden by `PD_RAW_DATA_ROOT`. V4 introduces no `researchdata` runtime dependency.
- **[UNRESOLVED · DOCUMENTATION]** The requested Dropbox workspace is unavailable. The active checkout is the effective workspace and must be recorded in provenance.
- **[PROPOSED · SCIENTIFIC]** VSB development gating uses seeds 42–44. VSB-trained public weights are forbidden.

## Scientific design

V4 retains MATLAB's 400-sample `K=1` representation and VSB parent-level labels with deterministic end-aligned, unpadded windows. The VSB policy uses 16,384-sample windows with 8,192 stride and top-10% mean aggregation. The CWT remains complex Morlet (`w0=6`, 32 scales, 120 time bins, log-power) and is computed per window.

The bounded development ablation compares the V3-style encoder control with a geometry-adapted pair: a residual/dilated multi-scale temporal CNN with local/coarse context and a residual CWT CNN. No window is treated as an independent labelled sample.

The local VSB audit records whether physical phase alignment is actually available. Cycle-consistency remains disabled unless the raw data and preprocessing contract prove that alignment. Dual-CyCon is treated as literature inspiration or a separately documented reproduction candidate; its reported MCC is not a direct benchmark for V4.

## Gate and evaluation

For each candidate and seeds 42–44, thresholds are fitted from training OOF predictions and validation performance is measured without opening the grouped holdout. A mean best-individual validation MCC of at least 0.70 and no seed at or below 0.60 is required to freeze. Mean MCC at or below 0.60 is `STOP`; `(0.60, 0.70)` is `WEAK — DO NOT FREEZE`; 0.75–0.80 is preferred.

After freeze, seeds 42–46 execute independently: experts → OOF → calibration/reliability → fusion → evaluation. The primary static contrast is `best_fixed − best_individual`; adaptive contrasts are `adaptive − best_fixed` and `adaptive − best_individual`. Paired bootstrap and exact McNemar testing are performed per seed on identical signals.

## Computational design

V4 caches train-only standardizers, raw-window mappings, temporal arrays, and CWT arrays under `results/cache/v4-windowed`. Cache metadata records raw provenance, split-manifest information, window policy, transform parameters, normalizer fingerprint, dtype, library versions, and preprocessing version. Memmaps are written atomically and incomplete artifacts are ignored. Valid artifacts are reused across seeds.

The V4 executor records stage durations. Optimizations such as bounded streaming, pinned transfers, worker/prefetch settings, and optional mixed precision are acceptable only after shape, ID, numerical, and prediction-equivalence checks.

## Acceptance

V4 is accepted only when unit/integration tests pass, the gate passes before holdouts open, the frozen five-seed pipeline completes independently, the primary and secondary comparisons exist for both datasets, and every scientific claim is traceable to a versioned configuration, split manifest, prediction table, and report. Raw data, caches, checkpoints, archives, and generated large outputs remain outside Git.
