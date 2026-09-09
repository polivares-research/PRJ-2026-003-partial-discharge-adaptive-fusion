# VSB Modality Contribution Diagnostic

## Status and scope

This registered diagnostic runs on branch `diagnostic/vsb-modality-contribution`, based on commit `04f54a3`. It is development-only and is not V6, a neural-model search, fusion, or confirmatory holdout evaluation. It reuses immutable outputs of the VSB literature forensic audit and never opens the grouped VSB test, the official unlabeled test, or MATLAB Te2.

**OBSERVED / SCIENTIFIC.** The audit reports valid data identity and alignment, 6,972 accessible development signals, and a group-safe HGB baseline around MCC 0.626. The primary unit remains the original phase signal; `id_measurement` is the grouping unit.

**INHERITED / COMPUTATIONAL.** Data are addressed through `PD_RAW_DATA_ROOT` with the repository-local default `data/raw`. Dropbox and `researchdata` are forbidden runtime dependencies.

**PROPOSED / SCIENTIFIC.** The frozen 153-predictor forensic baseline is decomposed into shared/context (`S`, 113), temporal/pulse (`T`, 32), and CWT (`C`, 8) families. The only primary model is the fixed `HistGradientBoostingClassifier`; `LogisticRegression` is secondary robustness evidence.

## Data and source locks

Full labeled VSB has 8,712 signals, 2,904 measurements, 525 positive signals, and 38 mixed-label measurements. The accessible development pool has 6,972 signals and 2,324 measurements: train 5,229 / 1,743 measurements / 313 positives and validation 1,743 / 581 measurements / 106 positives. The grouped test has 1,740 signals / 580 measurements / 106 positives and remains locked.

The audit alignment status is **VALID**. The diagnostic requires the four forensic completion markers and exact SHA-256/fingerprint locks recorded in `source_artifact_manifest.json`. Exactly 68 rows may lack all 36 `dualcycon_strict_*` features; these are registered structural zeros. Any other non-finite predictor, duplicate source ID, split leakage, missing phase, or holdout row is a global STOP.

## Frozen factor decomposition

`S` contains detector candidate/selected/noise/boundary features, four quadrant counts, twenty phase-bin counts for each of the four audited detectors, and `cwt_event_count`. `T` contains detector score, SNR, polarity, spacing, and width summaries. `C` contains only the eight existing bounded CWT summaries. Metadata, labels, raw summaries, morphology assignments, noise tables, and detector-match diagnostics are excluded.

The canonical feature order is inherited from the successful `detectors_plus_cwt` baseline. `S+T+C` is constructed by filtering that order, never by concatenating independently ordered lists. Feature sets are `S`, `S+T`, `S+C`, and `S+T+C`, with dimensions 113, 145, 121, and 153. Measurement-aware variants pivot each selected family into fixed phase order `0,1,2` while preserving the original signal target, producing 339, 435, 363, and 459 columns.

## Fitting and statistics

For every seed 42, 43, and 44, train/validation identities are fixed by the forensic manifest. Medians, scaling, OOF thresholds, and classifiers are fit from train only. Thresholds are selected from five-fold `StratifiedGroupKFold` OOF probabilities over 0.05–0.95 in increments of 0.005. HGB settings are fixed: 200 iterations, learning rate 0.05, 15 leaf nodes, L2 regularization 1.0, and the seed. Logistic regression uses train-only standardization and balanced class weights.

The combined baseline must reproduce every historical HGB MCC within `1e-9` and threshold within `1e-12` before any decomposition is interpreted. Required deltas are ΔT, ΔC, ΔC|T, and ΔT|C. Ten-thousand replicate paired bootstrap intervals resample complete `id_measurement` clusters. A hierarchical interval independently resamples groups within each seed and averages the three seed deltas. Random signal-level splits are secondary protocol diagnostics only.

Prediction overlap compares `S+T` with `S+C` using both-correct, temporal-only, CWT-only, and both-wrong strata for all, PD, and NonPD signals, plus disagreement, oracle MCC, and headroom.

## Verdict rules

CWT is strongly complementary when mean ΔC|T is at least +0.010, at least two seeds are positive, and no seed is systematically negative; stronger evidence additionally requires a positive aggregate lower confidence bound. It is weakly complementary in [+0.005,+0.010) with at least two positive seeds, redundant when the absolute mean is below 0.005 or fewer than two seeds are positive, and harmful at or below -0.005 with at least two negative seeds. Independent MCC strength is strong at >=0.60, meaningful at 0.50–<0.60, weak at 0.40–<0.50, and very weak below 0.40.

The runner emits only the registered executive verdicts: `TEMPORAL+CWT SUPPORTED`, `CWT COMPLEMENTARY BUT WEAKER`, `CWT REDUNDANT`, `CWT NOT SUPPORTED`, `TEMPORAL NOT SUPPORTED`, or `REPRESENTATION DECOMPOSITION INCONCLUSIVE`. No result opens a holdout or establishes an adaptive-fusion claim.

## Outputs and acceptance

Generated outputs are isolated under `results/audits/vsb-modality-contribution/` and focused figures under `reports/figures/vsb-modality-contribution/`. Curated report, JSON, and manifest are under `reports/audits/`. Acceptance requires all source locks, tests, baseline regression, feature partition, group isolation, finite probabilities, one validation prediction per original signal/configuration, deterministic group bootstrap, and report schema checks to pass.

Generated raw-derived tables, predictions, figures, caches, checkpoints, archives, and raw data are not committed. Only small curated results may be committed after execution.
