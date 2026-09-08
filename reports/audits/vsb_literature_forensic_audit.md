# VSB Literature Forensic Audit

## Decision

Final audit decision: GO. This is a development-only diagnostic, not V6, not model search, and not confirmatory holdout evaluation.

## Observed

- Audit version: vsb-literature-forensic-v1
- Data contract: local data/raw through PD_RAW_DATA_ROOT.
- Integrity status: PASS
- Development signals: 8712
- Development measurements: 2904
- Positive phase signals: 525
- Any-positive measurements: 194
- Mixed-label measurements: 38
- Alignment status: VALID
- Best grouped phase-independent diagnostic MCC: 0.6320
- Selected group-safe classifier: hist_gradient_boosting; mean MCC across seeds: 0.6262; minimum seed MCC: 0.6194

### Three-phase patterns

pattern  measurements  proportion
    000          2710    0.933196
    001            12    0.004132
    010             1    0.000344
    011             3    0.001033
    100             6    0.002066
    101            10    0.003444
    110             6    0.002066
    111           156    0.053719

### Baseline results

            classifier  seed  grouped       diagnostic_protocol                     mode           feature_family      mcc  roc_auc
hist_gradient_boosting    43    False random_signal_level_split        phase_independent       detectors_plus_cwt 0.721605 0.981092
hist_gradient_boosting    42    False random_signal_level_split        phase_independent       detectors_plus_cwt 0.705302 0.979191
hist_gradient_boosting    44     True                       NaN        measurement_aware       detectors_plus_cwt 0.689158 0.964846
hist_gradient_boosting    42     True                       NaN        measurement_aware       detectors_plus_cwt 0.662283 0.964846
hist_gradient_boosting    43     True                       NaN measurement_any_positive measurement_any_positive 0.659823 0.960450
hist_gradient_boosting    43     True                       NaN        measurement_aware       detectors_plus_cwt 0.658111 0.964846
hist_gradient_boosting    44    False random_signal_level_split        phase_independent       detectors_plus_cwt 0.658049 0.976917
hist_gradient_boosting    42     True                       NaN measurement_any_positive measurement_any_positive 0.636393 0.960450
hist_gradient_boosting    43     True                       NaN        phase_independent       detectors_plus_cwt 0.632050 0.953597
hist_gradient_boosting    42     True                       NaN        phase_independent       detectors_plus_cwt 0.627081 0.953597
hist_gradient_boosting    44     True                       NaN measurement_any_positive measurement_any_positive 0.626708 0.960450
hist_gradient_boosting    44     True                       NaN        phase_independent       detectors_plus_cwt 0.619428 0.953597
   logistic_regression    43    False random_signal_level_split        phase_independent       detectors_plus_cwt 0.615546 0.958742
   logistic_regression    42    False random_signal_level_split        phase_independent       detectors_plus_cwt 0.573791 0.947980
   logistic_regression    44    False random_signal_level_split        phase_independent       detectors_plus_cwt 0.528325 0.944572
   logistic_regression    42     True                       NaN measurement_any_positive measurement_any_positive 0.524698 0.922083
   logistic_regression    44     True                       NaN measurement_any_positive measurement_any_positive 0.512722 0.922083
   logistic_regression    43     True                       NaN measurement_any_positive measurement_any_positive 0.500585 0.922083
   logistic_regression    43     True                       NaN        measurement_aware       detectors_plus_cwt 0.497143 0.909317
   logistic_regression    42     True                       NaN        measurement_aware       detectors_plus_cwt 0.476886 0.909317

## Interpretation

The primary unit is the original phase signal. Grouped train/validation partitions preserve id_measurement. Measurement-aware and any-positive measurement diagnostics are secondary and do not replace the signal-level target. The reported decision MCC excludes rows with grouped=False and diagnostic_protocol=random_signal_level_split. Random signal-level splits are intentionally optimistic and are not deployable evidence.

Phase-derived quantities are called electrical phase only when the predeclared alignment criteria pass. Otherwise they are normalized temporal positions. The detector and CWT settings marked as project literature anchors are adaptations, not exact reproductions. The CWT output is bounded event summaries, not a full scalogram cache.

## Unresolved

- The literature measurement-positive count and repository phase-signal labels use different analytical units.
- Dropbox is unavailable in the active checkout.
- The exact external provenance of Michau and Chen parameter anchors was not independently verified during this audit.
- No external weights, researchdata package, or holdout signals were used.

## Root-cause ranking

1. Measurement-unit mismatch and mixed-label groups can make literature-level and signal-level claims non-equivalent.
2. An invalid or ambiguous cycle reference prevents physical phase interpretation.
3. Low grouped baseline performance would indicate insufficient stable structure in these label-free representations, not failed raw-data identity.
4. Any random-split gain is protocol-dependent optimism if measurement groups cross partitions.

## Scientific consequence

This verdict only controls whether a later representation revision is justified. It does not confirm adaptive fusion and does not open locked holdouts. Detailed provenance is in the JSON and stage outputs.

## Figures

- reports/figures/vsb-literature-forensic/measurement_label_patterns.png
- reports/figures/vsb-literature-forensic/baseline_mcc_by_mode.png
