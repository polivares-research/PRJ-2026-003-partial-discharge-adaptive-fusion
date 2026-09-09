# Temporal vs Spectrogram Diagnostic — Model Evaluation Handoff

Date: 2026-09-09  
Branch: `diagnostic/temporal-spectrogram-cross-dataset`  
Scope: development-only diagnostic; not a confirmatory adaptive-fusion result.

## Executive summary

The diagnostic produced a large MATLAB/VSB performance gap:

- MATLAB spectrogram: mean validation MCC **0.9824** across seeds 42–44.
- VSB spectrogram: mean validation MCC **0.0306** across seeds 42–44.
- VSB frozen temporal reference: mean validation MCC **0.6342**.
- Runner verdict: `SPECTROGRAM SUPPORTED ONLY ON MATLAB`.

The VSB result should not yet be interpreted as evidence that spectrograms are intrinsically invalid for VSB. The implementation audit found protocol and implementation issues that must be resolved first, especially that the VSB spectrogram trainer consumed unnormalized log-power cache values even though the registered protocol requires train-only per-frequency/time-bin standardization. The final-fit helper also selected epochs using the outer validation set, and the runner did not actually enforce the temporal regression gate or execute the registered bootstrap/fixed-mixture analyses.

The current result is therefore best classified as a **diagnostic failure of the VSB spectrogram pipeline**, not a definitive scientific rejection of the representation.

## Data and protocol boundaries

The run used the repository-local contract `PD_RAW_DATA_ROOT=data/raw`. MATLAB used `Tr1.mat` and `Va1.mat`. VSB used the 6,972-signal forensic development pool with grouped train/validation identities. The following remained locked:

- VSB grouped test and official unlabeled test;
- MATLAB `Te1.mat` and `Te2.mat`;
- external weights, Dropbox, `researchdata`, CWT reruns, adaptive fusion, and reliability models.

The cache and expert stages were reused during the final `--stage all` invocation; the last reporting repair reran only evaluation and report generation.

## Observed validation metrics

### MATLAB

| Seed | Temporal MCC | Temporal threshold | Spectrogram MCC | Spectrogram threshold |
|---:|---:|---:|---:|---:|
| 42 | 0.952365 | 0.205 | 0.978287 | 0.365 |
| 43 | 0.854643 | 0.095 | 0.983246 | 0.550 |
| 44 | 0.971041 | 0.155 | 0.985552 | 0.370 |
| Mean | 0.926016 | — | 0.982362 | — |

The spectrogram is strong on MATLAB under the registered threshold (`mean >= 0.95`, all seeds `>= 0.93`). However, the MATLAB temporal regression reference was not reproduced for seed 43: observed MCC `0.854643` versus expected historical reference `0.969980` with tolerance `0.005`. This should have blocked scientific interpretation.

### VSB

| Seed | Temporal MCC | Temporal threshold | Spectrogram MCC | Spectrogram threshold | Spectrogram ROC AUC |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.628855 | 0.265 | -0.040890 | 0.540 | 0.634326 |
| 43 | 0.622575 | 0.105 | 0.132777 | 0.565 | 0.743456 |
| 44 | 0.651204 | 0.195 | 0.000000 | 0.575 | 0.762687 |
| Mean | 0.634211 | — | 0.030629 | — | 0.713490 |

The temporal MCC values reproduce the registered VSB reference. The VSB spectrogram has weak-to-moderate ranking information by ROC AUC, but its thresholded MCC is near zero because the probability calibration/decision threshold is unstable: seed 42 produces few positives, seed 43 predicts nearly every signal positive, and seed 44 predicts no positives.

## What the low VSB MCC establishes

**Observed:** Under the executed implementation, the VSB spectrogram expert did not provide reliable signal-level binary decisions on the fixed development validation split.

**Observed:** The problem is not equivalent to “the STFT contains no information.” VSB ROC AUC values are approximately `0.63–0.76`, while MCC is `-0.041–0.133`. This pattern indicates that ranking information exists but calibration, threshold transfer, training stability, or signal-level aggregation is inadequate.

**Observed:** Performance varies strongly across independent seeds. This is consistent with class imbalance and unstable optimization, but the current run does not isolate the cause.

**Interpretation:** The VSB task is substantially harder than MATLAB because VSB converts each 800,000-sample signal into a masked bag of localized 512-sample event segments. The model must learn from signal-level labels while only a subset of events may contain discriminative evidence. The fixed top-10% aggregation can dilute or amplify a small number of event predictions, and the two half-cycle mean adds another constraint.

## Highest-priority implementation issue

The registered VSB protocol requires:

```text
unnormalized float16 log-power cache
→ train-only per-frequency/time-bin standardizer
→ standardized arrays/views for each fold
→ signal-level training
```

The current runner correctly writes unnormalized VSB log-power arrays to cache, but the VSB expert stage passes those raw arrays directly to `cross_fitted_pulse_logits` and `_pulse_validation_predictions`. The fitted `SpectrogramStandardizer` is not applied in the VSB expert path.

This creates an asymmetry:

- MATLAB spectrogram values are standardized before training.
- VSB spectrogram values are not standardized before training.

This is the first issue to fix and re-evaluate. Until fixed, the VSB MCC result cannot support a representation-level conclusion.

## Other issues that affect interpretation

1. **Outer-validation model selection leakage.** The final VSB spectrogram fit passes the outer validation indices into `train_pulse_expert`, whose epoch selection uses validation MCC. The final MATLAB neural helper has the same pattern through `train_expert`. Epoch selection should use a train-only inner split or a fixed registered epoch count; the outer validation must be used once for final evaluation.

2. **Fold-local normalization is incomplete.** The MATLAB spectrogram standardizer is fitted over all training signals before OOF training rather than separately within each OOF fit. The VSB standardizer is currently not applied at all. The required implementation should fit the normalizer inside each fold and apply it only to that fold’s fit/OOF/validation data.

3. **Temporal MATLAB regression failed but was not enforced.** The report runner currently sets `regression_ok=True` instead of comparing observed values to the registered MATLAB temporal references. The seed-43 mismatch should force `INCONCLUSIVE` until resolved.

4. **Registered statistics were not executed.** The runner produced metrics and overlap/oracle tables but did not produce the planned 10,000-replicate paired bootstrap, hierarchical seed confidence intervals, or fixed 50/50/weight-grid diagnostics.

5. **Modality comparison is not architecture-matched for VSB.** VSB temporal is the proven classical HGB `S+T` baseline, whereas VSB spectrogram is a neural pulse-bag CNN. This is a registered diagnostic choice, but a difference in model family, optimization, calibration, and capacity can contribute to the gap. The result should be described as a pipeline comparison, not a pure representation ablation.

6. **The current verdict is provisional.** The report says `SPECTROGRAM SUPPORTED ONLY ON MATLAB`, but the failed MATLAB temporal regression and missing gate/statistical enforcement mean this must not be treated as a final registered scientific verdict.

## Plausible causes ranked for investigation

### 1. Missing VSB spectrogram standardization — highest confidence

The code path and protocol differ directly. Log-power values have frequency/time-dependent offsets and scale. A compact CNN trained on unstandardized values can be dominated by global energy and cache-level scale rather than stable local morphology. This is a concrete implementation defect, not a hypothesis.

### 2. Calibration and threshold instability

The VSB ROC AUC values are materially above 0.5, but MCC is close to zero. The seed-specific threshold selected from OOF probabilities does not transfer stably to validation. Inspect OOF/validation probability histograms, prevalence, calibration curves, and MCC as a continuous threshold curve before changing architecture.

### 3. Signal-level MIL and fixed top-10% pooling

The label is attached to the parent signal, while event bags may contain mostly non-informative events. Top-10% pooling over 86 events per half-cycle may be too aggressive or too diffuse for the actual event distribution. This must be diagnosed with fixed, predeclared pooling summaries; it must not become an unconstrained pooling search.

### 4. Event-localization mismatch

The VSB STFT is computed only on the `v5_current` flattened local segments. If the detector selects high-amplitude but non-diagnostic events, the spectrogram can be internally well-formed while discarding the relevant signal information. Compare event coverage, pulse counts, boundary rejection, and positive/negative distributions before changing detector parameters.

### 5. Training instability and imbalance

The VSB validation has 106 positive signals among 1,743 signals. Seed-specific all-positive/all-negative tendencies indicate unstable optimization or calibration. Evaluate fixed seven-epoch checkpoints, train-only inner validation, positive weighting, and deterministic probability distributions before interpreting architecture quality.

### 6. Dataset geometry/domain difference

MATLAB uses one 400-sample signal-level spectrogram with a fixed compact CNN. VSB uses two half-cycle bags of up to 86 local spectrograms and masked signal-level aggregation. Strong MATLAB performance does not guarantee that the same local STFT semantics transfer to VSB’s event geometry.

## Recommended next actions for the evaluating model

1. Repair VSB train-only standardization and implement fold-local standardizers without changing the raw cache.
2. Add a hard regression gate for MATLAB temporal values; stop report generation when it fails.
3. Remove outer-validation epoch selection; use a train-only inner split or fixed epochs.
4. Regenerate only VSB spectrogram OOF/validation predictions first; do not rerun raw cache preparation unless cache fingerprints fail.
5. Inspect per-seed probability distributions and threshold curves, not only MCC.
6. Execute the registered fixed-mixture diagnostics and paired bootstrap after prediction alignment passes.
7. Reassess the verdict only after all gates pass. If standardized, leakage-safe VSB spectrogram MCC remains near zero with stable calibration, then the negative representation finding becomes more credible.

## Scientific status

The current output is useful for debugging and model evaluation, but it is **not a publishable confirmatory result**. The VSB spectrogram path has a concrete preprocessing defect, the MATLAB temporal regression gate failed, and the planned uncertainty/fixed-mixture analyses were not completed. No holdout data were opened.

## Source artifacts

- Main diagnostic report: `reports/audits/temporal_spectrogram_cross_dataset_diagnostic.md`
- Machine-readable summary: `reports/audits/temporal_spectrogram_cross_dataset_diagnostic.json`
- Validation predictions: `results/audits/temporal-spectrogram-cross-dataset/validation_predictions.parquet`
- Metrics: `results/audits/temporal-spectrogram-cross-dataset/metrics.csv`
- Overlap/oracle: `results/audits/temporal-spectrogram-cross-dataset/prediction_overlap.csv`
- Configuration: `configs/experiments/temporal-spectrogram-cross-dataset-diagnostic-localraw.yaml`
