# VSB Modality Contribution Diagnostic

## Decision

**Verdict:** CWT NOT SUPPORTED

**V6 recommendation:** V6 should not invest in CWT fusion without a new predeclared representation rationale.

This is a development-only diagnostic. It uses fixed classical baselines, does not train neural models, and never opens locked holdouts.

## Observed

- Data contract: `PD_RAW_DATA_ROOT/data/raw`; immutable forensic artifacts were reused.
- Full labeled VSB: 8,712 signals / 2,904 measurements / 525 positive signals / 38 mixed-label measurements.
- Accessible development: 6,972 signals / 2,324 measurements; train 5,229 and validation 1,743.
- Locked grouped test: 1,740 signals / 580 measurements / 106 positive signals; no rows were loaded.
- Alignment inherited from the audit: **VALID**.
- Feature decomposition: S=113, T=32, C=8; canonical combined frame=153 predictors.
- Combined HGB regression: **PASS**.
- Independent temporal strength: **STRONG** (S+T mean MCC=0.634211).
- Independent CWT strength: **WEAK** (S+C mean MCC=0.483604).
- Conditional CWT status: **HARMFUL** (mean ΔC|T=-0.008025; positive seeds=1/3).

## Primary group-safe HGB results

| mode | feature_family | mean_mcc | min_mcc | max_mcc |
| --- | --- | --- | --- | --- |
| measurement_aware | S | 0.5328226107496491 | 0.5207627195288952 | 0.5388525563600262 |
| measurement_aware | S+C | 0.5444450717416586 | 0.5388525563600262 | 0.5485457270557097 |
| measurement_aware | S+T | 0.6679984891811844 | 0.6646486518283241 | 0.6729320038266781 |
| measurement_aware | S+T+C | 0.6698508145156943 | 0.6581113302240205 | 0.6891576471408442 |
| phase_independent | S | 0.45742635201882414 | 0.4176471730557036 | 0.5032891751945276 |
| phase_independent | S+C | 0.48360387646341946 | 0.4616429218021561 | 0.5034102847732768 |
| phase_independent | S+T | 0.6342114241118163 | 0.6225752546362788 | 0.6512036719499463 |
| phase_independent | S+T+C | 0.6261864547298233 | 0.6194280719688356 | 0.632049896620652 |

## Required modality deltas

| comparison | point_estimate | ci_95 | fraction_gt_zero |
| --- | --- | --- | --- |
| delta_t | 0.1767850720929921 | [0.126125, 0.228057] | 1.0 |
| delta_c | 0.0261775244445953 | [-0.015938, 0.066897] | 0.8882 |
| delta_c_given_t | -0.0080249693819931 | [-0.036749, 0.019906] | 0.2845 |
| delta_t_given_c | 0.1425825782664037 | [0.099555, 0.189100] | 1.0 |

The four rows are ΔT, ΔC, ΔC|T, and ΔT|C. Confidence intervals are hierarchical 10,000-replicate bootstrap intervals over complete `id_measurement` clusters within each independent seed.

## Three-phase contribution

| feature_family | mean_delta | min_delta | max_delta |
| --- | --- | --- | --- |
| S | 0.07539625873082496 | 0.0355633811654985 | 0.1212053833043225 |
| S+C | 0.06084119527823909 | 0.0451354422824329 | 0.07720963455787 |
| S+T | 0.03378706506936804 | 0.0217283318767317 | 0.0420733971920452 |
| S+T+C | 0.043664359785871 | 0.0260614336033684 | 0.0697295751720085 |

These are secondary signal-target results comparing measurement-aware and phase-independent predictions while preserving mixed-label phase targets.

## Prediction overlap and oracle headroom

| seed | disagreement_rate | oracle_mcc | stronger_individual_mcc | oracle_headroom |
| --- | --- | --- | --- | --- |
| 42.0 | 0.0378657487091222 | 0.6896038608503715 | 0.6288553457492239 | 0.0607485151011476 |
| 43.0 | 0.04245553643144 | 0.7212401885456761 | 0.6225752546362788 | 0.0986649339093972 |
| 44.0 | 0.0327022375215146 | 0.7090705630636724 | 0.6512036719499463 | 0.0578668911137261 |

The full all/PD/NonPD state table is in `results/audits/vsb-modality-contribution/prediction_overlap.csv`. Mean oracle MCC=0.706638; mean stronger individual MCC=0.634211; mean oracle headroom=0.072427.

## Protocol sensitivity (diagnostic only)

| feature_family | seed | mcc | group_overlap_count |
| --- | --- | --- | --- |
| S | 42 | 0.6488095325827347 | 1308 |
| S+T | 42 | 0.7002881882035973 | 1308 |
| S+C | 42 | 0.687393075351064 | 1308 |
| S+T+C | 42 | 0.6869360806406193 | 1308 |
| S | 43 | 0.6566502964461061 | 1330 |
| S+T | 43 | 0.6711685823810511 | 1330 |
| S+C | 43 | 0.6742203954553522 | 1330 |
| S+T+C | 43 | 0.678601341852436 | 1330 |
| S | 44 | 0.6491967110059873 | 1293 |
| S+T | 44 | 0.6804869509254883 | 1293 |
| S+C | 44 | 0.6627589019119988 | 1293 |
| S+T+C | 44 | 0.6933209030869852 | 1293 |

These random signal-level splits intentionally allow measurement groups to cross partitions, so they are labelled `DIAGNOSTIC ONLY — NOT DEPLOYABLE / NOT CONFIRMATORY` and are excluded from all scientific verdicts and uncertainty claims.

## Interpretation

The primary comparison is group-safe validation by `id_measurement`, with train-only medians, five-fold grouped OOF threshold selection, and one prediction per original phase signal. Shared context is not sufficient to explain the temporal contribution: S+T substantially exceeds S. The CWT-only family is below the strong threshold, and adding C to S+T decreases mean MCC; the conditional CWT contribution is therefore not supported by this diagnostic.

The result does not establish adaptive fusion, does not justify opening holdouts, and does not replace a preregistered confirmatory study.

## Unresolved

- Locked VSB test, official unlabeled VSB test, and MATLAB Te2 remain inaccessible by policy.
- Literature-inspired detector parameters remain historical audit inputs, not an exact external reproduction.
- The diagnostic evaluates fixed classical feature contributions; it is not a neural representation or fusion search.

## Figures

- reports/figures/vsb-modality-contribution/modality_mcc.png — mean group-safe HGB MCC by feature family
