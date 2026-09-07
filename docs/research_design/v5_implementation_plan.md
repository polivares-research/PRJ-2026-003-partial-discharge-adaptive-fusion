# V5 implementation plan: pulse-aware VSB fusion

## Decision ledger

| Decision | Class | Status | V5 record |
|---|---|---|---|
| Preserve PD/NonPD, MATLAB partitions, signal-level loss, five confirmatory seeds, MCC, paired statistics, and separate dataset reports. | SCIENTIFIC | INHERITED | V3/V4 invariants remain unchanged. |
| Use `data/raw` through `PD_RAW_DATA_ROOT`. | COMPUTATIONAL | INHERITED | The requested Dropbox workspace is not mounted and `researchdata` is not installed in `partial-discharge`; both remain `UNRESOLVED`. |
| Keep MATLAB at 400 samples with no architecture search. | SCIENTIFIC | INHERITED | MATLAB is the stable reference; `Te2` remains locked until freeze. |
| Localize VSB observations around deterministic phase-aligned pulses. | SCIENTIFIC | PROPOSED | V5 pulse candidates are evaluated only on development data. |
| Describe the temporal model as Dual-CyCon-informed rather than equivalent or exact. | DOCUMENTATION | OBSERVED | Current data uses signal-level labels and differs from the paper's measurement-level any-phase unit; exact code/training provenance is unavailable. |
| Reuse only external literature ideas after provenance/license/security review. | DOCUMENTATION | PROPOSED | No external weights are allowed. |

## VSB representation

Each labelled VSB signal remains one statistical unit and retains its
`signal_id`, `id_measurement`, phase, label, and split. The preparation sequence
is:

1. Smooth with a 10,000-sample moving average and identify a rising/falling
   zero-crossing pair near one half-cycle apart.
2. Circularly align the signal to the rising crossing.
3. Apply the paper-inspired alpha=100, beta=1 first-order flattening.
4. Detect absolute local peaks, rank them, and apply the deterministic `L=9`
   knee rule.
5. Extract 128-sample temporal and 512-sample CWT views around real peaks.
   Segment values are never padded; only absent bag entries are zero-filled and
   marked invalid.

The predeclared candidates are the V4 uniform-window control, `Np=86` with
32-scale CWT, `Np=86` with 64-scale CWT, and `Np=257` with 64-scale CWT. All
candidate losses and outputs are at parent-signal level.

## Gate and confirmation

Seeds 42--44 are development-only. Calibration, thresholds, fixed weights, and
reliability features use training OOF predictions. A candidate is frozen only
when mean best-expert MCC is at least 0.70, every seed's best MCC is above 0.60,
and the second expert's mean MCC is at least 0.65. Any safety-floor failure is
`STOP`; intermediate performance is `REVIEW_DO_NOT_FREEZE`.

After freeze, seeds 42--46 execute independently as experts, OOF, calibration,
fusion, evaluation, and paired statistics. The primary static claim is
`best_fixed - validation_selected_best_individual`; adaptive deltas are reported
separately against both baselines.

## Protected boundaries

The VSB official test and MATLAB `Te2` cannot be read during selection. V3/V4
artifacts remain historical and immutable. V5 caches, checkpoints, predictions,
and generated reports are written under versioned ignored paths and are never
committed as source artifacts.
