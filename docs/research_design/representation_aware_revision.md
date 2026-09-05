# Representation-aware MATLAB/VSB revision

## Scientific reason for the amendment

The historical VSB pipeline fed each 800,000-sample signal to a small CNN
with global pooling. That geometry is not comparable to MATLAB's 400-sample
signals: short transient events can be diluted across the native VSB sequence.
The historical VSB outputs therefore remain diagnostic only and are marked:

> INVALID FOR FINAL SCIENTIFIC INTERPRETATION DUE TO INPUT-REPRESENTATION MISMATCH

The revision changes the VSB representation while preserving the objective,
the two CNN families, grouped measurement split, OOF protocol, five seeds,
MCC, fusion families and paired statistics.

## Common abstraction and legitimate dataset adaptations

Every parent signal is treated as one bag:

```text
parent signal -> K deterministic windows -> shared CNN encoder -> pooling -> one signal probability
```

MATLAB uses `K=1` and its inherited 400-sample signal. VSB uses end-aligned,
unpadded windows. The loss, labels, IDs and evaluation remain at parent-signal
level; a positive window is never promoted to an independent positive example.
All VSB phases retain their `id_measurement`, and no group crosses a split.

The Morlet CWT remains complex with `w0=6`, 32 scales from 1.5 to 64 sample
units, 120 time bins and log power. VSB applies this transform per window and
records the derived 40 MHz pseudo-frequency range. Standardization is fitted
only on training windows.

## Predeclared VSB selection

`configs/experiments/two-dataset-vsb-window-selection-localraw.yaml` declares
the candidates. Candidate A is the historical native diagnostic and is not
eligible for the new selection. Candidates B, C and D are evaluated with seed
42 on development train/validation data only. The rule is mean expert MCC,
then minimum expert MCC within 0.005, then lower storage/compute cost.

Run selection from the repository root:

```bash
export PD_RAW_DATA_ROOT="$PWD/data/raw"
MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba \
  micromamba run -n partial-discharge \
  python scripts/select_vsb_window_protocol.py
```

The script writes a small selection manifest and creates
`configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml` only
after the winner is selected. It never opens MATLAB `Te2`, the grouped VSB
holdout or the unlabeled VSB official test.

## Confirmatory execution order

After the v3 YAML exists, run the two expert stages and then fusion:

```bash
python scripts/run_representation_aware_experts.py --dataset both \
  --config configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml
python scripts/run_representation_aware_fusion.py \
  --config configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml
```

Outputs are versioned below `results/runs/v3-windowed/`,
`results/cache/v3-windowed/` and `reports/{metrics,statistical_analysis}/v3-windowed/`.
The cache metadata includes dataset/version, window, stride, pooling, wavelet,
scales, bins, normalization and preprocessing version. Raw data, caches,
checkpoints and archives are not version-control artifacts.

## Interpretation boundary

Reliability uses OOF correctness targets and validation-only model/threshold
selection. Window statistics are labelled `PROPOSED DUE TO MULTI-INSTANCE VSB
REPRESENTATION` and are retained only when they improve development reliability
selection. MATLAB and VSB are summarized separately; raw samples are never
pooled across datasets. The final verdict must be one of the five registered
outcomes in the frozen v3 record.
