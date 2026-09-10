# Current-Expert Adaptive Fusion Execution

Run from the repository root on the registered branch:

```bash
git switch feature/current-expert-adaptive-fusion
export PD_RAW_DATA_ROOT="$PWD/data/raw"
export MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba
```

Use the project launcher:

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_current_expert_adaptive_fusion.py \
  --config configs/experiments/current-expert-adaptive-fusion-localraw.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --stage preflight \
  --log-file logs/current_expert_adaptive_fusion.log
```

Then execute the development stages in order:

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge python scripts/run_current_expert_adaptive_fusion.py --config configs/experiments/current-expert-adaptive-fusion-localraw.yaml --raw-root "$PD_RAW_DATA_ROOT" --stage matlab_clean --log-file logs/current_expert_adaptive_fusion.log
/opt/micromamba/bin/micromamba run -n partial-discharge python scripts/run_current_expert_adaptive_fusion.py --config configs/experiments/current-expert-adaptive-fusion-localraw.yaml --raw-root "$PD_RAW_DATA_ROOT" --stage development_fusion --log-file logs/current_expert_adaptive_fusion.log
/opt/micromamba/bin/micromamba run -n partial-discharge python scripts/run_current_expert_adaptive_fusion.py --config configs/experiments/current-expert-adaptive-fusion-localraw.yaml --raw-root "$PD_RAW_DATA_ROOT" --stage freeze --log-file logs/current_expert_adaptive_fusion.log
/opt/micromamba/bin/micromamba run -n partial-discharge python scripts/run_current_expert_adaptive_fusion.py --config configs/experiments/current-expert-adaptive-fusion-localraw.yaml --raw-root "$PD_RAW_DATA_ROOT" --stage report --log-file logs/current_expert_adaptive_fusion.log
```

`--stage all` is equivalent to these development stages. It does not open any
holdout. If a dataset fails its gate, the freeze stage stops and no
confirmatory command is valid. After freeze, create a fresh expert handoff
with five-seed `train_oof` rows and the holdout rows only for the eligible
dataset. It must contain one temporal and one global-spectrogram row per
signal. Then evaluate the handoff explicitly:

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge python scripts/run_current_expert_adaptive_fusion.py \
  --config results/audits/current-expert-adaptive-fusion/frozen_config.yaml \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --confirmatory-source results/audits/current-expert-adaptive-fusion/confirmatory_expert_handoff.parquet \
  --stage confirmatory \
  --log-file logs/current_expert_adaptive_fusion_confirmatory.log
```

The runner validates the handoff, fits the registered fusion components from
all-development OOF rows for each seed, and scores the eligible holdout.
Expert generation itself must be completed by the frozen post-freeze expert
runner; the development `all` command never opens a holdout. The
confirmatory stage rejects a non-frozen config, missing seeds, unpaired IDs,
and holdouts from an ineligible dataset.

Before any full run, the unit tests can be run without raw-data access:

```bash
MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba \
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python -m pytest -q tests/unit/test_current_expert_fusion.py
```

Resume is safe because each completed stage has a JSON payload and a
`.complete` marker. Do not delete or edit those markers manually; an
incompatible source or configuration must be written to a new output
namespace.
