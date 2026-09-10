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
confirmatory command is valid. A confirmatory command must use the generated
`frozen_config.yaml`, must select only the eligible dataset, and must write
under the versioned output namespace; the runner refuses a non-frozen config.

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

