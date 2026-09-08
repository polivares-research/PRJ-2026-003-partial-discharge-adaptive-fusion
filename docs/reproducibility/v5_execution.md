# V5 execution order

Run from the repository root in the `partial-discharge` environment with
`PD_RAW_DATA_ROOT` pointing to `data/raw`:

1. Run the Dual-CyCon/data audit script.
2. Prepare the VSB pulse and CWT caches for development signals only.
3. Run the cache-backed development candidates on seeds 42--44:

Command: export PD_RAW_DATA_ROOT=data/raw; MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba micromamba run -n partial-discharge python scripts/run_v5_development_experts.py --config configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml --cache-root results/cache/v5-pulse-aware --results-root results/runs/v5-pulse-aware --records results/runs/v5-pulse-aware/development_records.json --num-workers 0 --log-file logs/v5_development_experts.log

The runner writes parent-signal OOF and validation predictions, fold-local
train-only standardizers, training records, and the JSON consumed by the gate.
The current checkout has no compatible V5 uniform-window cache, so
v4_uniform_control is reported as unavailable and is not silently replaced.
Prepare that cache before claiming a complete control comparison.

4. Run the development gate using the generated records:

Command: MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba micromamba run -n partial-discharge python scripts/run_v5_development_gate.py --config configs/experiments/two-dataset-confirmatory-v5-pulse-aware-localraw.yaml --records results/runs/v5-pulse-aware/development_records.json --output reports/metrics/v5-pulse-aware/development_gate.json --log-file logs/v5_development_gate.log

A STOP or REVIEW result blocks freezing and all
holdouts remain closed.
5. Freeze the selected candidate only after the gate passes, recording config,
   source, split, cache, environment, and preprocessing fingerprints.
6. Run confirmatory seeds 42--46, then fusion, paired statistics, and separate
   MATLAB/VSB reports.

CUDA is mandatory for training. The initial V5 runtime keeps mixed precision
disabled; it can be enabled only after the deterministic equivalence benchmark
passes. The official VSB test is unlabeled and is excluded from all scientific
metrics.
