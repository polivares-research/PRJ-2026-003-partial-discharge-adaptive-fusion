# V5 execution order

Run from the repository root in the `partial-discharge` environment with
`PD_RAW_DATA_ROOT` pointing to `data/raw`:

1. Run the Dual-CyCon/data audit script.
2. Prepare the VSB pulse and CWT caches for development signals only.
3. Run the development candidates on seeds 42--44.
4. Inspect the gate report. A STOP or REVIEW result blocks freezing and all
   holdouts remain closed.
5. Freeze the selected candidate only after the gate passes, recording config,
   source, split, cache, environment, and preprocessing fingerprints.
6. Run confirmatory seeds 42--46, then fusion, paired statistics, and separate
   MATLAB/VSB reports.

CUDA is mandatory for training. The initial V5 runtime keeps mixed precision
disabled; it can be enabled only after the deterministic equivalence benchmark
passes. The official VSB test is unlabeled and is excluded from all scientific
metrics.
