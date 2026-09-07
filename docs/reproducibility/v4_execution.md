# V4 Reproducibility and Execution

## Runtime

Run from the repository root in the `partial-discharge` environment with CUDA available:

```bash
export PD_RAW_DATA_ROOT="$PWD/data/raw"
MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba \
  micromamba run -n partial-discharge \
  python scripts/execute_v4_pipeline.py \
  --raw-root "$PD_RAW_DATA_ROOT" \
  --io-batch-size 4
```

The executor performs the cycle-reference audit, development gate, protocol freeze, confirmatory experts, and signal-level fusion in that order. It writes stage timing to `results/manifests/v4_execution.json`.

Runtime controls are versioned in the V4 YAML. The default is `num_workers: 0`, `persistent_workers: false`, and `mixed_precision: false`; these conservative defaults preserve V3 behavior. Worker prefetch and mixed precision may be enabled only after the equivalence benchmark passes, with predictions compared against the default path.

## Holdout protection

`run_v4_development_gate.py` reads only labelled VSB train/validation data. The grouped VSB holdout and MATLAB Te2 are inaccessible until `freeze_v4_protocol.py` has produced the frozen YAML. The confirmatory runner rejects a non-frozen configuration.

## Outputs

V4 outputs are namespaced below `results/runs/v4-windowed`, `results/cache/v4-windowed`, and `reports/*/v4-windowed`. V3 outputs are not overwritten. Do not commit `data/`, caches, checkpoints, archives, or generated large manifests.
