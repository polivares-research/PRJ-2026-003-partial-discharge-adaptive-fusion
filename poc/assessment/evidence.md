# Evidence

PoC executed with the leakage-safe Tr0/Va0/Te0 protocol.

- Dataset: `dataset-pd-noise@v1`.
- Workspace commit: `acfe24dd316150c3228a23e30fceafc637d1dcbb`.
- Seed/device: `42` / `cpu`.
- Loaded partitions: `Tr0.mat, Va0.mat, Te0.mat`; Te1.mat and Te2.mat were not loaded.
- Best baseline MCC on Te0: `0.9489`.
- Adaptive fusion MCC: `0.9444`.
- Adaptive gain: `-0.0045`; criterion >= 0.005: `False`; automatic recommendation: `REVISE`.

Complete metrics are in `poc/assessment/metrics.json` and `poc/results/summary.json`.
