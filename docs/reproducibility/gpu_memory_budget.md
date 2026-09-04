# GPU memory budget

This budget was calculated from the frozen architectures and input policy. It
does not run a model on CUDA. The calculation counts FP32 input/intermediate
tensors and reports a planning envelope of three times their total, plus
parameters, gradients and the two Adam state tensors. CUDA context, allocator
fragmentation and cuDNN workspaces are not included.

The original v1 frozen protocol uses batch 128. The new amended
`two-dataset-confirmatory-v2-batch4-localraw.yaml` uses a common physical batch of 4
for every expert and dataset.

| Case at batch 4 | Parameters | Largest traced tensor | Traced FP32 tensors | 3x activation envelope | Payload planning estimate |
|---|---:|---:|---:|---:|---:|
| MATLAB temporal, 400 samples | 13,313 | 0.000 GiB | 0.000 GiB | 0.001 GiB | 0.001 GiB |
| VSB temporal, native 800,000 samples | 13,313 | 0.191 GiB | 0.775 GiB | 2.325 GiB | 2.325 GiB |
| MATLAB CWT, 32×120 | 23,585 | 0.001 GiB | 0.002 GiB | 0.006 GiB | 0.006 GiB |
| VSB CWT, frozen native policy 32×120 | 23,585 | 0.001 GiB | 0.002 GiB | 0.006 GiB | 0.006 GiB |

The exact reproducible calculation is:

```bash
export PYTHONPATH="$PWD/src"
mamba run -n partial-discharge python scripts/estimate_vram.py \
  --batch-size 128 \
  --output results/manifests/gpu_memory_budget_batch128.json
```

For the amended batch-4 experiment:

```bash
export PYTHONPATH="$PWD/src"
mamba run -n partial-discharge python scripts/estimate_vram.py \
  --batch-size 4 \
  --output results/manifests/gpu_memory_budget_batch4.json
```

## Interpretation

The bottleneck is not the number of trainable parameters. It is the native VSB
temporal sequence: the first convolution alone produces a 6.104 GiB FP32
tensor at batch 128, already above the observed 5.68 GiB available on the
local GPU. The complete training envelope is approximately 74 GiB before
driver/framework overhead. A GPU with at least 80 GiB is the practical target
for the original batch-128 protocol; 96 GiB gives useful operational margin.
At batch 4, the same planning envelope scales to approximately 2.325 GiB,
but this symbolic budget excludes CUDA context, allocator fragmentation,
cuDNN workspaces and system instability. The local run is therefore deferred
to a larger remote GPU rather than treated as validated by this estimate.

MATLAB temporal and both CWT experts are not expected to be VRAM-limited by
these architectures. CWT generation is primarily a CPU/RAM/disk operation in
this implementation.

The remote server should also reserve roughly 30 GB for the VSB temporal
memmaps (all three labeled splits) and roughly 13 GB for the complete MATLAB
temporal/CWT cache set, in addition to the external Parquet/MAT artifacts.
The VSB runner reads Parquet in small column batches so these caches do not
require materializing the full 800,000×8,712 matrix in RAM.

If a destination server has 24 GiB or less, it can use the amended batch-4
configuration. Batch reduction is a protocol change, which is why it is
registered in a new versioned configuration and applied to both experts and
both datasets. Gradient accumulation, windowing or resampling would be a
different amendment and are not silently applied.

The interrupted local run produced no expert predictions or checkpoints. Its
partial representation caches are ignored by Git and are not scientific
results.
