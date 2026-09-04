# Partial-Discharge Adaptive Fusion

Clean repository for binary partial-discharge classification:

- `PD`: partial discharge.
- `NonPD`: noise or absence of partial discharge.

The study compares heterogeneous representation/model families, using MCC as
the primary metric, and prepares a future reliability-based adaptive fusion.

## Datos

Raw datasets are not duplicated in this repository. On an execution server,
copy them manually into `data/raw` (or set `PD_RAW_DATA_ROOT`):

```bash
export PD_RAW_DATA_ROOT="$PWD/data/raw"
```

See [data/README.md](data/README.md) and [environment/SETUP-REMOTE.md](environment/SETUP-REMOTE.md).

## Current status

The confirmatory pipeline is implemented in `src/` and exposed through the
ordered notebooks under `notebooks/`. The data audits, grouped manifests and
frozen protocol are recorded under `results/manifests/` and
`configs/experiments/`. Full neural training has not been accepted from this
machine because the small local GPU caused instability. The amended batch-4
protocol is prepared for execution on a larger remote GPU; the original v1
native VSB temporal case with batch 128 requires substantially more memory.
See [the GPU budget](docs/reproducibility/gpu_memory_budget.md) and
[the remote setup guide](environment/SETUP-REMOTE.md).

## Organization

- `configs/`: declarative configuration for datasets, representations, models, and experiments.
- `data/`: local references and derived products ignored by Git.
- `docs/`: scientific design, reproducibility, and migration.
- `paper/`: canonical manuscript and submission materials.
- `reports/`: metrics, tables, figures, and statistical analysis.
- `src/`: reusable scientific code for adapters, representations, experts, fusion and evaluation.
- `tests/`: data contracts, unit tests, and integration tests.

The repository does not track raw data, representation caches, checkpoints or
per-run predictions. Those outputs are ignored by Git and must be regenerated
on the destination server with the selected frozen configuration. The active
low-memory variant is `two-dataset-confirmatory-v2-batch4-localraw.yaml`; the
catalog-backed v2 YAML remains as a historical reference and v1 remains
available for a larger GPU.

## Initial scientific rules

- `Te2` is reserved for final generalization to unseen objects.
- In VSB, all three phases of an `id_measurement` must remain in the same split.
- Preprocessing and threshold decisions must use training/validation only.
- Every run must record dataset, split, representation, model, seed, imbalance strategy, and configuration version.
- Notebooks are exploratory and, if created, must use `NN-por-description.ipynb`.

## Legacy

The original legacy project is strictly read-only. Its resolved source path
and transfer provenance are documented in
[docs/migration/researchhub-transfer.md](docs/migration/researchhub-transfer.md)
and
[docs/migration/legacy_inventory.md](docs/migration/legacy_inventory.md).
