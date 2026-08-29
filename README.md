# Partial-Discharge Adaptive Fusion

Clean repository for binary partial-discharge classification:

- `PD`: partial discharge.
- `NonPD`: noise or absence of partial discharge.

The study compares heterogeneous representation/model families, using MCC as
the primary metric, and prepares a future reliability-based adaptive fusion.

## Datos

Raw datasets are not duplicated in this repository. They are consumed from the
external catalog through `PD_DATASETS_CATALOG_ROOT`:

```bash
export PD_DATASETS_CATALOG_ROOT=/path/to/datasets_catalog
```

See [data/README.md](data/README.md) and
[docs/data_sources/datasets_catalog_snapshot.md](docs/data_sources/datasets_catalog_snapshot.md).

## Current status

This stage contains only structure, documentation, configuration, and
manifests. It does not yet contain experimental code, notebooks, training, or
files migrated from the legacy repository.

## Organization

- `configs/`: declarative configuration for datasets, representations, models, and experiments.
- `data/`: local references and derived products ignored by Git.
- `docs/`: scientific design, reproducibility, and migration.
- `paper/`: canonical manuscript and submission materials.
- `reports/`: metrics, tables, figures, and statistical analysis.
- `src/`: destination for future scientific code.
- `tests/`: data contracts, unit tests, and integration tests.

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
