# Data policy

Original datasets are not stored inside this repository. The canonical local
source is `datasets_catalog`, configured through:

```bash
export PD_DATASETS_CATALOG_ROOT=/path/to/datasets_catalog
```

Project-internal paths are relative to the catalog. `data/external/` may contain
local symlinks, but they are not versioned. `data/interim/` and `data/processed/`
contain regenerable products and remain ignored.

Small manifests and metadata are versioned in `configs/datasets/`.
