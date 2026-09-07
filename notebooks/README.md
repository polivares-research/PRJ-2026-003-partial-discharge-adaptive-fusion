# Notebooks

Notebooks are exploratory and are not the main source of truth for the
pipeline. If one is created, it must follow the Cookiecutter convention:

```text
NN-por-descripcion.ipynb
```

The confirmatory stage is organized into data, expert, fusion, evaluation and
results notebooks under `01-data/` through `05-results/`. They call reusable
code from `src/`; they do not contain the scientific implementation or hidden
dataset paths.

The active execution path is the frozen, generated
`configs/experiments/two-dataset-confirmatory-v3-windowed-localraw.yaml`
configuration. Run the development-only window selector before opening this
file. The expert notebooks are dry wrappers by default and require
the `partial-discharge` kernel, `PD_RAW_DATA_ROOT` (default `data/raw`), CUDA
and `PD_RUN_EXPERIMENT=1` to launch training. The original v1 batch-128 path
and native VSB v2 YAML/results are kept separately as historical references.


## V4 development gate

The V4 notebooks describe the local `data/raw` interface, geometry-aware representations, the development-only expert gate, cache provenance, runtime timing, and the best-individual fusion comparison. They are documentation and reporting wrappers; reusable implementation remains in `src/` and `scripts/`.
