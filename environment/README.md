# Environment

The managed Atlas environment record remains in `environment.yaml` for
provenance only. The portable experiment recipe is
`partial-discharge.yml`; the remote-server procedure is documented in
[SETUP-REMOTE.md](SETUP-REMOTE.md).

The target runtime is the mamba environment `partial-discharge` with Python
3.11, CUDA-enabled PyTorch, `pyarrow`, `ipykernel`, and the scientific Python
stack. Runtime code does not install or import ResearchHub Atlas or
`researchdata`. Raw files are supplied separately under `data/raw`.

Do not copy the local environment directory or raw datasets into Git. Recreate
the environment from the portable YAML, copy the datasets manually, set
`PD_RAW_DATA_ROOT` if needed, and run the preflight before opening notebooks.
