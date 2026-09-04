# Local Git handoff

The repository is prepared for review and publication to the internal local
Git history and, after remote compatibility is checked, to GitHub. No raw
dataset is part of the handoff and no force push is permitted.

Before staging, review the worktree and confirm that only intended files are
selected:

```bash
git status --short
git diff --check
git ls-files --others --exclude-standard
git ls-files data/raw
git status --ignored --short data/raw results/cache results/runs
git diff --cached --check
```

The Git policy excludes raw datasets, Parquet/NumPy caches, checkpoints,
per-seed predictions and generated confirmatory reports. The row-level MATLAB
manifest is also excluded because it is regenerated deterministically and is
large; compact manifests, configs, source code, notebooks, documentation and
historical PoC handoff files remain reviewable. The local interrupted run is recorded in
`results/manifests/matlab_execution_status.json`; its ignored caches are not
evidence.

The functional commit sequence is:

1. `data: switch runtime adapters to repository-local raw data`
2. `experiment: register localraw v2 batch4 protocol`
3. `notebooks: bind confirmatory execution to local raw data`
4. `environment: make partial-discharge setup independent of Atlas`
5. `tests: add local data contracts and preflight checks`
6. `docs: document manual data transfer and remote execution`

After reviewing each staged group, create the commits explicitly. Never use
`git add .` because local data and generated outputs may exist:

```bash
git diff --cached --stat
git commit -m "Prepare confirmatory partial-discharge pipeline"
```

Before adding the GitHub remote, inspect its history with `git fetch origin`.
Push only when `main` is empty or compatible with the local history. If the
remote contains unrelated commits, stop for an explicit merge decision; never
merge blindly, overwrite history, or force push. The remote server regenerates
ignored outputs using the frozen localraw config; do not transfer the local
environment directory or raw data through Git.
