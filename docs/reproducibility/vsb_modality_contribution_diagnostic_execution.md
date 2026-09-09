# VSB Modality Contribution Diagnostic Execution

Use the active checkout and the local raw-data contract. Do not switch to `researchdata` or a Dropbox path.

```bash
git switch main
git rev-parse HEAD
git switch -c diagnostic/vsb-modality-contribution 04f54a3
export PD_RAW_DATA_ROOT="$PWD/data/raw"
export MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba
```

The branch is already created when this document is used from the implementation checkout. Verify that `git status --short` still shows the historical untracked V5 metrics, MATLAB manifest, and local ZIP and do not stage them.

Run the stages in order:

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge \
  python scripts/run_vsb_modality_contribution_diagnostic.py \
  --config configs/experiments/vsb-modality-contribution-diagnostic-localraw.yaml \
  --audit-root results/audits/vsb-literature-forensic \
  --output-root results/audits/vsb-modality-contribution \
  --stage preflight \
  --log-file logs/vsb_modality_contribution_diagnostic.log
```

After preflight passes, run `regression`, `grouped`, `random`, `statistics`, and `report`, or use `--stage all`. Existing completion markers make the runner resumable; add `--force` only when intentionally regenerating an entire stage. The grouped stage runs fixed HGB and secondary logistic robustness matrices; it performs no neural training.

Verify the final outputs:

```bash
/opt/micromamba/bin/micromamba run -n partial-discharge python -m pytest -q
sed -n '1,240p' reports/audits/vsb_modality_contribution_diagnostic.md
```

The report and JSON are development-only. The locked VSB test, official unlabeled test, and MATLAB Te2 must remain unopened.
