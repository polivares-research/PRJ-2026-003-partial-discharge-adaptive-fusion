# VSB literature forensic audit execution

The audit is isolated to the branch audit/vsb-literature-forensic and uses only repository-local raw data.

## Full resumable execution

    export PD_RAW_DATA_ROOT="$PWD/data/raw"
    MAMBA_ROOT_PREFIX=/data/envs/polivares/atlas-micromamba \
    /opt/micromamba/bin/micromamba run -n partial-discharge \
    python scripts/run_vsb_literature_forensic_audit.py \
      --config configs/experiments/vsb-literature-forensic-audit-localraw.yaml \
      --raw-root "$PD_RAW_DATA_ROOT" \
      --io-batch-size 4 \
      --log-file logs/vsb_literature_forensic_audit.log

Stages skip when their completion marker exists. The heavy scan and bounded CWT stages stream only the 6,972 development signals. The VSB grouped test, official unlabeled test, and MATLAB Te2 remain locked.

## Stage-by-stage execution

Use the same command with --stage integrity, --stage scan, --stage morphology, --stage cwt, --stage baseline, or --stage report. Run integrity first; run report only after the preceding stages complete.

## Outputs

Generated raw-derived stage outputs are under results/audits/vsb-literature-forensic/ and figures under reports/figures/vsb-literature-forensic/. The compact handoff is:

- reports/audits/vsb_literature_forensic_audit.md
- reports/audits/vsb_literature_forensic_audit.json
- reports/audits/vsb_literature_forensic_manifest.md

No raw data, V5 caches, checkpoints, archives, or external weights are used or committed.
