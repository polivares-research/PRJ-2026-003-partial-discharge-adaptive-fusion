# Confirmatory v2 Notebook Execution Summary

- Generated (UTC): 2026-09-05T00:01:37.219368+00:00
- Configuration: `two-dataset-confirmatory-v2-batch4-localraw`
- Environment: `partial-discharge`
- Neural training policy: CUDA required; physical and inference batch 4; gradient accumulation 1
- Notebook outputs: persisted in the corresponding notebook files

## Notebook status

| Notebook | Status | Duration (s) | Errors |
|---|---:|---:|---|
| `notebooks/00-infrastructure/0.0-por-gpu-memory-budget.ipynb` | passed | 1.4 | — |
| `notebooks/01-data/1.0-por-dataset-inventory.ipynb` | passed | 1.8 | — |
| `notebooks/01-data/1.1-por-matlab-data-audit.ipynb` | passed | 6.0 | — |
| `notebooks/01-data/1.2-por-vsb-data-audit.ipynb` | passed | 1.7 | — |
| `notebooks/01-data/1.3-por-experimental-protocol.ipynb` | passed | 0.6 | — |
| `notebooks/02-experts/2.0-por-matlab-experts.ipynb` | passed | 10430.4 | — |
| `notebooks/02-experts/2.1-por-vsb-experts.ipynb` | passed | 9929.9 | — |
| `notebooks/02-experts/2.2-por-expert-complementarity.ipynb` | passed | 0.6 | — |
| `notebooks/03-fusion/3.0-por-static-fusion.ipynb` | passed | 3.8 | — |
| `notebooks/03-fusion/3.1-por-reliability-aware-fusion.ipynb` | passed | 0.6 | — |
| `notebooks/03-fusion/3.2-por-conservative-adaptive-fusion.ipynb` | passed | 0.6 | — |
| `notebooks/04-evaluation/4.0-por-multiseed-evaluation.ipynb` | passed | 0.5 | — |
| `notebooks/04-evaluation/4.1-por-statistical-analysis.ipynb` | passed | 0.6 | — |
| `notebooks/04-evaluation/4.2-por-cross-dataset-analysis.ipynb` | passed | 0.5 | — |
| `notebooks/05-results/5.0-por-final-results.ipynb` | passed | 0.6 | — |
| `notebooks/05-results/5.1-por-publication-figures.ipynb` | passed | 0.9 | — |

## Scientific interpretation

The execution log is an audit trail, not evidence of publication readiness by itself. Publication claims require complete expert predictions for all five seeds, locked MATLAB Te2 and VSB grouped-holdout outputs, paired statistics, and the pre-specified MCC criterion. Any failed or incomplete expert notebook makes the confirmatory result incomplete.
