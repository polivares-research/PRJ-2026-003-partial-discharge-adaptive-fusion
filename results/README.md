# Confirmatory results

`results/manifests/` contains the small, reproducible sample-to-split CSVs and
is intended for version control. Large representation caches, predictions,
features and per-run artifacts belong under ignored `results/cache/`,
`results/predictions/`, `results/features/` or `results/runs/` paths. Curated
CSV/JSON/YAML summaries and publication figures are retained separately under
`reports/` according to the repository policy.

The native VSB policy/results from the v2 execution are retained as historical
diagnostics only and are explicitly invalid for final scientific interpretation
because the 800,000-sample input geometry does not match MATLAB's 400-sample
signals. The revised v3 selection, expert outputs and signal-level fusion use
separate versioned paths and must not be mixed with the historical artifacts.
