# Protocolo de reproducibilidad

Each run must record:

- repository commit;
- configuration version;
- dataset and catalog version;
- split manifest;
- representation and parameters;
- model and hyperparameters;
- seed and environment;
- imbalance strategy;
- threshold and calibration;
- metrics and prediction location.

Structured output must support regenerating tables, figures, and statistical
analysis without depending on a notebook.

Local path configuration is not versioned. Small manifests, hashes, and
provenance metadata are versioned.
