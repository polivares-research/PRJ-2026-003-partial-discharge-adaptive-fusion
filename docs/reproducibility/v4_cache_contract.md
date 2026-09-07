# V4 Cache Contract

- **[PROPOSED · COMPUTATIONAL]** A representation cache is valid only when its shape, dtype, dataset/version, split, window geometry, aggregation, CWT parameters, preprocessing version, source provenance, and train-only normalizer fingerprint match the request.
- **[PROPOSED · COMPUTATIONAL]** Standardizers are fitted only on the training parent signals and stored separately with a fingerprint of mean and standard deviation values.
- **[PROPOSED · COMPUTATIONAL]** Arrays are written to temporary memmaps and atomically renamed only after complete coverage and metadata serialization.
- **[PROPOSED · COMPUTATIONAL]** Missing, malformed, incompatible, or partial files are cache misses and are rebuilt; they are never silently reused.
- **[INHERITED · SCIENTIFIC]** Cached windows retain parent IDs and split assignments conceptually; all model losses and metrics remain at parent-signal level.

- **[PROPOSED · COMPUTATIONAL]** Runtime worker/prefetch settings and mixed-precision state are recorded with the run; mixed precision is disabled until numerical/prediction equivalence is demonstrated.
