# ADR 0001: initial repository architecture

## Decision

Adopt a structure inspired by Cookiecutter Data Science, extended with
explicit layers for representations, models, evaluation, calibration,
imbalance, fusion, and data contracts.

## Rationale

The study is not just a single neural-network pipeline: it requires comparing
heterogeneous experts, preserving source splits, evaluating generalization,
and maintaining a canonical manuscript.

## Consequences

- Notebooks will not be the source of truth.
- Raw datasets will remain in `datasets_catalog`.
- Large products will be regenerable and remain outside Git.
- `Te2` will have explicit protection in configuration and documentation.
- Legacy migration will be selective and reviewed.
