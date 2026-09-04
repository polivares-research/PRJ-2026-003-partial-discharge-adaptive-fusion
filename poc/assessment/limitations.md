# Limitations

- The PoC uses one seed and two small 1D CNNs; it is not an exhaustive search
  or a final architecture.
- Calibration, threshold selection, and early stopping use `Va0`; `Te0` is
  evaluated only at the end.
- Adaptive fusion uses probability entropy as a reliability proxy rather than
  a learned gating network.
- Evaluation is limited to `Tr0/Va0/Te0`; generalization to the held-out
  objects (`Te2`) is deferred to a later phase.
- The dataset license and some MATLAB format details still require formal
  verification; the reader validates the observed schema and fails explicitly
  on incompatible changes.
