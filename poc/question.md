# PoC question

Does an uncertainty-based adaptive fusion of two heterogeneous neural experts
—one temporal and one spectral—improve MCC over the best individual expert and
over a fixed probability average?

The protocol uses `Tr0` for training, `Va0` for early stopping, calibration,
and threshold selection, and `Te0` for final evaluation. `Te1` and `Te2` are
excluded from this PoC.

Initial feasibility support requires the adaptive fusion to improve MCC over
all baselines by at least `0.005`.
