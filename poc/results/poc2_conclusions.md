# PoC-2 conclusion

- Strongest individual expert selected from development OOF: **temporal**; frozen anchor used for the primary locked comparison: **Temporal**; descriptive Te1 leader: **CWT**.
- OOF recoverable anchor-error fraction: **0.5710**; OOF Oracle ΔMCC: **+0.029079**; locked Te1 Oracle ΔMCC vs frozen anchor: **+0.133143**.
- Conservative gate: **0.931780 MCC**, ΔMCC vs frozen anchor **+0.083328**.
- Overrides: **22,698**; beneficial **21,842**; harmful **856**; precision **0.9623**; recovered anchor errors **0.5881**.
- Paired bootstrap 95% CI for ΔMCC: **[+0.082195, +0.084435]**; fraction positive **1.0000**.
- McNemar b/c: **856/21842**, exact p-value **0**.
- Decision: **GO** — Conservative fusion improved the locked best individual with a positive paired-bootstrap lower bound and meaningful error reduction.

`Te2` remains reserved for future out-of-distribution/generalization evaluation.
