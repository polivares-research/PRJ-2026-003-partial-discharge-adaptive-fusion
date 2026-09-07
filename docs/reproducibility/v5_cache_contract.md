# V5 cache contract

V5 caches are reusable only when their completion marker and metadata
fingerprint both validate. The fingerprint includes:

- dataset identifier and version;
- raw-data/source fingerprint and split-manifest hash;
- pulse detector policy, window lengths, bag capacity, and validity-mask rule;
- CWT wavelet, Morlet parameter, scales, time bins, log-power floor, and dtype;
- preprocessing version and Python/scientific-library versions; and
- exact array shape.

Artifacts are written to a temporary `.incomplete` path, flushed, atomically
renamed, and marked with a `.complete` file containing the expected hash.
Missing markers, partial files, or mismatched metadata are hard failures. A
failed preparation may be resumed only after the incomplete artifact is
recreated from scratch.

Raw pulse segments and unstandardized CWT arrays are reusable across seeds.
Normalizers are fitted independently from training rows of each OOF fold and
are never fitted from validation, test, Te2, or the unlabeled official VSB
test.
