# Decision 0003: V5 pulse-aware VSB representation

- Status: proposed development protocol
- Scope: VSB representation and V5 gate only
- Data contract: repository-local `data/raw` through `PD_RAW_DATA_ROOT`
- Statistical unit: one labelled parent signal
- Group isolation: all phases of one `id_measurement` remain in one split
- MATLAB: inherited 400-sample representation
- Holdouts: VSB grouped holdout and MATLAB `Te2` remain closed until freeze
- Historical boundary: V3/V4 artifacts are not overwritten or reinterpreted

The V5 temporal candidate is described as a
`LITERATURE-INSPIRED_SIGNAL-LEVEL_ADAPTATION` because the available paper and
current dataset do not establish an exact code-, label-, or split-level
reproduction.
