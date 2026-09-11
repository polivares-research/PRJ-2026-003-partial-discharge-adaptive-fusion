# Full-Signal Temporal vs Global Spectrogram Diagnostic

Status: `PROPOSED / DEVELOPMENT-ONLY`

This diagnostic tests whether the poor VSB result was caused by the local-event
representation rather than by the usefulness of spectral information itself.
It is a controlled representation diagnostic, not V6 and not a complete
adaptive-fusion confirmation.

## Registered representation

Each VSB signal remains a complete 800,000-sample waveform at 40 MHz. The new
path computes one Hann STFT with `n_fft=512`, `win_length=512`, `hop=256`,
`center=False`, one-sided power, and `10*log10(power+1e-8)`. Frequencies below
500 kHz are removed; the Nyquist limit is taken dynamically from the recorded
sampling frequency. The resulting array is `[1, 250, 3124]` and one array
produces one parent-signal probability.

There is deliberately no pulse detector, local segment extraction, circular
boundary handling, MIL aggregation, event mask, CWT rerun, or event-level
label. The existing local-event spectrogram path remains historical and is not
overwritten.

MATLAB keeps the registered 400-sample Tr1/Va1 global STFT path. The temporal
experts, target, grouped VSB `id_measurement` split, train-only normalization,
OOF threshold selection, and protected holdouts remain inherited invariants.

## Evaluation boundary

Development uses seeds 42–44. MATLAB opens only `Tr1.mat` and `Va1.mat`.
VSB uses only the forensic train/validation identities. Te1, Te2, the VSB
grouped test split, and the official unlabeled test are inaccessible. Fixed
50/50 and OOF-selected fixed mixtures are diagnostics only; no adaptive or
reliability model is fitted.

Global arrays are cached as unnormalized float16 memmaps. Standardization is
fit per frequency across time using only the relevant outer-training fold for
OOF and the complete training subset for final validation. A failed temporal
regression, provenance check, split check, or cache check forces the final
verdict to `INCONCLUSIVE`.

## Resource expectation

The VSB development cache is approximately 10.1 GiB (`6972 × 1 × 250 × 3124`
float16). This estimate is disk/cache usage, not a promise that the full array
will be loaded into RAM or VRAM. Streaming uses bounded Parquet batches;
training uses `num_workers=0`, pinned memory, and one parent signal per sample.

## Interpretation limits

If global VSB spectrogram performance recovers while local-event performance
does not, that supports a geometry/coverage explanation but does not prove a
physical mechanism. If it remains weak, the result is evidence against the
tested compact global spectrogram model under this protocol, not against all
spectral representations. Historical CWT and local-event outputs are context
only and are not recomputed.
