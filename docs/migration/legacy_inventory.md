# Initial legacy-repository inventory

## Fuente

```text
/home/polivares/Dropbox/Work/Research/PartialDischarges/
```

It must be treated as read-only. Inspection found approximately 7,062 files,
22 GB of raw data, 109 GB of processed data, 77 GB of models, 16 MB of
notebooks, 382 GB of references including backups, and 346 MB of reports.

## Rescue after review

- `src/data/make_dataset.py`: reference for the VSB reader and metadata.
- `src/data/spectrogram.py`: conceptual reference for transformations.
- MCC implementations and metric calculations, always with new tests.
- Notebook ideas involving spectrograms, wavelets, and ensembles.
- Papers on PD, Rauscher et al., MCC, and signal processing.
- Historical results labeled as legacy baselines.

## Archive as reference

- Historical notebooks and their outputs.
- Old checkpoints/models.
- Manuscripts, reviews, cover letters, and previous submissions.
- Experimental results that do not follow the new protocol.

## Do not migrate

- Raw data and large derived datasets.
- Per-sample images.
- Backups, caches, LaTeX builds, and temporary files.
- `.env`, credentials, and personal configuration.
- `.git` directories created by synchronization conflicts.
- Duplicate copies of notebooks or manuscripts.

## Review observations

The legacy loader contains hardcoded parameters, and the image generator creates
random signal-level splits. This is insufficient for VSB because phases share a
measurement. The code must be rewritten and validated rather than copied
automatically.
