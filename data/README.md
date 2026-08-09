# Data

**Nothing in this directory is committed except this file and `checksums.txt`.**

## Why

The dataset is public and de-identified, so committing it would not breach
anything. It still doesn't belong in git:

- **It sets the wrong default.** A repository that demonstrates responsible
  handling of health-adjacent data should not normalise putting clinical
  records in version control, whatever their licence.
- **Git history is permanent.** A file committed once survives every later
  deletion. Establishing "row-level data never enters history" from the first
  commit is easier than fixing it afterwards.
- **It makes the repo unreviewable.** ~20 MB of CSV buries the code a reviewer
  actually came to read.

Reproducibility is preserved by `checksums.txt` instead: the fetch script
verifies the archive against a recorded SHA256, so everyone works from
byte-identical data without anyone shipping it.

## Getting the data

```bash
uv run mlservice data download     # or: make data
```

Downloads from UCI, verifies the checksum, and unpacks into `raw/`.

## Layout

| Directory | Contents | Written by |
|:--|:--|:--|
| `raw/` | Original archive, unmodified | `data download` |
| `interim/` | Cleaned, leakage removed, not yet featurised | `data clean` |
| `processed/` | Train/val/test splits ready for modelling | `data split` |
| `reference/` | Frozen training window — the drift baseline | `data split` |
| `monitoring/` | Prediction log and matured outcomes | the API, at runtime |

`reference/` is the comparison window every drift report is measured against.
It is written once and then left alone: a reference that silently tracks recent
data cannot detect drift, because it drifts along with it.

## Source

**Diabetes 130-US Hospitals for Years 1999–2008**
UCI Machine Learning Repository, dataset #296 — <https://doi.org/10.24432/C5230J>

101,766 inpatient encounters across 130 US hospitals. Inclusion criteria:
inpatient admission, documented diabetes diagnosis, 1–14 day stay, laboratory
tests performed, medications administered.

Originally published with:

> Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J.,
> Clore, J.N. (2014). *Impact of HbA1c Measurement on Hospital Readmission
> Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research
> International.

Licensed CC BY 4.0. Attribution is carried in `docs/MODEL_CARD.md`.

## Two properties that shape the whole project

**There is no timestamp column.** Not a missing one — none exists. The only
time signal is the ordering of `encounter_id`. Every "temporal" claim in this
repo rests on that proxy, which Phase 1 tests empirically before relying on.
See [`docs/DECISIONS/0004-temporal-split-proxy.md`](../docs/DECISIONS/0004-temporal-split-proxy.md).

**Some rows have a deterministic label.** `discharge_disposition_id` encodes
expired and hospice discharges. A patient who died cannot be readmitted, so
those rows carry a label the model can learn to predict from the discharge code
alone. They are excluded during cleaning; the audit reports the before/after.

## Disclaimer

This data supports an engineering demonstration only. Nothing derived from it
is clinically validated or fit to inform patient care.
