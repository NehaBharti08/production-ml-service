# Data

**Nothing in this directory is committed except this file and `checksums.txt`.**

## Why

The dataset is public and de-identified, so committing it would not breach
anything. It still doesn't belong in git:

- **It sets the wrong default.** A repository that demonstrates responsible
  handling of consumer financial data should not normalise putting
  borrower-level records in version control, whatever their licence.
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

**Lending Club accepted loans, 2007–2018**
Mirror: <https://huggingface.co/datasets/codesignal/lending-club-loan-accepted>

2,260,701 originated loans with 151 columns, issued 2007-06 to 2018-12. After
the label is resolved to terminal outcomes only and immature loans are removed,
**672,379 loans across 2007-06 to 2015-12, 14.8% default**. 1.6 GB of CSV in,
43 MB of parquet out.

**Provenance rests on a mirror, and that is stated rather than hidden.** Lending
Club withdrew the official download, so these bytes come from a community copy
on the Hugging Face hub. `checksums.txt` pins them:

    3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a

A mirror that changes underneath the project would alter every number in the
model card in silence. The checksum converts that into a loud failure. It does
not recover the original source, which is gone.

## Two properties that shape the whole project

**It has a real date, and that date is a string.** `issue_d` is a genuine
origination date — the chronological split needs no proxy argument — but it is
stored as `"Dec-2018"`. Sorting it directly orders it *alphabetically*: Apr,
Aug, Dec, Feb, Jan, Jul, Jun, Mar, May, Nov, Oct, Sep. That looks chronological
to every check short of parsing it, and it silently produced all 64 drift
thresholds and both replays before being caught. Ordering by time goes through
`clean.order_by_time`, which parses first, and nothing else is allowed to do
it. See [`docs/DECISIONS/0004-chronological-split.md`](../docs/DECISIONS/0004-chronological-split.md).

**The maturity rule censors by loan term.** A label only exists once a loan
reaches a terminal state, so the pipeline keeps only loans whose full term has
elapsed: `issue_d + term <= observation_end`. Because the term is 36 or 60
months, that rule bites unevenly — 36-month loans are kept through 2015-12,
60-month loans only through 2013-11. The consequence is structural:

| Split | 60-month share | Default rate |
|:--|--:|--:|
| train | 13.19% | 14.75% |
| val | 0.00% | 15.12% |
| test | 0.00% | 14.71% |

The model is **trained** on 60-month loans, which default at 25.22% against
13.95% for 36-month, and **evaluated** on a population containing none of them.
Its behaviour on the longer term is therefore unvalidated, while a real lender
would certainly be asked to score them. See
[`docs/DECISIONS/0010-term-censoring.md`](../docs/DECISIONS/0010-term-censoring.md).

## Disclaimer

This data supports an engineering demonstration only. Nothing derived from it
has been validated for lending or reviewed for fair-lending compliance, and
none of it may decide anyone's access to credit or its price.
