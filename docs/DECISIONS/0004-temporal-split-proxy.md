# ADR 0004 — Chronological split on an `encounter_id` proxy

- **Status:** Accepted
- **Date:** 2026-08-10
- **Phase:** 1

## Context

A random split on time-structured data leaks the future into training and
inflates every number that follows. This project therefore requires a
chronological split.

**But this dataset has no timestamp column.** Not a missing one — none exists.
The only ordering signal available is `encounter_id`.

Sorting by `encounter_id` is common practice in published work on this dataset,
and it is almost certainly assigned sequentially at admission. But "almost
certainly" is an assumption, and asserting "temporal validation" on the strength
of it would be exactly the kind of unearned claim this project exists to avoid.

## Decision

Split chronologically on `encounter_id` ordering — **but verify the proxy
empirically first**, and let the evidence determine what may be claimed.

Verification is a gate in `mlservice.data.split.verify_time_proxy`, not a
one-off notebook check. If it ever fails, the claim automatically downgrades
from *"chronological split"* to *"ordered holdout"*.

## The test, and why it works

Clinical practice change is **directional**: a drug is adopted, a data field
starts being captured, a lab test falls out of favour. A meaningless row
ordering produces non-monotonic noise. Spearman rank correlation across deciles
distinguishes the two.

Eight signals were chosen to be independent of each other and of the target —
two data-capture practices, two lab-ordering practices, four prescribing
patterns. If unrelated columns trend together, the common cause is time.

**Criteria:** |ρ| > 0.8 and p < 0.01, with at least 3 signals trending.

**Result: 5 of 8 trending — passed.**

| Signal | First decile | Last decile | ρ | p |
|:--|--:|--:|--:|--:|
| `payer_code` missing | 1.000 | 0.138 | −0.875 | 0.0009 |
| `medical_specialty` missing | 0.364 | 0.679 | +0.891 | 0.0005 |
| insulin prescribed | 0.465 | 0.602 | +0.891 | 0.0005 |
| metformin prescribed | 0.168 | 0.216 | +0.964 | 0.00001 |
| pioglitazone prescribed | 0.037 | 0.080 | +0.903 | 0.0003 |

`payer_code` alone is close to conclusive: capture goes from **100% missing to
13.8% missing**, monotonically. That is a data-capture system being rolled out
over time. It cannot happen by chance.

## The strongest evidence: a dated external event

Trend tests establish that the ordering carries *some* time signal. A
discontinuity does something stronger — it dates the ordering against the real
world, which is a prediction that could have failed.

Rosiglitazone prescribing holds steady at ~7.5%, then drops **3.7 percentage
points in a single 5% bin** at the 80th percentile of the ordering — **7.7×
larger than any other bin-to-bin change** in the series. Pioglitazone, the
competing drug in the same class, is unaffected, so this is drug-specific rather
than a general recording change.

That is the Avandia collapse: Nissen & Wolski's NEJM meta-analysis (May 2007)
linking rosiglitazone to myocardial infarction, followed by an FDA black-box
warning that November. Prescribing halved and never recovered.

**The ordering reproduces a dated pharmacovigilance event**, placing the 80th
percentile of `encounter_id` at roughly mid-2007. This is about as
well-evidenced as a temporal claim can be without a date column.

## Two consequences that changed the pipeline

### 1. `payer_code` is dropped — it is time-confounded, not merely missing

The same evidence that validates the proxy disqualifies the feature. Under a
chronological split, `payer_code` present/absent is almost a pure indicator of
*era*. A model would learn "payer_code recorded → later period → different base
rate", which is an artifact of data capture with no clinical meaning and no
ability to generalise.

This is a subtle failure: the column looks like ordinary missingness, and
imputing it would have baked the confound in rather than removing it.

### 2. A censoring buffer discards the final 5% of the ordering

A first encounter is labelled positive only if a **subsequent** encounter exists
in the data. Near the end of the collection window those later encounters are
unobserved, so the label is *missing, not negative* — textbook right-censoring.

The effect is invisible on all encounters (positive rate is flat across the
period) and appears only after first-encounter deduplication: the rate sits at
~9.5% through the first 80% of the ordering, then falls to **4.17% in the final
5% — a 56% relative collapse**.

Without a buffer that region lands entirely in the test set, depressing every
held-out metric for a reason unrelated to the model, and Phase 6 drift
monitoring would chase an artifact of data collection.

## What may and may not be claimed

**May:** "chronological split on a verified `encounter_id` time proxy, with a
censoring buffer."

**May not:** "temporal validation with timestamps." There are no timestamps.
The README, model card and audit all state this plainly.

## Residual limitation, stated openly

A positive-rate gradient survives the censoring buffer: 9.87% (train) → 8.99%
(val) → 7.57% (test). Part is genuine prior-probability shift; part is residual
censoring the buffer does not fully remove. The two cannot be cleanly separated
without timestamps.

Phase 2 must therefore **not** attribute train-to-test performance differences
to model quality alone. This is recorded here so the limitation is confronted
rather than discovered later.

## Revisit if

Timestamps become available (e.g. via MIMIC-IV), or evidence emerges that
`encounter_id` is not assigned sequentially — in which case `verify_time_proxy`
should fail and the downgrade is automatic.
