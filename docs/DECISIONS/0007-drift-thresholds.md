# ADR 0007 — Drift thresholds by empirical-null calibration

- **Status:** Accepted — amended 2026-09-23 (see *The first calibration was wrong*)
- **Date:** 2026-08-19
- **Phase:** 6

## Context

Drift alerting needs a number per feature: how much movement is too much. Two
conventional answers exist, and both fail here.

**A statistical test.** A Kolmogorov–Smirnov test on a 5,000-row monitoring
window returns p < 0.05 for shifts far too small to act on. Statistical
significance scales with n; practical significance does not. Alerting on
p-values at monitoring window sizes guarantees a permanently red dashboard, and
a permanently red dashboard is one nobody reads.

**A convention.** The credit-risk PSI bands (< 0.1 stable, 0.1–0.2 moderate,
> 0.2 significant) are defensible and widely cited. But they are *generic*: they
say nothing about how much any particular feature in this dataset naturally
churns.

## Decision

Derive each feature's threshold from **its own observed churn between windows
already accepted as stable**:

1. Split the training period into 20 consecutive windows of 19,467 rows,
   ordered by **parsed** issue date. These are periods we accepted by training
   on them.
2. Compute PSI per feature between **adjacent** pairs — 19 comparisons each.
3. That distribution is this dataset's normal churn for that feature.
4. `threshold[f] = clamp(percentile_99(null_psi[f]), 0.10, 0.25)`

Every threshold then answers "why that number?" with **"because this feature
moved that much between stable training windows only 1% of the time."**

Regenerate with `uv run mlservice monitor calibrate`. The full null distribution
is written to `reports/null_calibration.json`, not just the chosen value —
keeping only the threshold would make it unfalsifiable.

## The first calibration was wrong

`issue_d` is stored as a string — `"Dec-2018"`. The calibration sorted on it
directly, which orders it **alphabetically**: Apr, Aug, Dec, Feb, Jan, Jul, Jun,
Mar, May, Nov, Oct, Sep. Every "consecutive" pair of windows was adjacent in the
alphabet, and every threshold was the churn between them.

Nothing failed. The thresholds were plausible, the headline finding read well,
and the defect was only proved by reproducing the recorded PSI values *exactly*
from a lexicographic sort. `chronological_split` never had the bug because it
parsed the date first; two monitoring modules did not.

The ordering now lives in `clean.order_by_time`, which parses and refuses
unparseable dates, and no call site sorts the column itself. A regression test
carries a positive control: a fixture on which the naive sort *demonstrably*
produces a different answer, so the test cannot keep passing if ordering
silently stops mattering.

What follows is the second calibration.

## The measurement that justifies this

Running both approaches over the 19 stable comparisons, for all 64 features
(1,216 checks):

| Approach | False alarms | Rate |
|:--|--:|--:|
| Uniform PSI > 0.10 | 20 | **1.64%** |
| Calibrated per-feature | 12 | **0.99%** |

The calibrated rate lands on the 1% the 99th percentile predicts, which is a
check that the method does what it claims.

**The finding that settles it is in the medians.** The features the convention
handles worst are not noisy — they are flat, with a single step change each:

| Feature | Median null PSI | p99 | The step |
|:--|--:|--:|:--|
| `initial_list_status` | 0.0069 | 0.6726 | whole-loan listing introduced, late 2012 |
| `term` | 0.0016 | 0.6169 | the pipeline's own maturity rule, 2013-11 |
| `int_rate` | 0.0331 | 0.3339 | repricing, late 2011 |
| `verification_status` | 0.0170 | 0.2644 | income verification tightened, late 2010 |

A uniform 0.10 breaches `int_rate` in 4 of 19 windows accepted as stable, while
its median churn is a third of that bar. One constant either pages on every step
or sleeps through everything in between. Alphabetical ordering had hidden this
entirely, by smearing each step across windows until the features looked
uniformly volatile.

## What calibration produced

| Outcome | Count | Meaning |
|:--|--:|:--|
| Clamped to the 0.10 floor | 54 | Genuinely stable; the convention is right for them |
| Set their own threshold | 6 | Natural churn above the floor |
| Clamped to the 0.25 ceiling | 4 | Step changes the convention would page on |

The six that set their own: `purpose` (0.218), `inq_last_6mths` (0.149), `dti`
(0.134), `sub_grade` (0.132), `installment` (0.131), `revol_bal` (0.104).

Most features **do** land on the conventional floor. Calibration did not
overturn the convention — it identified the ten features for which the
convention was wrong, which is the entire value.

## Why the clamp

Floor and ceiling are the credit-risk convention, retained as guardrails:

- **Floor 0.10** — a pathologically stable feature could otherwise set a
  hair-trigger threshold near zero and fire on rounding.
- **Ceiling 0.25** — a pathologically noisy one could set a bar so permissive
  that real drift hides beneath it. Four features hit this. Three are lender
  policy levers whose alerts are correspondingly weak evidence, visible in the
  config rather than hidden. The fourth, `term`, turned out to be ours — see
  [ADR 0010](0010-term-censoring.md) — and is excluded from monitoring
  altogether.

## Alert conditions, and why confirmation

Three or more features breaching, for **two consecutive windows**.

With 63 monitored features at a 99th-percentile threshold, roughly 0.63 breach
per window by chance. Requiring three puts the per-window false-alarm rate at
about 2.6% under a Poisson tail; requiring two consecutive windows takes it to
about 0.07%, and makes sustained-versus-momentary the deciding factor. A single-window blip is noise,
and paging on noise is how a pager gets ignored.

## Consequences

**Real drift is detected between train and test, and it is not a false alarm.**
Replaying the untouched test split, `int_rate` breaches in 31 of 32 windows and
`initial_list_status` in 26 — the book was repriced and its listing mix changed
between the 2014 reference and the 2015 loans. The alert confirms at window 16.
The default rate barely moves (14.75% train, 14.71% test), so this is drift in
*who is offered what*, not in how often borrowers default.

**One ceiling feature was an artefact of the pipeline.** `term` reported PSI
1.5738 in all 32 replay windows, because the maturity rule leaves 60-month loans
in the reference and removes them from every 2015 window. It is excluded, with
its reason and revisit condition recorded in config. See
[ADR 0010](0010-term-censoring.md).

**This complicates the induced-drift demo, honestly.** Because real drift is
already present, a "clean" window can alert too — window 0 puts `grade` at
0.1021 against a 0.10 threshold. The induced effect is therefore shown on the
manipulated feature specifically: over-sampling the D-G shoulder takes `grade`
to 0.2746, 0.3137 and 0.3032, and the alert rate from 0.3334 to 0.4388–0.4620.
Reporting "the detector fired after we induced drift" without that isolation
would be misleading, since it fires beforehand as well.

**Thresholds must be regenerated when the feature set changes.** They are tied
to a specific schema, and a feature with no calibrated threshold is skipped with
a warning rather than given a default — a silently defaulted threshold is
precisely the arbitrary number this ADR exists to avoid.

## Revisit if

The training window is replaced (retraining shifts the reference), the feature
set changes, or the monitoring window size moves materially — PSI's null
distribution depends on sample size, so a much smaller window would need
recalibration.
