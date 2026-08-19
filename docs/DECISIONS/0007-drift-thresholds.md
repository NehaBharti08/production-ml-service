# ADR 0007 — Drift thresholds by empirical-null calibration

- **Status:** Accepted
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

1. Split the training period into 20 consecutive windows of 1,994 rows. These
   are periods we accepted by training on them.
2. Compute PSI per feature between **adjacent** pairs — 19 comparisons each.
3. That distribution is this dataset's normal churn for that feature.
4. `threshold[f] = clamp(percentile_99(null_psi[f]), 0.10, 0.25)`

Every threshold then answers "why that number?" with **"because this feature
moved that much between stable training windows only 1% of the time."**

Regenerate with `uv run mlservice monitor calibrate`. The full null distribution
is written to `reports/null_calibration.json`, not just the chosen value —
keeping only the threshold would make it unfalsifiable.

## The measurement that justifies this

Running both approaches over the 19 stable comparisons, for all 43 features
(817 checks):

| Approach | False alarms | Rate |
|:--|--:|--:|
| Uniform PSI > 0.10 | 21 | **2.57%** |
| Calibrated per-feature | 6 | **0.73%** |

The calibrated rate closely matches the 1% the 99th percentile predicts, which
is a check that the method does what it claims.

**The finding that settles it: `medical_specialty` has a median null PSI of
0.1196 — above the conventional 0.10 threshold.** It would have breached in
**11 of 19 windows we had already accepted as stable**, alarming more than half
the time on data with no drift in it at all.

That feature is exactly the one Phase 1 identified as having shifting recording
practice across 1999–2008. Its volatility is real; the conventional threshold
simply cannot express it.

## What calibration produced

| Outcome | Count | Meaning |
|:--|--:|:--|
| Clamped to the 0.10 floor | 37 | Genuinely stable; the convention is right for them |
| Set their own threshold | 4 | Natural churn above the floor |
| Clamped to the 0.25 ceiling | 2 | So volatile the convention would alarm constantly |

The four that set their own: `medical_specialty` (0.242), `number_diagnoses`
(0.221), `num_lab_procedures` (0.168), `admission_type_id` (0.151).

Note that most features **do** land on the conventional floor. Calibration did
not overturn the convention — it identified the six features for which the
convention was wrong, which is the entire value.

## Why the clamp

Floor and ceiling are the credit-risk convention, retained as guardrails:

- **Floor 0.10** — a pathologically stable feature could otherwise set a
  hair-trigger threshold near zero and fire on rounding.
- **Ceiling 0.25** — a pathologically noisy one could set a bar so permissive
  that real drift hides beneath it. `admission_source_id` and
  `discharge_disposition_id` both hit this; their alerts are correspondingly
  weak evidence, and that is visible in the config rather than hidden.

## Alert conditions, and why confirmation

Three or more features breaching, for **two consecutive windows**.

With 43 features at a 99th-percentile threshold, roughly 0.43 features breach
per window by chance. Requiring three puts the per-window false-alarm rate into
the low percent under a Poisson tail; requiring two consecutive windows makes
sustained-versus-momentary the deciding factor. A single-window blip is noise,
and paging on noise is how a pager gets ignored.

## Consequences

**Real drift is detected between train and test, and it is not a false alarm.**
Replaying the untouched test split flags 6 then 5 features — `medical_specialty`,
`admission_source_id`, `admission_type_id`, `number_diagnoses` — the same
recording-practice features Phase 1 found shifting. The model genuinely operates
on a drifted population, consistent with the positive rate falling from 9.87%
(train) to 7.57% (test).

**This complicates the induced-drift demo, honestly.** Because real drift is
already present, the "clean" windows also alert. The induced effect is therefore
demonstrated on the *manipulated feature specifically*: `age` PSI is 0.021 and
0.014 in the untouched windows and **0.529** in the induced one — 25× its
threshold. Prediction drift responds too, with the alert rate going from a 0.26
reference to 0.414.

Reporting "the detector fired after we induced drift" without that isolation
would have been misleading, since it fires beforehand as well.

**Thresholds must be regenerated when the feature set changes.** They are tied
to a specific schema, and a feature with no calibrated threshold is skipped with
a warning rather than given a default — a silently defaulted threshold is
precisely the arbitrary number this ADR exists to avoid.

## Revisit if

The training window is replaced (retraining shifts the reference), the feature
set changes, or the monitoring window size moves materially — PSI's null
distribution depends on sample size, so a much smaller window would need
recalibration.
