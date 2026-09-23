# ADR 0010 — Term-dependent censoring, and what it costs

- **Status:** Accepted
- **Date:** 2026-09-23
- **Phase:** 6 (found), 1 (caused)

## Context

[ADR 0003](0003-dataset-selection.md) chose this dataset partly for its
right-censoring, and `clean.drop_immature_loans` handles it with a rule stated
rather than tuned:

    issue_d + term <= observation_end

A loan's outcome is not final until its term ends, so only loans whose full term
has elapsed by the last observation in the file can carry a label. That rule is
correct and it is the most consequential one in the pipeline.

It is also **term-dependent**, and nothing in the project had accounted for what
that implies. The term is 36 or 60 months. Against a 2015-12 observation end:

| Term | Last issue date that can mature | Share of the book | Default rate |
|:--|:--|--:|--:|
| 36 months | 2015-12 | 92.36% | 13.95% |
| 60 months | 2013-11 | 7.64% | 25.22% |

The rule removes two extra years of 60-month originations that it does not
remove for 36-month loans. The split inherits that directly:

| Split | Issue range | 60-month share | Default rate |
|:--|:--|--:|--:|
| train | 2007-06 → 2014-12 | 13.19% | 14.75% |
| val | 2015-01 → 2015-06 | 0.00% | 15.12% |
| test | 2015-07 → 2015-12 | 0.00% | 14.71% |

## How it was found

Not by inspection. The drift threshold calibration was fixed (it had been
ordering windows alphabetically — see the commit history for
`clean.order_by_time`), and after the fix `term` showed a PSI profile unlike
any other feature: median churn **0.0016** across 19 consecutive stable
training windows, and a single spike of **0.7038** at the 2013-11 boundary.

Flat, then one cliff, at exactly the month the maturity rule starts excluding
60-month loans. Three other features hit the calibration ceiling and all three
are genuine lender policy changes. This one was ours.

The live monitor made it unmistakable: `term` reported PSI **1.5738 in all 32
replay windows** — the identical value every time, because the difference is
fixed by the pipeline rather than by the world.

## Decision

**Three things, none of them "change the maturity rule".**

**1. `term` is excluded from drift monitoring**, with its reason, its
measurement and its revisit condition recorded in `configs/thresholds.yaml`
under `drift.not_monitored`. Every drift report that omits it says so in its
notes.

The bar for exclusion is deliberately high — a feature comes out only when the
alarm is uninformative *by construction*, not when it is merely noisy. This one
qualifies: the training reference structurally contains a population that any
2015 window structurally cannot, so the PSI is a constant, and a constant alarm
carries no information. It was also actively harmful, because `int_rate`'s
genuine 2014-to-2015 repricing was sitting underneath it.

**2. `term` stays a model feature.** It is genuinely predictive — 25.22% versus
13.95% — and at serving time a real lender would be scoring 60-month
applications. Dropping it would throw away real signal to tidy up a monitoring
problem.

**3. The evaluation gap is disclosed, not patched.** The model is trained on
60-month loans and evaluated on a split containing none, so **its performance
on 60-month loans is unvalidated**. That is stated in the model card's
limitations and in `data/README.md`, and it is not something a different split
strategy fixes.

## Options rejected

**Drop `term` as a feature.** Would silence the monitor and cost real signal.
The monitoring problem does not justify a modelling change.

**Move the observation end back to 2013-11 so both terms censor alike.** Makes
the populations comparable and throws away 2014 and 2015 entirely — the most
recent, most relevant, and largest cohorts in the book. Paying two years of
data for a tidier drift report is a bad trade.

**Restrict the whole project to 36-month loans.** Honest, and it would make the
train and test populations match. Rejected because it hides the finding rather
than reporting it, and because a lender's book contains both.

**Leave the alarm firing and explain it in the runbook.** This is what was
happening by default, and it is how alert fatigue is manufactured. An operator
who learns that one feature is always red learns to skim the list.

## Consequences

**Good**

- The drift monitor's remaining 63 features are all ones whose movement means
  something. `int_rate` is now visible.
- The exclusion is auditable: a reason, a measurement, and a condition under
  which it becomes wrong.
- A genuine evaluation limitation is documented rather than unnoticed.

**Costs and limits**

- **A real distribution shift in `term` would now go unseen.** If the lender
  stopped offering 60-month loans tomorrow, nothing would alert. This is the
  direct cost of the exclusion and it is accepted knowingly.
- **The held-out metrics describe a 36-month population.** Every number in the
  model card carries that qualification.
- The exclusion is hand-maintained, so it does not regenerate with
  `monitor calibrate` — by design, since it encodes a judgement rather than a
  measurement, but it does mean someone has to revisit it.

## Revisit if

- **The observation window extends far enough for 60-month loans to mature
  alongside 36-month ones.** Then the populations are comparable again and this
  exclusion becomes wrong — `term` should go back under monitoring.
- A 60-month holdout becomes available from any source, which would let the
  unvalidated half of the book actually be evaluated.
- The maturity rule changes. Everything here follows from it.
