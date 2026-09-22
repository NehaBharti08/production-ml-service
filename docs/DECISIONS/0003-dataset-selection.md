# ADR 0003 — Dataset selection: Lending Club accepted loans

- **Status:** Accepted
- **Date:** 2026-09-11
- **Phase:** 1

## Context

The project needs a dataset that can carry a genuine MLOps demonstration:
monitoring, drift detection, subgroup analysis, calibration-gated retraining.
That imposes requirements most portfolio datasets fail.

It has to be **real and messy** rather than synthetic, **imbalanced** so that
accuracy is visibly the wrong metric, **modestly predictable** so the project
is about operating a model rather than tuning one, **time-ordered** so drift has
something to stand on, and it needs **slices worth reporting** for subgroup
analysis.

One requirement matters more than the others, and it is the one that decided
this: **someone downstream has to act on the probability itself, not merely the
ranking.** The project's central control is a promotion gate that blocks a
challenger which ranks *better* but calibrates *worse*. That argument only has
force where the number is used as a number.

## Decision

Lending Club's accepted-loan book: **2,260,701 loans issued 2007-06 to
2018-12**, reduced to **672,379 loans, 2007-06 to 2015-12**, after the label is
resolved and immature loans are removed. Target: charged off or defaulted,
**14.8%** positive.

The bytes come from a community mirror on the Hugging Face hub, pinned by
SHA256 — Lending Club withdrew the official download.

## Reasoning

**Calibration is how credit is priced.** Expected loss is probability of default
× exposure × loss given default, so a miscalibrated probability is a mispriced
loan. The calibration gate stops being an ethical nicety and becomes the thing
that protects the business. This is the single strongest reason for the choice.

**It has a real date.** `issue_d` is a date column. The split is chronological
by construction, and drift across 2007–2015 is real: the book grew by three
orders of magnitude and the credit policy changed repeatedly inside the window.
See [ADR 0004](0004-chronological-split.md).

**Its leakage is the kind practitioners actually fall into.** Thirty-one columns
are knowable only after the loan has run — `recoveries`, `total_rec_prncp`,
`last_fico_range_high`. They look like ordinary loan attributes. Training on
them reaches ROC-AUC **0.9983** against an honest ceiling near 0.70, and
`recoveries > 0` labels a default in **100.00%** of 80,312 cases. A dataset whose
leaks are obvious teaches nothing; this one's are subtle.

**Its right-censoring is large and easy to explain.** 38.9% of loans are still
in flight. The resolved fraction collapses from 100% for 2007–2013 to 11.4% for
2018 — and the 2018 default rate *falls*, which is survivorship among fast
resolvers rather than better lending. Handling it correctly moves the positive
rate from 0.1998 to 0.1481.

**Fair-lending analysis is a real discipline with real proxies.** US credit data
legally excludes race, gender and marital status, so disparate-impact analysis
uses geography and socioeconomic position instead — which is what compliance
teams actually do. Subgroups: `addr_state`, `emp_length`, `home_ownership`,
income band. 59 of 69 groups are large enough to analyse.

**The ceiling is modest.** ROC-AUC near 0.70, close to the published range for
this task. A model that cannot be perfected has to be operated.

## Options rejected

**UCI "Default of Credit Card Clients" (Taiwan, #350).** Checked first, because
it has *richer* demographics — sex, education, marital status, age — and downloads
directly. Rejected for one decisive reason: **it has no date column at all.**
Every row covers the same six months, so a chronological split is impossible,
real drift has nothing to stand on, and right-censoring does not exist to find.
Better protected attributes do not compensate for losing the time axis the
monitoring layer is built around.

**Home Credit Default Risk.** Relational, multi-table, and dated only in
relative days. The joins would dominate the project.

**Give Me Some Credit.** 150k rows, no dates, and clean enough to be
uninteresting.

**An ML-infrastructure dataset** — CI build failures, training-job failures.
Considered because it would be thematically neat. Rejected because nobody acts
on a build-failure probability as a number; they retune a threshold. The
calibration gate would lose its reason to exist, and there are no protected
attributes, so the fairness work would have gone too.

## Consequences

**Good**

- The calibration gate has an economic rationale, not only an ethical one.
- The chronological split needs no statistical argument that an ordering
  carries time signal.
- The leakage and censoring findings are both demonstrable with a number.
- Fairness analysis maps onto how the discipline is practised.

**Costs and limits**

- **Borrower identity is unavailable.** `member_id` is null for every row, so
  repeat borrowers cannot be detected and may appear in both train and test.
  Held-out metrics are inflated to whatever extent repeat borrowing occurs.
  Deduplicating by borrower is the standard defence and it is simply not
  available.
- **The model inherits the lender's judgment.** `grade`, `sub_grade` and
  `int_rate` are Lending Club's own risk output. See
  [ADR 0009](0009-lender-grade-as-a-feature.md).
- **Provenance rests on a mirror.** The checksum makes that acceptable — a
  changed mirror fails loudly — but the original source is gone.
- **1.6 GB raw.** Heavier than a portfolio dataset usually is; cleaning cuts it
  to 42 MB of parquet.

## Revisit if

- A dated dataset with protected attributes becomes available — it would
  strengthen the fairness analysis without costing the time axis.
- The mirror's checksum ever fails. Investigate before re-recording: every
  number in the model card depends on these bytes.
- Borrower identity becomes available, which would make deduplication possible
  and remove the largest caveat on the held-out metrics.
