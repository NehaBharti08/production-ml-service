# ADR 0004 — Chronological split on a real date, cut at month boundaries

- **Status:** Accepted
- **Date:** 2026-09-20
- **Phase:** 1

## Context

The data spans 2007–2015. Across that window Lending Club's book grew by three
orders of magnitude, its grade mix moved repeatedly as it chased volume, and its
credit policy changed more than once. A model trained and evaluated on shuffled
rows would see 2015 underwriting while being scored on 2015 loans, and every
held-out number would be flattering in a way deployment never is.

So the split must be chronological. The questions are what to order by, and
where exactly to cut.

## Decision

Order by **`issue_d`**, and cut at **month boundaries** closest to 60/20/20.

| Split | Loans | Ends | Positive rate |
|:--|--:|:--|--:|
| train | 389,353 | 2014-12 | 0.1475 |
| val | 120,790 | 2015-06 | 0.1512 |
| test | 162,236 | 2015-12 | 0.1471 |

## Reasoning

**`issue_d` is a real date, so no proof is needed.** A dataset without a
timestamp forces you to argue that some ordering — an ID sequence, say —
carries time signal, and to test that argument before trusting it. That is a
real and interesting problem, but it is absent here: the column is a date, and
the split is chronological by construction. The whole statistical apparatus
such an argument requires simply does not exist in this repository, and it is
worth noticing how much complexity one trustworthy column removes.

**Month boundaries, not row positions.** Loans issued in the same month share a
credit policy, a marketing campaign and a macro environment. Cutting through a
month puts near-siblings on both sides of the boundary. The leak is small, but
it is real and it is free to avoid.

The cost is approximate fractions — **57.9 / 18.0 / 24.1** rather than 60/20/20
— because monthly volume varies enormously: 2007 has hundreds of loans a month
and 2015 has tens of thousands. The achieved split is reported rather than
assumed, and the imprecision is a better trade than the leak.

**Censoring is handled before splitting, not by the split.** Loans whose term
had not elapsed by the observation date are removed in cleaning (see
[ADR 0003](0003-dataset-selection.md)). Without that, the most recent months —
exactly the ones that land in test — would carry a label contaminated by export
date, and the test set would look healthier than the model is.

**The drift reference is the training split.** Drift means "different from what
the model learned", so the reference must be exactly what it learned from. A
later window would measure drift against data the model never saw.

## Options rejected

**Random split.** Leaks the future into training. Rejected outright.

**Row-position cut after sorting.** Simpler and hits 60/20/20 exactly, at the
cost of splitting origination cohorts across the boundary.

**Year boundaries.** Too coarse: the book is so back-loaded that a year cut
cannot get near the requested fractions. 2015 alone is 42.1% of the cleaned
data and 2014 another 24.2%, so the only year-aligned cuts land at roughly
34/24/42 or 58/42 — nowhere near 60/20/20.

## Consequences

**Good**

- The chronological claim needs no hedging.
- Positive rates are stable across the split (0.1475 / 0.1512 / 0.1471), which
  is itself evidence that the maturity rule removed the censoring that would
  otherwise dominate them.
- Every split contains both classes, asserted in the data-quality suite.

**Costs and limits**

- **Repeat borrowers can straddle the split.** `member_id` is null for every
  row, so a borrower with a 2014 loan and a 2015 loan can appear in both train
  and test. This is recorded in the split result itself, not only in prose, and
  the data-quality suite pins it so the caveat cannot go stale.
- **Train is dominated by its last two years.** 2013 and 2014 together are
  43.7% of all cleaned loans, while 2007–2011 contribute 6.4%. The "seven-year"
  training window is in practice mostly two years, so the model knows little
  about the lending environment of 2007–2009 — including the financial crisis.
  That is realistic for a model deployed in 2015, but it means the training
  distribution is not the deployment distribution, and a crisis-era shock
  would reach it largely unseen.

## Revisit if

- Borrower identity becomes available: split by borrower within the
  chronological order to close the straddling gap.
- The observation window extends, which would allow later cohorts to mature
  and shift every boundary.
