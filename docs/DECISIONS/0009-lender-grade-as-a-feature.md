# ADR 0009 — Keep the lender's own grade as a feature, and say what that costs

- **Status:** Accepted
- **Date:** 2026-09-21
- **Phase:** 2

## Context

Three columns in this dataset are not facts about the borrower. `grade`,
`sub_grade` and `int_rate` are the **output of Lending Club's own underwriting
model** — its assessment of how risky the loan is, and the price it charged for
that risk.

They are legitimately available at origination, so they are not leakage in the
usual sense. But including them changes what this model *is*: it stops being an
independent risk estimate and becomes, in part, a function of someone else's.

The trained champion shows how large that part is:

| Feature | Coefficient |
|:--|--:|
| `grade_A` | **−0.7762** |
| `sub_grade_A1` | **−0.5671** |
| `purpose_small_business` | +0.5370 |
| `addr_state_MS` | +0.3802 |
| `grade_F` | +0.3543 |
| `grade_G` | +0.3482 |

The two largest weights in the model are the lender's grade.

## Decision

**Keep them**, name them explicitly in `schema.LENDER_ASSESSMENT_FEATURES`, and
monitor them per-feature — while stating plainly in the model card what their
inclusion means.

## Reasoning

**Removing them would not make the model independent — it would make it worse
and hide the dependency.** The grade was computed from borrower attributes, most
of which are also features here. Dropping the grade leaves a model that
re-derives much of the same judgment from the same inputs, less accurately.
The dependency on the lender's view of risk does not disappear; it just stops
being visible in the coefficients.

**It is also what the baseline measures.** The strongest trivial baseline ranks
loans by `int_rate` alone — the lender's own priced risk — and reaches PR-AUC
0.2529. The champion reaches 0.2723 with non-overlapping intervals. So the
honest claim is: *this model adds something to the lender's own assessment*,
and a margin of 0.019 PR-AUC is real but narrow. That claim only makes sense
if the grade is in the model.

**Their policy changes arrive as drift, which is useful.** The empirical-null
calibration found the lender's policy levers are by far the most volatile
inputs:

| Feature | Measured p99 PSI between stable windows |
|:--|--:|
| `initial_list_status` | 0.638 |
| `term` | 0.563 |
| `int_rate` | 0.416 |

against 57 of 64 borrower features sitting on the 0.1 floor. Lending Club kept
changing its product. Keeping those columns — and monitoring them against their
own measured churn — means a change in the lender's policy surfaces as a drift
signal rather than as unexplained performance decay months later.

## Consequences

**Good**

- The model is more accurate, and the comparison against the lender's own
  pricing is meaningful.
- The dependency is explicit: named in the schema, visible in the coefficients,
  stated in the model card.
- Lender policy shifts are detectable as drift.

**Costs — and these are the reason this ADR exists**

- **The model inherits any bias in the lender's judgment.** A model that
  reproduces an existing lender's decisions reproduces that lender's
  disparities, and looks accurate while doing so. Accuracy measured against
  that lender's own historical outcomes cannot reveal this.
- **Part of the model's apparent skill is theirs.** Beating a baseline built
  from their own pricing by 0.019 is the honest measure of what this model adds.
- **It is not transferable.** A different lender's grades mean something
  different, or do not exist. This model cannot be applied to another lender's
  applicants without retraining on that lender's own assessments.

## Revisit if

- The project ever needs an **independent** risk estimate — for instance, to
  audit the lender's grading itself. Then the grade must come out, and the
  resulting drop in accuracy is the price of independence.
- A disparate-impact review is performed. If `grade` carries a geographic or
  socioeconomic disparity of its own, including it launders that disparity
  into this model with a veneer of neutrality.
