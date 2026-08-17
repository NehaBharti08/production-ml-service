# ADR 0005 — Model selection: simplest not-significantly-worse candidate

- **Status:** Accepted
- **Date:** 2026-08-16
- **Phase:** 2

## Context

The plan committed to a deliberately boring model, with XGBoost admitted *only
if the audit justified it*. Phase 2 had to turn that commitment into a decision
rule that could not be quietly bent when a more complex model showed a higher
number.

The failure mode being guarded against is specific and common: a candidate
scores 0.1147 against another's 0.1234, someone reports "the ensemble did
better", and a model nobody can explain ends up in production on a difference
smaller than the sampling noise.

## Decision

**The simplest candidate whose confidence interval overlaps the best candidate's
wins.** A more complex model must beat the simpler one with *non-overlapping*
bootstrap confidence intervals to be preferred.

Implemented in `mlservice.models.train.select_champion` and reported in every
run's `selection_rationale.json`, so the rule is executed rather than promised.

## What actually happened

| Candidate | PR-AUC | 95% CI | ECE | MCE |
|:--|--:|:--|--:|--:|
| baseline_prevalence | 0.0756 | [0.0716, 0.0799] | 0.0231 | 0.0231 |
| **logistic_l2** | **0.1234** | **[0.1110, 0.1375]** | 0.0140 | 0.1000 |
| logistic_l2_strong | 0.1220 | [0.1098, 0.1355] | 0.0143 | 0.1424 |
| random_forest_shallow | 0.1147 | [0.1037, 0.1268] | 0.0139 | 0.8000 |

All three real candidates have overlapping intervals. The rule selects
`logistic_l2`, which also happens to hold the highest point estimate — so in
this run the rule and the raw number agree, but the rule is what the decision
rests on.

**The shallow random forest did not beat logistic regression.** That is the
empirical result the "boring model" commitment needed, and it means the choice
is now evidence rather than preference.

## XGBoost was not tried, and that is the audit's conclusion

The plan permitted it on evidence of non-linear structure worth the operational
cost. The audit produced the opposite evidence:

- An **unconstrained decision tree** — able to represent arbitrary feature
  interactions — reached **0.519 test ROC-AUC** while memorising the training
  set. There is no rich interaction structure being left on the table.
- The **shallow random forest** here, which explicitly models interactions,
  scored *below* plain logistic regression.

Adding a gradient booster would buy a compiled dependency, longer build times, a
larger attack surface in the serving image, and a model that is harder to
explain — in exchange for signal the data does not appear to contain. Declining
it is the defensible call, and it is a stronger interview answer than a marginal
score would have been.

## The calibration finding that justifies reporting MCE

The random forest has the **best** ECE (0.0139) and by far the **worst** MCE
(0.80). One probability bin was catastrophically miscalibrated while the
count-weighted average looked healthy.

Had calibration been summarised with a single number, that model would have
looked like the best-calibrated candidate. This is precisely why
`mlservice.models.calibration` reports both, and why the gate is not the only
thing looked at.

## Consequences

**The champion is a regularised logistic regression**, calibrated with isotonic
regression fitted on validation, at a decision threshold of 0.1011 chosen for a
50% recall target on validation.

**Serving is cheap and inspectable.** Inference is a dot product; coefficients
can be read directly. This matters for the Phase 5 latency budget and for
debugging an incident at 2am.

**The threshold does not transfer perfectly.** Validation recall 0.507 became
test recall 0.465. Reported in the model card rather than smoothed over, and it
is exactly the behaviour Phase 6 monitoring exists to catch.

**A documented fairness disparity ships with the model.** Recall is 0.311 for
AfricanAmerican patients against 0.478 for Caucasian patients, and runs from
0.233 to 0.692 across age bands. Measured, published in the model card, and
gated against worsening in Phase 7 — but **not mitigated**, because the likely
mechanism is unequal completeness of prior-utilisation records, which
reweighting would obscure rather than fix.

## Revisit if

Retraining on a later window shows the random forest opening a non-overlapping
gap, or the feature set gains genuinely interacting variables. The rule stays
the same; only its outcome would change.
