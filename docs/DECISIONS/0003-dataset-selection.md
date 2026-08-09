# ADR 0003 — Dataset selection: Diabetes 130-US Hospitals

- **Status:** Accepted
- **Date:** 2026-08-10
- **Phase:** 1

## Context

The project needs a dataset that can carry a genuine MLOps demonstration:
monitoring, drift detection, subgroup analysis, and retraining. That imposes
requirements most portfolio datasets fail.

The widely-circulated symptom-to-disease datasets on Kaggle are **synthetically
generated and trivially separable** — a decision tree reaches near-100% accuracy
on them. Reporting that number signals inexperience to any reviewer who knows
the data. More importantly, a dataset with no irreducible error has nothing to
monitor: there is no degradation to detect and no drift worth acting on.

## Decision

Use **Diabetes 130-US Hospitals** (UCI #296): 101,766 real inpatient encounters
across 130 US hospitals, 1999–2008.

## Reasoning

**It is real and messy.** Missingness is structural rather than sprinkled:
`weight` 96.9%, `medical_specialty` 49.1%, `payer_code` 39.6%. Two columns
(`examide`, `citoglipton`) are constant across all 101,766 rows. Missing values
are encoded as a literal `?`, so naive null-counting reports zero missingness
and is wrong — the dataset punishes carelessness, which is the point.

**It is genuinely hard.** An unconstrained decision tree reaches **0.519 test
ROC-AUC** — barely above chance — while memorising the training set. The
published literature ceiling is ROC-AUC 0.65–0.68. That ceiling is a *feature*:
a model that cannot be made excellent must be carefully operated, which is this
project's thesis.

**It contains a real leakage trap.** `discharge_disposition_id` encodes expired
discharges, and those 1,652 encounters contain exactly **zero** positive labels.
A model would learn `died → not readmitted` from the discharge code. Finding and
removing this is a stronger demonstration of judgement than any accuracy number.

**It has authentic time structure.** Verified, not assumed — see
[ADR 0004](0004-temporal-split-proxy.md). Practice genuinely shifts across the
decade, which makes drift monitoring real rather than staged.

**Fairness dimensions are analysable.** `race`, `gender` and `age` all have
subgroups above n=500, so Phase 2's subgroup breakdown can say something
statistically meaningful rather than decorative.

## Options rejected

**UCI Heart Disease (303 rows).** Too small for drift monitoring, subgroup
analysis, or load testing. Retained only as a smoke-test fixture.

**MIMIC-IV.** Has genuine timestamps, which would be strictly better than a
proxy. Rejected on schedule: PhysioNet credentialing plus CITI training is a
multi-week delay before any code could be written.

**Synthetic symptom-to-disease datasets.** Rejected on the grounds described
above — they are the specific failure mode this project is meant to avoid.

## Consequences

**Headline metrics will be modest.** PR-AUC around 0.10–0.25 against a ~7.6%
test prevalence floor. The README must frame this correctly and up front, or a
casual reader will mistake an honest result for a weak one.

**Accuracy is unusable as a headline.** The majority-class baseline scores
92.4% accuracy with 0% recall. That number appears in the audit specifically to
make the argument concrete.

**The dataset is 1999–2008.** Clinical practice has moved on considerably. This
is stated in the model card's limitations and is not a defect for an operations
demonstration, but it would disqualify the model from any real use.

## Revisit if

Credentialed MIMIC-IV access becomes available and real timestamps would
materially strengthen the drift work. The pipeline is written against a schema
module, so the change would be contained rather than pervasive.
