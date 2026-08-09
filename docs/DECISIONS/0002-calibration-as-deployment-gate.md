# ADR 0002 — Calibration is a deployment gate, not a report

- **Status:** Accepted
- **Date:** 2026-08-09
- **Phase:** 0 (enforced from Phase 7)

## Context

The standard portfolio treatment of a binary classifier is accuracy, maybe
ROC-AUC, and a confusion matrix. Calibration — whether a predicted probability
of 0.3 corresponds to a 30% observed event rate — is almost never measured.

For a health-adjacent prediction task that omission is not cosmetic. A
readmission risk score is only useful if the number means something. A model
that ranks patients correctly but systematically outputs 0.4 where the true
rate is 0.1 will misallocate any resource allocated on the basis of its
magnitude, while looking excellent on every ranking metric.

## Decision

Calibration is measured, reported, **and enforced**:

1. Reliability diagrams, Brier score and expected calibration error are reported
   alongside discrimination metrics in the model card — never instead of them.
2. A challenger model is **blocked from promotion** if it calibrates materially
   worse than the incumbent, regardless of its discrimination:

   ```yaml
   promotion.calibration:
     max_brier_ratio_vs_incumbent: 1.02   # at most 2% worse
     max_ece: 0.05                        # 10-bin expected calibration error
   ```

The gate is executable code in `mlservice.retraining.gates`, unit-tested with a
deliberately mis-calibrated model that must fail it.

## Reasoning

**A better-ranking, worse-calibrated model is a regression here, not an
improvement.** Encoding that as a hard gate rather than a guideline is what
makes the claim credible — anything softer gets waived the first time it is
inconvenient.

**The 2% Brier margin is a tolerance, not a target.** It permits noise between
retrains without permitting drift in calibration quality.

**ECE ≤ 0.05 at 10 bins** is a conventional operating point: a mean absolute
gap of five percentage points between predicted and observed frequency. Chosen
as a standard rather than derived, and labelled `STANDARD` in
`configs/thresholds.yaml` for that reason.

## Consequences

A model may be refused promotion despite better PR-AUC. That is the intended
behaviour and the log names calibration as the failing gate.

Calibration methods (Platt, isotonic) are selected on the validation split in
Phase 2, not the test split — otherwise the calibration report measures the fit
to its own evaluation data.

The project spends effort on calibration that a conventional portfolio project
would spend on model sophistication. That trade is the thesis: a well-operated
simple model beats an unmonitored complex one.

## Revisit if

The service ever moves to a purely ranking-based use (top-N triage lists), where
only ordering matters. It has not, and the model card's intended-use section
should be checked before anyone assumes otherwise.
