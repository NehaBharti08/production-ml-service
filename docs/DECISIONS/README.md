# Architecture Decision Records

Short records of decisions that were not obvious, written when the decision was
made rather than reconstructed afterwards. Each states the context, the
decision, the reasoning, the consequences, and what would cause a revisit.

Numbered chronologically by when the decision was taken, so Phase 0 decisions
precede Phase 1 ones.

| # | Decision | Phase | Status |
|:--|:--|:--|:--|
| [0001](0001-local-first-no-managed-cloud.md) | Local-first infrastructure, no managed cloud | 0 | Accepted |
| [0002](0002-calibration-as-deployment-gate.md) | Calibration is a deployment gate, not a report | 0 | Accepted |
| [0003](0003-dataset-selection.md) | Dataset selection — Diabetes 130-US Hospitals | 1 | Accepted |
| [0004](0004-temporal-split-proxy.md) | Chronological split on an `encounter_id` proxy | 1 | Accepted |
| [0005](0005-model-selection.md) | Model selection — simplest not-significantly-worse | 2 | Accepted |
| [0006](0006-prediction-log-schema.md) | Prediction log schema and durability | 3 | Accepted |
| [0007](0007-drift-thresholds.md) | Drift thresholds by empirical-null calibration | 6 | Accepted |
| [0008](0008-promotion-gates-and-rollback.md) | Promotion gates run all-or-nothing, and the rollback path is exercised | 7 | Accepted |

## Why these exist

Interview questions about a portfolio project are rarely "what did you build".
They are "why did you build it that way", and an answer reconstructed months
later is visibly reconstructed. These records also make the *rejected* options
visible, which is usually the more interesting half.

## Format

```markdown
# ADR NNNN — Title

- **Status:** Proposed | Accepted | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD
- **Phase:** N

## Context      What forced a decision
## Decision     What was decided, stated plainly
## Reasoning    Why, including options rejected and why
## Consequences What this makes easier, and what it costs
## Revisit if   The condition that would reopen this
```
