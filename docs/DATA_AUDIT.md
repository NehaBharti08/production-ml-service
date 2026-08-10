# Data Audit

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

> **Status: not yet written — filled in by Phase 1.**
> This placeholder states what the document will contain so the gap is
> visible rather than silent.

---

What this will contain, once Phase 1 runs:

- **Leakage hunt** — expired/hospice discharge codes, patient overlap across
  splits, and any column that encodes the outcome. Reported with before/after
  metrics, because removing a leak should *lower* the headline numbers and that
  drop is the evidence.
- **Missingness analysis** — per column, with a recorded decision and its
  reasoning (`weight` is ~97% missing; the right move is to drop it and say so,
  not to impute it).
- **Class imbalance** — positive rate and what it implies for metric choice.
- **Temporal proxy verification** — whether ordering by `encounter_id` actually
  carries time signal. See [DECISIONS/0004](DECISIONS/0004-temporal-split-proxy.md).
- **Separability sanity check** — an unconstrained decision tree, fitted
  deliberately. If it reaches >0.95 AUC something is leaking.
- **Trivial baselines** — majority class, prevalence, and a one-feature
  heuristic, so later improvements are measured against something real.

Findings will be reported as found, including unflattering ones.
