# Monitoring Design

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

> **Status: not yet written — filled in by Phase 5.**
> This placeholder states what the document will contain so the gap is
> visible rather than silent.

---

Will document every metric and every threshold *with its derivation*. The
short version of the argument, which `configs/thresholds.yaml` already encodes:

- **Effect sizes, not p-values.** At ~20k rows per monitoring window a K-S test
  returns p < 0.05 for shifts far too small to act on. Statistical significance
  scales with n; practical significance does not.
- **Thresholds calibrated against an empirical null.** Per-feature PSI
  thresholds come from measuring how much each feature naturally churns between
  stable training windows, then alerting at the 99th percentile of that
  distribution. "Why this number?" then has a per-feature, data-derived answer.
- **Two-window confirmation.** Single-window blips are noise, and paging on
  noise is how a pager gets ignored.
