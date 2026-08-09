# Model Card

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

> **Status: not yet written — filled in by Phase 2.**
> This placeholder states what the document will contain so the gap is
> visible rather than silent.

---

Will follow the Mitchell et al. (2019) model card structure, with these
sections treated as mandatory rather than optional:

- **Intended use** — and, at equal prominence, **out-of-scope use**
- **Training data** — provenance, vintage, inclusion criteria, known biases
- **Metrics** — PR-AUC as headline (not accuracy, not ROC-AUC alone), with
  bootstrap confidence intervals
- **Calibration** — reliability diagram, Brier score, expected calibration
  error. Reported alongside discrimination, never instead of it.
- **Subgroup performance** — across race, gender and age bands, with n and CIs,
  reported openly including where disparities are unflattering
- **Limitations** — single-institution, 1999-2008 vintage, proxy-temporal
  split, no clinical validation
