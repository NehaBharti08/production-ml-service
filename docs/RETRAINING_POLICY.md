# Retraining Policy

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

> **Status: not yet written — filled in by Phase 7.**
> This placeholder states what the document will contain so the gap is
> visible rather than silent.

---

Will document triggers, promotion gates and rollback. Encoded in
`configs/thresholds.yaml` and executed by `mlservice.retraining.gates`.

The gate that matters most: **calibration is a hard block, not a report.** A
challenger that discriminates better but calibrates worse is a regression for a
health-adjacent use case, and is refused promotion.

Every gate is unit-tested with deliberately-bad models that must fail each gate
independently.
