# Load Test Report

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

> **Status: not yet written — filled in by Phase 4.**
> This placeholder states what the document will contain so the gap is
> visible rather than silent.

---

Will contain measured p50/p95/p99 across four k6 scenarios, the throughput
knee, and a named breaking point.

This document is what turns `slo.latency` in `configs/thresholds.yaml` from
`PLACEHOLDER` into measured numbers. The derivation rule is fixed in advance so
the thresholds cannot be reverse-engineered to whatever the service happens to
do:

    slo_p99 = 2 x measured_p99 @ target_rps
    page    = slo_p99 breached for 5 minutes
    ticket  = 1.5 x measured_p99 sustained 30 minutes
