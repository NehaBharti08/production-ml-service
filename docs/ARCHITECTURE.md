# Architecture

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

> **Status: not yet written — filled in by Phase 3.**
> This placeholder states what the document will contain so the gap is
> visible rather than silent.

---

Will contain the system diagram, data flow, and component responsibilities:
the API and its prediction log, MLflow tracking and registry, the monitoring
job, Prometheus/Grafana, and the retraining path.

The prediction log schema is the load-bearing design decision — Phases 5, 6 and
7 all read it, so it is designed deliberately in Phase 3 rather than evolved.
