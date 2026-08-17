# Architecture

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration
> of ML operations. Nothing here is clinically validated or fit to inform
> patient care.

---

## Components

```
                    ┌──────────────────────────────┐
   client  ───────► │  FastAPI  (mlservice.api)    │
                    │                              │
                    │  middleware: request-ID,     │
                    │    timing, access log        │
                    │  routes: /v1/predict         │
                    │          /v1/predict/batch   │
                    │          /v1/outcomes        │
                    │          /v1/model           │
                    │          /health/live        │
                    │          /health/ready       │
                    │          /metrics            │
                    └───┬───────────┬──────────┬───┘
                        │           │          │
              ┌─────────▼──┐  ┌─────▼──────┐  ┌▼─────────────┐
              │ ModelStore │  │ prediction │  │ Prometheus   │
              │            │  │    log     │  │  registry    │
              │ registry   │  │ (NDJSON,   │  │              │
              │   ↓ fail   │  │  append)   │  └──────────────┘
              │ local      │  └─────┬──────┘
              │ fallback   │        │
              └─────┬──────┘        │ read by
                    │               ▼
              ┌─────▼──────┐  ┌──────────────────────────┐
              │  MLflow    │  │ Phase 5 dashboards       │
              │  registry  │  │ Phase 6 drift + labels   │
              │  (optional)│  │ Phase 7 retrain triggers │
              └────────────┘  └──────────────────────────┘
```

## Load-bearing decisions

**The API does not depend on MLflow being up.** `ModelStore` tries the registry,
then falls back to an artifact baked into the image, and records which one served
each prediction (`model_source`). The compose file deliberately does not make the
API wait for MLflow to be healthy — availability should not depend on the
tracking server.

**Liveness and readiness answer different questions.** Liveness checks nothing
external, because a failure restarts the container and a registry outage must not
cause a restart loop. Readiness requires a loaded model *and* a successful canary
inference, because an instance that cannot score should be removed from the load
balancer rather than restarted. This split is what makes the Phase 7 rollback
demo work.

**The feature transformer travels inside the model artifact.** The API sends raw
values and never reimplements encoding. Reimplementing it at serving time is one
of the most common ways a service silently serves garbage.

**The threshold comes from the trained model, not config.** It is a property of
the fitted model in the same way its coefficients are. See
[ADR 0006](DECISIONS/0006-prediction-log-schema.md) and the note in
`model_loader._decision_threshold`.

**The prediction log is the monitoring substrate**, designed up front rather than
evolved. See [ADR 0006](DECISIONS/0006-prediction-log-schema.md).

## Measured latency (host, single process)

Measured on an i5-12450H, not estimated. The plan's guess of 4–8 ms p50 was wrong
by an order of magnitude, and the reason is worth knowing:

| Component | Time |
|:--|--:|
| Pydantic validation | 0.05 ms |
| `ColumnTransformer.transform` | **64 ms** |
| `LogisticRegression.predict_proba` | 0.99 ms |
| Calibrator wrapper | ~0.5 ms |
| Prediction log write (fsync) | ~5 ms |
| **End-to-end single prediction** | **~50 ms** |

**The model is 1 ms; the feature pipeline is 64 ms.** The one-hot encoder's
`infrequent_if_exist` path has large per-*call* overhead that is nearly
independent of row count — the output is only 190 columns. Batch scaling confirms
it:

| Batch size | Total | Per item |
|--:|--:|--:|
| 1 | 85.8 ms | 85.8 ms |
| 200 | 47.7 ms | **0.24 ms** |
| 500 | 107 ms | 0.21 ms |

So the batch endpoint is a ~350× per-item throughput multiplier, not a
convenience. **Phase 4 sets the latency SLO from a proper load test**, using the
derivation rule already committed in `configs/thresholds.yaml`
(`slo = 2 × measured p99`) — the rule was fixed in advance precisely so the
numbers could not be reverse-engineered to whatever the service happened to do.

If that budget proves too tight, the optimisation with the clearest payoff is the
transform, not the model.
