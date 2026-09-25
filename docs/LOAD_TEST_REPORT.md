# Load Test Report

> **NOT A CREDIT DECISIONING SYSTEM.** This document reports engineering
> measurements of a demonstration service. Nothing here has been validated for
> lending.

**Date:** 2026-09-24 · **Tool:** Locust 2.x · **Target:** the API container from
`docker-compose.yml`, single uvicorn worker, 512 MB limit · **Hardware:**
i5-12450H (8C/12T), 15.6 GB RAM, Windows 11 · **Model:** `credit-default-risk`,
threshold 0.2070, schema `f5566c96d5eed4de`

---

## Read this first: what these numbers are and are not

The service now runs **in its container**, which is half of what the first
report asked for. The other half is still missing: the load generator runs on
**the same machine**, so Locust and the service compete for the same 12 threads
and every latency figure includes that contention.

It shows. Three identical 45-second runs at the same load produced p99s of
**100, 430 and 350 ms**.

So:

- **Trustworthy:** relative results — where the knee is, whether throughput
  inverts past it, and the before/after of the fix in §4, which ran back to back
  on the same container.
- **Not trustworthy:** the absolute latencies as a characterisation. Treat them
  as an **upper bound**.

`configs/thresholds.yaml` keeps `remeasure_required: true`.

---

## 1. Where the service saturates

Ramp sweep, `--tags ramp`, 25 s per step, back to back, against the container.
Aggregated across all endpoints the scenario hits:

| Users | Requests | Median | **req/s** |
|--:|--:|--:|--:|
| 1 | 144 | 38 ms | 6.1 |
| 2 | 291 | 34 ms | 12.0 |
| 5 | 577 | 39 ms | 23.7 |
| 10 | 858 | 98 ms | **35.3** |
| 20 | 861 | 340 ms | 35.7 |
| 40 | 896 | 850 ms | 36.9 |

**The knee is at about 10 concurrent users and ~36 req/s.** Past it, throughput
stays flat and extra load only queues: latency rises roughly 3.5× per doubling
of users while requests per second barely move.

Zero failures at every step, including far past saturation. The service
degrades in latency, not by dropping requests.

## 2. Steady state at the target rate

The SLO rule measures p99 at `target_rps`, set at about half the knee: 20 req/s.
Three 45 s runs at 4 users:

| Run | req/s | p50 | p95 | p99 |
|--:|--:|--:|--:|--:|
| 1 | 23.3 | 34 ms | 75 ms | 100 ms |
| 2 | 20.3 | 42 ms | 210 ms | 430 ms |
| 3 | 19.6 | 53 ms | 190 ms | 350 ms |

0 failures across 2,758 requests. The 4× spread in p99 comes from the
co-located generator, not the service. The SLO takes the **median run**, not the
best one.

## 3. The derived SLO

The derivation rule was fixed in `configs/thresholds.yaml` **before any
measurement was taken**, so the thresholds could not be reverse-engineered to
whatever the service happened to do:

```
slo_p99   = 2   × measured p99 @ target_rps
page      = slo_p99, breached for 5 consecutive minutes
ticket    = 1.5 × measured p99, sustained for 30 minutes
```

| Setting | Value | From |
|:--|--:|:--|
| `target_rps` | **20** | about half the ~36 req/s knee |
| `measured_p99_ms` | 350 | median of the three runs above |
| `slo_p99_ms` | **700** | 2 × 350 |
| `slo_p95_ms` | 380 | 2 × 190 — the rule tightened this from 520 |
| `page_p99_ms` | 700 | SLO breached for 5 min |
| `ticket_p99_ms` | 525 | 1.5 × 350, sustained for 30 min |

The p99 SLO came out at 700 ms, the same as the medical service's. That's a
coincidence of two different measurements under one rule, not an inherited
number. The p95 SLO did change, and it tightened.

**The capacity alert changed too, and it had been dead.** `ApproachingSaturation`
fired at 42 req/s: 80% of the medical service's 52 req/s knee. The credit
service can't reach 42. It saturated at ~20 req/s before the fix below and at
~36 after it. An alert set above the maximum achievable rate can never fire, and
a capacity alert that can't fire looks exactly like a service with plenty of
headroom. It now fires at 29 req/s. An existing test asserting
`fires_at < measured_knee_rps` caught this the moment the re-measured knee
reached config. Until then it had passed only because config still held the
medical number.

## 4. 99.7% of inference was one library call

The first sweep on the credit service was about **3× slower than the medical
one**: capacity ~20 req/s, 76 ms mean latency for a single user. Rather than
record that, it was profiled step by step on one row:

| Component | Time |
|:--|--:|
| Numeric preprocessing | 109 ms |
| of which **`QuantileTransformer`** | **104 ms** |
| of which `SimpleImputer` | 4 ms |
| Categorical one-hot encoding | 8 ms |
| **The model itself** | **0.31 ms** |

The interpolation inside `QuantileTransformer` is cheap. The cost comes from
`output_distribution="normal"`, where sklearn calls `scipy.stats.norm.ppf`
**three times per column**: the mapping itself plus two clip bounds. Every one
of those calls runs through scipy's generic distribution machinery, argument
checking included. With 54 numeric columns that's 162 calls per request.
That's also why a 200-row batch cost the same as one row: the cost was per
call, not per row.

`scipy.special.ndtri` is the same function evaluated directly in C.
`FastNormalQuantileTransformer` in `src/mlservice/data/features.py` overrides the
forward path to use it and leaves everything else to the parent class.

**It changes no score by any amount.** The retrained artifact was compared
against the previous one on all **162,236 test rows**: maximum difference 0.0,
the same 0.20700843608046723 threshold, the same schema hash. The equivalence
tests require exact equality against the parent class on skewed, zero-heavy,
NaN-bearing and out-of-range data. That's the only protection a mirror of a
private sklearn method has against a future sklearn release, and it's why they
are strict.

| | Before | After |
|:--|--:|--:|
| Single-row `predict_proba`, in process | 67 ms | **19 ms** |
| Peak throughput | ~20 req/s | **~36 req/s** |
| Median at 10 users | 300 ms | **98 ms** |
| Past the knee (40 users) | throughput **falls**, 19.9 → 17.3 | throughput holds, 36.9 |

**This is a correction to the first report.** The medical service measured the
same kind of per-call cost, 64 ms in `ColumnTransformer.transform`, and
attributed it to the one-hot encoder's `infrequent_if_exist` path. That
attribution was never profiled to the individual step. On credit the encoder is
8 ms and the quantile transform 104 ms. The medical pipeline can't be
re-measured now, so it's unknown whether the earlier attribution was right for
that pipeline. What's known is that nobody checked.

## 5. Batch is still a throughput multiplier, but a smaller one

Through the running API (`/v1/predict/batch`), median of 15 calls each:

| Batch size | Total | Per item |
|--:|--:|--:|
| 1 | 36.3 ms | 36.3 ms |
| 10 | 54.0 ms | 5.4 ms |
| 50 | 57.8 ms | 1.2 ms |
| 200 | 128.3 ms | **0.64 ms** |
| 500 | 305.4 ms | 0.61 ms |

About **57× cheaper per item at batch 200**. The medical service reported ~350×,
but that was mostly amortising the per-call cost §4 removed. Any caller scoring
more than a handful of applications should still use `/v1/predict/batch`.

## 6. A 10× win found by profiling (earlier service)

*Measured on the medical service on 2026-08-18. The code path is unchanged.*

The first 200-item batch took **490 ms**. Profiling attributed ~400 ms of it to
**200 separate `fsync` calls**, against ~85 ms for the model itself. Durability,
not computation, was the dominant cost of batch prediction.

`write_many` fsyncs once per batch: **490 ms → 47.7 ms**. Crash semantics are
unchanged in any way that matters. `read_records` already tolerates a truncated
tail, which is the expected state after an unclean shutdown. See
[ADR 0006](DECISIONS/0006-prediction-log-schema.md).

## 7. What did *not* degrade

**Error handling under load.** The invalid-payload scenario runs throughout, and
422s count as successes, because a 422 is the *correct* response to malformed
input and scoring it as a failure would make the error rate meaningless. The
scenario now sends `fico_range_low: 5000` against its real `[300, 900]` bound.
Until today it sent `"age": "75"`, a medical field. That still got a 422, but
through `extra="forbid"`'s unknown-field rejection, so the scenario was timing a
different code path from the one it claimed.

**Prediction log growth** *(earlier service)*. Tested directly on 2026-08-18:
with a 19 MB, 11,936-record log, single predictions took 79–97 ms; with a fresh
file, 70–110 ms. No effect. Append plus fsync is O(1) in file size. The log
still has no rotation, which is a real gap for a long-running deployment, but
it's not a latency problem.

## 8. Reproducing this

```bash
docker compose up -d api

# PYTHONUTF8=1 is required on Windows: locust reads pyproject.toml with the
# ANSI codepage and chokes on the non-ASCII characters in its comments.
for u in 1 2 5 10 20 40; do
  PYTHONUTF8=1 uv run locust -f loadtest/locustfile.py --headless \
    --users $u --spawn-rate $u --run-time 25s \
    --host http://127.0.0.1:8000 --tags ramp --only-summary
done
```

Scenarios: `smoke` (does it work at all), `steady` (latency profile with a
realistic mix), `ramp` (find the knee), `soak` (degradation over time).

**Until 2026-09-23 the load test couldn't have measured the credit service at
all.** `_varied_features()` spread `time_in_hospital`, `number_inpatient` and
`age` on top of the payload, and `extra="forbid"` would have turned every
request into a 422. A run would have timed the validation path and reported it
as prediction latency.

## 9. Outstanding

- [x] Re-measure with the service containerised.
- [ ] **Re-measure with the generator off-box.** Every number here is an upper
      bound until then.
- [ ] A soak run long enough to show or rule out drift over hours.
- [ ] Multi-worker measurement. Everything here is a single uvicorn worker.
- [ ] Profile what's left. After §4 a single call is ~36 ms through the API
      against 19 ms in-process, so HTTP, validation and the log write are now a
      comparable share. The next target should be measured, not assumed.
