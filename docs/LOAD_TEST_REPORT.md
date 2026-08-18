# Load Test Report

> **NOT FOR CLINICAL USE.** This document reports engineering measurements of a
> demonstration service. Nothing here is clinically validated.

**Date:** 2026-08-18 · **Tool:** Locust 2.x · **Target:** host uvicorn process,
single worker · **Hardware:** i5-12450H (8C/12T), 15.6 GB RAM, Windows 11

---

## Read this first: what these numbers are and are not

The load generator ran on **the same machine as the service**. Locust and
uvicorn competed for the same 12 threads, so every latency figure below includes
contention between the measurement and the thing being measured.

Repeat runs varied by **2–3×** with machine state — median 47 ms at 5 users in
one sweep, 130 ms at 6 users in another, with no code change between them.

So:

- **Trustworthy:** the *shape*. Where the saturation knee is, and the fact that
  throughput inverts past it. The ramp sweep ran back to back under comparable
  conditions, so the relative comparison holds.
- **Not trustworthy:** the absolute latency numbers as a characterisation. Treat
  them as an **upper bound**.

`configs/thresholds.yaml` carries `remeasure_required: true` for this reason.
The SLO derived below is deliberately loose — a tight SLO derived from a
contended measurement would page on healthy behaviour, which is worse than
having no SLO.

**Proper re-measurement needs the service in a container with the generator
off-box.** Docker is not yet installed on this machine.

---

## 1. Where the service saturates

Ramp sweep, `--tags ramp`, 25 s per step, back to back:

| Users | Requests | Avg ms | Median ms | **req/s** |
|--:|--:|--:|--:|--:|
| 1 | 138 | 34 | 36 | 5.7 |
| 2 | 263 | 40 | 39 | 10.7 |
| 5 | 464 | 56 | 47 | 19.1 |
| 10 | 823 | 76 | 68 | 33.5 |
| 20 | 1,274 | 139 | 110 | **52.3** |
| 40 | 1,162 | 522 | 500 | 47.6 ⚠ |

**The knee is between 20 and 40 concurrent users.**

At 40 users the service is past capacity, and the signature is unambiguous:
**throughput goes down** (52.3 → 47.6 req/s) while latency rises 3.8× (139 →
522 ms). Work is queueing rather than completing. Adding load past this point
makes the service slower *and* less productive.

**Maximum sustainable throughput: ~52 req/s** on one worker.

Zero failures at every step, including past saturation — the service degrades in
latency rather than by dropping requests.

## 2. Steady state at the target rate

`--tags ramp`, 6 users, 60 s, machine settled:

| Percentile | Latency |
|:--|--:|
| p50 | 130 ms |
| p75 | 180 ms |
| p90 | 240 ms |
| p95 | 260 ms |
| p99 | 350 ms |
| max | 461 ms |

787 requests, **0 failures**, 13.2 req/s.

## 3. The derived SLO

The derivation rule was fixed in `configs/thresholds.yaml` **before any
measurement was taken**, specifically so the thresholds could not be
reverse-engineered to whatever the service happened to do:

```
slo_p99   = 2   × measured p99 @ target_rps
page      = slo_p99, breached for 5 consecutive minutes
ticket    = 1.5 × measured p99, sustained for 30 minutes
```

| Setting | Value | From |
|:--|--:|:--|
| `target_rps` | **25** | ~half the measured knee of 52, for headroom |
| `measured_p99_ms` | 350 | steady-state run above |
| `slo_p99_ms` | **700** | 2 × 350 |
| `page_p99_ms` | 700 | SLO breached 5 min |
| `ticket_p99_ms` | 525 | 1.5 × 350, sustained 30 min |

**Why `target_rps` moved from 50 to 25.** The original plan guessed 50 rps.
Measurement put the knee at ~52 rps, so a 50 rps target would sit *at*
saturation with no headroom — the service would be one traffic spike from the
inverted-throughput regime.

## 4. That discipline mattered, because the estimate was badly wrong

The plan predicted **p50 of 4–8 ms**. Measured p50 is **~130 ms** — wrong by
more than an order of magnitude.

The Phase 3 profiling explains it, and it inverts the intuition:

| Component | Time |
|:--|--:|
| Pydantic validation | 0.05 ms |
| **`ColumnTransformer.transform`** | **64 ms** |
| `LogisticRegression.predict_proba` | 0.99 ms |
| Prediction log write (fsync) | ~5 ms |

**The model is 1 ms. The feature pipeline is 64 ms.** The one-hot encoder's
`infrequent_if_exist` path carries large per-*call* overhead that is nearly
independent of row count — the encoded output is only 190 columns.

Had the SLO been chosen after seeing the results, it would have been fitted to
this behaviour and would have hidden the finding entirely.

## 5. Batch is a throughput multiplier, not a convenience

Because the transform cost is per *call*, not per row:

| Batch size | Total | Per item |
|--:|--:|--:|
| 1 | 85.8 ms | 85.8 ms |
| 10 | 95.3 ms | 9.5 ms |
| 50 | 96.3 ms | 1.9 ms |
| 200 | 47.7 ms | **0.24 ms** |
| 500 | 107 ms | 0.21 ms |

**~350× better per item at batch 200.** Any caller scoring more than a handful
of patients should use `/v1/predict/batch`; the README and API docs should say
so rather than leaving it as an implementation detail.

## 6. A 10× win found by profiling

The first 200-item batch took **490 ms**. Profiling attributed ~400 ms of it to
**200 separate `fsync` calls** — against ~85 ms for the model itself. Durability,
not computation, was the dominant cost of batch prediction.

`write_many` fsyncs once per batch: **490 ms → 47.7 ms**. Per-item cost then
matched the isolated model benchmark exactly (0.24 vs 0.214 ms), confirming
fsync had been the entire gap.

Crash semantics are unchanged in any way that matters — `read_records` already
tolerates a truncated tail, which is the expected state after an unclean
shutdown. See [ADR 0006](DECISIONS/0006-prediction-log-schema.md).

## 7. What did *not* degrade

**Prediction log growth.** Suspected when latency appeared to drift upward
across runs, so it was tested directly rather than assumed: with a 19 MB /
11,936-record log, single predictions took 79–97 ms; with the log moved aside
and a fresh file, 70–110 ms. **No effect.** Append plus fsync is O(1) in file
size, and the apparent drift was machine contention from the preceding run.

The log still grows without bound and has no rotation. That is a real gap for a
long-running deployment and belongs in the Phase 8 runbook — but it is not a
latency problem.

**Error handling under load.** The invalid-payload scenario ran throughout;
422s were returned correctly and counted as successes by the scenario (a 422 is
the *correct* response to malformed input, and scoring it as a failure would
make the error rate meaningless).

## 8. Reproducing this

```bash
# terminal 1
uv run uvicorn mlservice.api.main:app --host 127.0.0.1 --port 8000 --no-access-log

# terminal 2 — PYTHONUTF8=1 is required on Windows: locust reads pyproject.toml
# with the ANSI codepage and chokes on the non-ASCII characters in its comments
PYTHONUTF8=1 uv run locust -f loadtest/locustfile.py --headless \
  --users 6 --spawn-rate 2 --run-time 60s \
  --host http://127.0.0.1:8000 --tags ramp
```

Scenarios: `smoke` (works at all), `steady` (latency profile with a realistic
mix), `ramp` (find the knee), `soak` (degradation over time).

## 9. Outstanding

- [ ] **Re-measure with the service containerised and the generator off-box.**
      Every number here is an upper bound until then.
- [ ] Soak run long enough to show or rule out drift over hours.
- [ ] Multi-worker measurement — all of this is a single uvicorn worker.
- [ ] Optimise `ColumnTransformer` if the budget proves too tight. It is 98% of
      the model call, so it is the only target worth attacking first.
