# Runbook

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

Incident response for the readmission risk service. One section per scenario:
**symptom → first check → diagnostics → likely causes, ranked → remediation →
escalation → post-incident**.

Every command here is one that runs. Every threshold is read from
[`configs/thresholds.yaml`](../configs/thresholds.yaml); where a number appears
in prose it is quoted from there, and if the two disagree the config wins.

**Honesty note.** This service has never taken production traffic. The
procedures below are derived from the system as built — real endpoints, real
metric names, real thresholds — and the failure modes are ones the code can
actually produce. But no incident here has been *lived through*. Where a step
has never been executed against a running system, it says so.

---

## 0. Orientation — read this first at 3am

**The three questions, in order.** Answering them out of order wastes the most
expensive minutes of an incident.

1. **Is it serving?** `curl -fsS localhost:8000/health/ready` → 200 or not.
2. **Is it serving *correctly*?** Ready only means a model loaded and scored a
   canary row. It says nothing about whether the model is the right one.
3. **Is it serving the *right model*?** `curl -s localhost:8000/v1/model` →
   version and source.

**The two levers.** Almost every remediation below ends at one of these:

| Lever | Command | Undoes |
|---|---|---|
| Model | `uv run mlservice retrain rollback --reason "..."` | A bad model. Alias flip, no deploy, seconds. |
| Container | `kubectl rollout undo deployment/readmission-api` | A bad image — dependency, config, serving code. |

Reach for them early. The rollback budget is **120 seconds**, and it is set
there precisely so that rolling back is cheaper than diagnosing. **Diagnose
after you have stopped the bleeding, not before.**

> **Verification status of the two levers.** The *model* lever is genuinely
> exercised — `mlservice retrain verify-rollback` runs a real promote → rollback
> cycle against a real registry, and it is a required CI check. The *container*
> lever is not: every `kubectl` command in this runbook is written against the
> committed manifests but has never been run, because Docker is not installed on
> the development machine. Treat them as reviewed, not proven.

**Dashboards** (`docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d`,
then http://localhost:3000):

| Dashboard | Read it when |
|---|---|
| **ML Service — Golden Signals** | Anything. Four panels, ten-second read. |
| **ML Service — Model Health** | The service is up but the predictions look wrong. |
| **ML Service — Drift** | An alert mentions drift, or performance is decaying. |

---

## 1. Latency spike

**Alert:** `LatencyP99AboveSLO` (page, p99 > 700 ms for 5 m) ·
`LatencyP99Degrading` (ticket, p99 > 525 ms for 30 m)

### Symptom
`Latency p99` on **Golden Signals** is red. Requests are slow but succeeding.

### First check — is it the model or the API?

```bash
curl -s localhost:8000/metrics | grep -E "model_inference_duration_seconds_(sum|count)"
curl -s localhost:8000/metrics | grep -E "http_request_duration_seconds_(sum|count)"
```

`model_inference_duration_seconds` exists precisely to answer this question
without guessing. A regularised logistic regression on ~50 features is a dot
product — inference is microseconds.

**If inference time is flat and total time is up, the model is not your problem.**
That is the common case, and chasing the model here costs an hour.

### Diagnostics

```bash
# Are we simply over the measured knee? Saturation beats every other theory.
curl -s localhost:8000/metrics | grep -E "^http_requests_total" | head

# Per-endpoint: is it /v1/predict, or /v1/predict/batch dragging the aggregate?
# Golden Signals -> "Latency percentiles" and "Request rate by endpoint"

docker stats --no-stream          # CPU throttling / memory pressure
docker compose logs --tail=100 api | grep -iE "slow|timeout|retry"
```

### Likely causes, ranked

1. **Load above the measured knee (52 RPS).** `ApproachingSaturation` fires at
   42 RPS for 10 m. The service was measured, not guessed — see
   [LOAD_TEST_REPORT.md](LOAD_TEST_REPORT.md). Above the knee, latency is
   supposed to rise.
2. **Batch requests skewing the aggregate.** One 500-row batch is one slow
   request in the histogram. Check `Request rate by endpoint`.
3. **CPU throttling.** The k8s manifest sets no CPU limit on purpose — CFS
   throttling turns a busy moment into p99 spikes. If a limit was added, that
   is your cause.
4. **Memory pressure / GC.** Check `docker stats` against the 512Mi limit.
5. **The model itself.** Last, and only if `model_inference_duration_seconds`
   actually moved. Retraining will not fix a slow container.

### Remediation

- Over the knee → scale out. `kubectl scale deployment/readmission-api --replicas=4`
  (HPA does this at 70% CPU, but only if metrics-server is installed — kind does
  not ship it, and without it the HPA reports `<unknown>/70%` and does nothing).
- Batch skew → not an incident. Note it and move on.
- Inference genuinely slow → roll the model back and diagnose off the hot path.

### Escalation
Latency alone does not justify a rollback if requests are succeeding. It becomes
an incident when the error rate follows — go to §2.

### Post-incident
If the knee moved, **re-measure rather than re-guess**. The latency block in
`thresholds.yaml` carries `remeasure_required: true` and
`measurement_caveat: "load generator co-located with the service"` — the numbers
have a known weakness, and an incident is evidence about where the real one is.

---

## 2. Error-rate spike

**Alert:** `ErrorBudgetBurningFast` (page, 14.4× burn over 1 h) ·
`ErrorBudgetBurningSlowly` (ticket, 6× over 6 h)

These are Google SRE multiwindow multi-burn-rate alerts — a citable standard,
not invented numbers. 14.4× burn exhausts a 30-day budget in ~2 days.

### Symptom
`Error rate (5xx)` is red on **Golden Signals**. Requests are failing.

### First check

```bash
curl -is localhost:8000/health/ready | head -1     # 200, or 503?
curl -s  localhost:8000/v1/model | head -c 300     # which model, and from where?
curl -s  localhost:8000/metrics | grep -E 'http_requests_total.*status="5'
```

### Diagnostics

```bash
# 4xx is NOT in the error budget (exclude_4xx: true). A validation-error flood
# is an upstream contract break, not a service failure -> §6.
curl -s localhost:8000/metrics | grep -E "^validation_errors_total"

docker compose logs --tail=200 api | grep -iE "error|traceback|exception"
docker compose ps                                  # is anything unhealthy?
```

### Likely causes, ranked

1. **The model is not loaded.** `ModelNotLoaded` fires after 2 m. Readiness
   returns 503 by design — see §3.
2. **A bad model was promoted.** Check `uv run mlservice retrain history -n 5`
   against the time the errors started. If they line up, stop reading and roll
   back.
3. **A dependency is down.** MLflow being down must *not* cause this — the API
   loads a baked-in fallback artifact for exactly this reason. If MLflow being
   down does take the API down, that is a bug worth its own issue.
4. **Genuine application error.** Read the traceback in the logs.

### Remediation

```bash
# If a promotion correlates with the onset. Do this FIRST; diagnose after.
uv run mlservice retrain rollback --reason "5xx spike after promotion"

# If the image is the problem rather than the model:
kubectl rollout undo deployment/readmission-api
```

### Escalation
Both levers pulled and errors continue → the fault is not in the model or the
image. Stop rolling things back and read the tracebacks.

### Post-incident
If a bad model reached production, **the gates missed it** — that is the real
finding, not the model. Go to §7.

---

## 3. Readiness flapping

**Alert:** `ModelNotLoaded` (2 m) · `ServingFromLocalFallback` (15 m)

### Symptom
Pods cycle in and out of the Service endpoints. `/health/live` stays 200 while
`/health/ready` alternates.

### First check

```bash
kubectl get pods -l app=readmission-api -w
kubectl describe pod -l app=readmission-api | grep -A5 -iE "readiness|liveness"
curl -is localhost:8000/health/ready
```

### The distinction that matters

`/health/live` answers **"is the process wedged?"** — restarting fixes that.
`/health/ready` answers **"can it serve traffic?"** — restarting does *not*.

Readiness fails when the model is absent or its canary inference fails. **That is
the system working.** The pod is held out of rotation, no traffic reaches it,
and the previous ReplicaSet keeps serving.

**Flapping means readiness is oscillating, which is different from readiness
failing.** A steady 503 is containment. Oscillation means something intermittent.

### Likely causes, ranked

1. **The registry is intermittently reachable.** Each reload attempt succeeds or
   falls back depending on timing. `ServingFromLocalFallback` firing alongside
   is the tell.
2. **Readiness probe too aggressive for a cold start.** The manifest sets a
   `startupProbe` (30 × 2 s = 60 s of grace) specifically so a slow model load is
   not mistaken for a hang. If that was removed, this is your cause.
3. **Memory pressure causing restarts.** `kubectl describe pod` → `OOMKilled`.
4. **Liveness wired to a readiness-style check.** The classic Kubernetes ML
   mistake: the pod is killed for failing a check a restart cannot satisfy,
   producing a crash loop. Verify liveness points at `/health/live` and nothing
   else.

### Remediation

```bash
# Pin to the fallback and stop the oscillation, then fix the registry calmly.
kubectl set env deployment/readmission-api MLSERVICE_MLFLOW__TRACKING_URI=""
```

Serving from the local fallback is a **degraded but correct** state. It is not
an outage. Availability was deliberately decoupled from the tracking server.

### Escalation
Flapping with the registry disabled → the artifact itself is bad. Roll back.

### Post-incident
Was the startup probe's 60 s grace enough? If a cold start now takes longer,
raise `failureThreshold` rather than loosening liveness.

---

## 4. Drift alert fired

**Alert:** `PredictionDistributionShifted` (mean predicted probability moved
> 25% for 1 h)

### Symptom
`Features breaching` on the **Drift** dashboard is above 3, or the prediction
distribution moved.

### First check — is this real drift or induced demo drift?

```bash
uv run mlservice monitor check
uv run mlservice retrain check          # reports HOW MANY windows it evaluated
```

**Read the window count.** `retrain check` prints
`evaluated N drift window(s)` and says so explicitly when N is zero. Zero
windows means *nothing was examined* — which is not the same as *no drift was
found*, and the tool refuses to let those read alike.

Every drift report carries `drift_origin`: `real` (the genuine 1999→2008 shift
in medication mix and specialty recording) or `induced` (Phase 6 demo
resampling). **Check it before treating a demo artefact as an incident.**

### Diagnostics

```bash
uv run mlservice monitor check          # per-feature PSI vs its own threshold
# Drift dashboard -> "PSI by feature" and "Per-feature drift"
```

Thresholds are **per-feature and empirically calibrated**: each is the 99th
percentile of that feature's PSI between stable training windows, clamped to
[0.10, 0.25]. So a breach means *this feature moved more than it did 99% of the
time in data we accepted* — not that it crossed a generic constant.

### Likely causes, ranked

1. **Induced demo drift.** Check `drift_origin` first. Costs five seconds and
   ends the investigation.
2. **A real population shift.** Legitimate. Drift is a reason to *look*, not
   proof the model got worse.
3. **An upstream schema break masquerading as drift.** A feature going from 2%
   to 40% missing looks like drift and is a pipeline bug. Check
   `feature_schema_hash` and `comparable` in the report.
4. **Sampling noise.** Requires 3+ features breaching in **2 consecutive
   windows** to confirm. With 43 features at a 99th-percentile threshold, ~0.4
   breach per window by chance.

### Remediation

**Drift alone does not justify retraining.** Drift is leading; performance on
matured labels is the ground truth. A system that retrains on drift alone will
retrain on population changes the model handles perfectly well.

```bash
uv run mlservice retrain check          # does the performance trigger agree?
```

- Confirmed drift **and** performance degradation → retrain (§5).
- Confirmed drift, performance fine → **record it and wait.** This is a
  legitimate outcome, not an unresolved incident.
- Schema break → fix the pipeline. Retraining bakes the break in permanently.

### Post-incident
If the same feature breaches repeatedly without performance moving, its
threshold is too tight. Re-run `mlservice monitor calibrate` — do not hand-edit
the number, or you lose the property that every threshold traces to data.

---

## 5. Model degradation on matured labels

**This is the signal that actually matters.** Everything else is a proxy for it.

### Symptom
`Rolling PR-AUC` on the **Drift** dashboard falls more than 2 bootstrap standard
errors below the test-set baseline.

### First check

```bash
uv run mlservice retrain check
```

Expressed in **standard errors, not points**, because a 0.02 PR-AUC drop means
something entirely different at n=500 than at n=5000.

### Diagnostics

```bash
uv run mlservice monitor check          # includes the label-drift block
# Model Health -> "Outcomes recorded"; Drift -> "Hours since last label"
```

**Check label sufficiency before believing the metric.** The performance trigger
checks label count *before* degradation for exactly this reason — otherwise a
label pipeline delivering twelve rows of noise trips a retrain, which is an
upstream failure laundered into a model decision.

### Likely causes, ranked

1. **The label pipeline is broken and the labels are unrepresentative.** Check
   §6 first. A biased sample of labels looks exactly like degradation.
2. **Genuine decay.** The honest and expected case over time.
3. **Population shift the model does not handle.** Cross-reference §4.
4. **A silently corrupted feature pipeline.** Run the behaviour suite:
   `uv run pytest tests/behavior -m behavior`. A transform that drops
   `number_inpatient` leaves PR-AUC nearly intact while blinding the model to
   its strongest predictor — only a directional test notices.

### Remediation

```bash
uv run mlservice retrain check --manual --by "<your name>"
# then: train, collect evidence, run the gates
uv run mlservice train run
uv run pytest tests/behavior -m behavior -q --json-report --json-report-file=b.json
uv run pytest tests/data     -m data     -q --json-report --json-report-file=d.json
uv run mlservice retrain evidence --behavioral b.json --data-quality d.json \
  --out reports/challenger_evidence.json
uv run mlservice retrain gates \
  --challenger reports/challenger_evidence.json \
  --incumbent  reports/incumbent_evidence.json \
  --out        reports/gate_decision.json
```

The gates decide. If the challenger is blocked, **the incumbent stays** — a
degraded model you understand beats an unvalidated replacement.

### Escalation
Challenger blocked and the incumbent keeps degrading → this is a modelling
problem, not an operations one. It needs analysis, not another retrain.

### Post-incident
Record the decay rate. Two data points make a trend, and a trend tells you
whether the 30-day scheduled refresh is the right cadence.

---

## 6. Label pipeline stopped

**Alert:** `NoOutcomesRecorded` (no outcomes in 5 days) ·
`PredictionLogWriteFailures`

**Silent label-pipeline failure is the most common way monitoring dies quietly.**
Nothing looks wrong. Every dashboard is green. The performance signal simply
stops updating, and you keep trusting a number that stopped moving weeks ago.

### Symptom
`Outcomes recorded` on **Model Health** is flat. `Hours since last label` on
**Drift** climbs without bound.

### First check

```bash
curl -s localhost:8000/metrics | grep -E "outcomes_recorded_total"
curl -s localhost:8000/metrics | grep -E "prediction_log_writes_total"
```

### Diagnostics

```bash
docker compose logs --tail=100 api | grep -i "prediction_log"
ls -la logs/ 2>/dev/null                # is the log being written at all?
df -h .                                 # disk full is a classic silent cause
```

### Likely causes, ranked

1. **Nobody is POSTing to `/v1/outcomes`.** The most likely cause by far, and it
   is upstream of this service entirely.
2. **Disk full / read-only filesystem.** The k8s manifest sets
   `readOnlyRootFilesystem: true` with an explicit `emptyDir` for `/app/logs`.
   If that mount is missing, every write fails.
3. **Maturation window not yet elapsed.** A 30-day readmission label cannot
   exist before 30 days. Confirm the window before declaring an incident.
4. **`prediction_id` join failing** — outcomes arriving but matching nothing.

### Remediation
Fix the writer. There is nothing to fix in this service if nobody is sending
outcomes.

**Critically: while labels are stale, treat the performance signal as absent,
not as passing.** The performance trigger already does this — it reports
"insufficient matured labels" rather than silently not firing. Do not let a
green dashboard imply a healthy model.

### Post-incident
How long was it broken before anyone noticed? If the answer is longer than the
5-day alert window, the alert is not the problem — nobody was reading it.

---

## 7. A bad model reached production

The one that matters most, because it means **the gates failed**.

### Symptom
Predictions are wrong, badly calibrated, or the subgroup gaps widened — and the
model passed promotion.

### First check

```bash
uv run mlservice retrain history -n 10       # what shipped, when, on whose approval
curl -s localhost:8000/v1/model
```

### Remediation — do this before diagnosing

```bash
uv run mlservice retrain rollback --reason "bad model in production: <what you saw>"
```

Rollback is **ungated**. It works when everything else is broken; that is the
whole point. The budget is 120 seconds.

Verify:

```bash
curl -s localhost:8000/v1/model | grep -i version
```

### Then: how did it get through?

This is the post-incident that matters. Read `reports/gate_decision.json` from
the promotion and ask which gate *should* have caught it:

| What was wrong | Gate that should have caught it |
|---|---|
| Worse ranking | `performance` |
| Miscalibrated probabilities | `calibration` |
| Widened subgroup disparity | `subgroup` |
| Corrupted feature pipeline | `behavioral` |
| Wrong feature contract | `operational` |
| Trained on broken data | `data_quality` |

Then one of three things is true, and they need different fixes:

1. **A gate's threshold was too loose.** Tighten it in a reviewed commit.
2. **A gate had no evidence and was skipped.** It should have *blocked* —
   `evidence.py` leaves uncollected evidence absent rather than assumed, so
   check whether something defaulted it to passing.
3. **No gate covers this failure.** The most valuable finding. Write the gate,
   and write the test that proves it blocks.

### Escalation
If the answer is (3), that is a design gap, not an incident. It gets an ADR.

---

## 8. Retraining failed

### Symptom
The `Retrain` workflow failed, or `mlservice retrain gates` errored rather than
returning a decision.

### First check — was it *blocked* or did it *fail*?

**These are completely different and the distinction is the whole section.**

```bash
cat reports/gate_decision.json
```

- **`"promote": false`** → the challenger was **blocked**. The system worked.
  Exit code 1 from `retrain gates` means blocked, not broken, and the workflow
  deliberately does not fail on it. **No action needed** beyond understanding
  which gate fired and why.
- **The workflow errored** → something is actually broken. Continue below.

### Diagnostics

```bash
uv run mlservice doctor                 # environment sanity; survives a broken config
uv run mlservice data download          # checksum verification
uv run mlservice retrain check --json
```

### Likely causes, ranked

1. **Missing gate evidence.** `retrain evidence` prints
   `gap  <name> evidence absent — that gate will BLOCK, by design`. Collect it;
   do not default it.
2. **Dataset download or checksum failure.** The UCI URL uses literal hyphens in
   `130-us` and `1999-2008` — a wrong slug 404s.
3. **The registry is unreachable.** Training falls back to the local SQLite
   store and logs the substitution loudly. A failed tracking call is not a
   reason to lose a training run.
4. **The rollback-path check failed.** The workflow verifies rollback *before*
   proposing any promotion, because the undo lever is a precondition for the
   change, not a follow-up task. If this fails, **fix rollback before shipping
   anything.**

### Remediation
Retraining failing is not urgent. The incumbent keeps serving. Fix it in
business hours.

### Escalation
`uv run mlservice retrain verify-rollback` failing is the one exception — that
means the safety net is gone, and it should be treated as urgent even though
nothing is currently broken.

---

## 9. Service down

**Alert:** `ServiceDown` (cannot scrape for 2 m) · `NoTrafficReceived`

### First check

```bash
docker compose ps
curl -is localhost:8000/health/live | head -1
docker compose logs --tail=50 api
```

### Remediation

```bash
docker compose up -d api
docker compose restart api            # only if live is failing — see §3 first
```

`NoTrafficReceived` on a demo service usually means nobody is using it, not that
it is broken. Confirm against `ServiceDown` before treating it as an incident.

---

## What this runbook does not cover

Stated rather than left to be discovered:

- **Canary rollout failures.** The canary is configured and has never been
  exercised — Docker is not installed on the development machine, so weighted
  traffic splitting has never run. There is no procedure here because there is
  no experience behind one.
- **Multi-instance / split-brain scenarios.** The prediction log is per-instance
  NDJSON. Aggregating across replicas is unsolved and out of scope.
- **Data-privacy incidents.** The service holds no PII: it reads a public,
  de-identified dataset and stores no patient identifiers beyond what a caller
  sends.
- **Anything about clinical impact.** This model is not clinically validated and
  must not inform care. There is no clinical escalation path because there must
  be no clinical use.

---

## Related

- [MONITORING.md](MONITORING.md) — every metric and the derivation of every threshold
- [RETRAINING_POLICY.md](RETRAINING_POLICY.md) — triggers, gates, promotion, rollback
- [LOAD_TEST_REPORT.md](LOAD_TEST_REPORT.md) — measured latency and the knee at 52 RPS
- [ARCHITECTURE.md](ARCHITECTURE.md) — components and data flow
