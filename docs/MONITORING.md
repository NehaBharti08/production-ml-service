# Monitoring Design

> **NOT FOR CLINICAL USE.** This document describes monitoring for an
> engineering demonstration. Nothing here is clinically validated.

Every threshold in this project answers "why that number?" with something other
than "it looked about right". Arbitrary thresholds are the clearest sign that
monitoring was copied rather than reasoned about, so this document exists to
show the derivation for each one.

`configs/thresholds.yaml` is the single source of truth. Alert rules, dashboards
and promotion gates all read from it, and every value carries a provenance tag:

| Tag | Meaning |
|:--|:--|
| `MEASURED` | Observed in this repo; `source` names where |
| `DERIVED` | Computed from a `MEASURED` value by a stated rule |
| `STANDARD` | External citable convention; `source` names it |
| `PLACEHOLDER` | Not yet real — **must not be used for alerting** |

`mlservice doctor` reports how many `PLACEHOLDER` blocks remain. One does
(`drift.per_feature`, owned by Phase 6).

---

## 1. What is measured

| Metric | Type | Labels | Why it exists |
|:--|:--|:--|:--|
| `http_requests_total` | Counter | method, endpoint, status | Traffic and error rate |
| `http_request_duration_seconds` | Histogram | method, endpoint | Latency percentiles |
| `model_inference_duration_seconds` | Histogram | model_version | Separates model cost from API cost |
| `model_predictions_total` | Counter | model_version, predicted_label | Prediction-mix shift |
| `model_predicted_proba` | Histogram | model_version | Score distribution — the earliest drift signal |
| `model_load_status` | Gauge | — | Drives readiness |
| `model_serving_info` | Info | name, version, source | Registry vs fallback |
| `validation_errors_total` | Counter | **field** | Upstream schema breakage, per field |
| `prediction_log_writes_total` | Counter | result | Monitoring pipeline health |
| `outcomes_recorded_total` | Counter | — | Label pipeline watchdog |
| `drift_score` | Gauge | feature, method | Per-feature drift (Phase 6) |
| `drift_features_breaching` | Gauge | — | Single number that drives the drift alert |
| `rolling_pr_auc` | Gauge | window | True performance, once labels mature |

Three of these earn specific comment:

**`validation_errors_total` is labelled by field.** A spike concentrated on one
field is an upstream producer changing its schema; the same volume spread across
many fields is a caller sending junk. Those need different responses, and an
unlabelled counter cannot distinguish them.

**`model_inference_duration_seconds` is separate from request latency.** When p99
rises, the first question is whether the model got slower or the layer around it
did. Phase 4 measured the model at ~1 ms and the feature transform at ~64 ms, so
this separation is what makes that diagnosable rather than a guess.

**Status labels are the status *class*, not the code.** Dashboards and the error
budget care about 5xx versus 4xx; per-code series multiply cardinality for no
operational gain.

### Two cardinality rules

Endpoint labels use the **route template**, never the raw path. A label that
grows with traffic eventually exhausts Prometheus's memory. There is no path
parameter today, but the habit is cheap and the failure is expensive.

This was not theoretical: an early version read the route template *before*
`call_next`, so every metric was labelled `endpoint="unmatched"` and per-endpoint
dashboards would have been useless. Caught by reading real `/metrics` output, and
now regression-tested.

### Latency buckets are chosen, not defaulted

Prometheus's default buckets top out coarsely around 10 s. On an endpoint whose
p99 sits in the tens of milliseconds, nearly every observation lands in the first
bucket and the histogram cannot resolve p95 from p99 at all.

These buckets are dense between 1 ms and 250 ms, where the distribution actually
lives, with 1 s and 5 s retained only to catch pathology.

---

## 2. Service level objectives

### Availability: 99.5% over 30 days

**Provenance: `STANDARD`.** 99.5% allows a 216-minute error budget per 30 days.

Chosen over 99.9% (43 minutes) deliberately: this is a single-replica free-tier
demo whose infrastructure legitimately restarts. Promising three nines on it
would be a fake SLO, and a fake SLO is worse than none — it gets breached in
normal operation and then everybody learns to ignore it.

### Burn-rate alerting

**Provenance: `STANDARD`** — Google SRE Workbook, Ch. 5, multiwindow
multi-burn-rate. The factors are not invented:

| Alert | Long window | Short window | Factor | Budget consumed |
|:--|:--|:--|--:|:--|
| Page | 1 h | 5 m | 14.4× | 2% in one hour |
| Ticket | 6 h | 30 m | 6× | 5% in six hours |

**Both windows must be breached.** The short window is what stops a resolved
incident from paging for another hour: without it, a burst ending at minute 5
keeps the 1-hour rate elevated for 55 more minutes and the alert continues firing
long after there is anything to do.

### Latency

**Provenance: `MEASURED`.** Source: `docs/LOAD_TEST_REPORT.md`.

The derivation rule was fixed in config **before any measurement was taken**,
specifically so the thresholds could not be reverse-engineered to whatever the
service happened to do:

```
slo_p99   = 2   × measured p99 @ target_rps
page      = slo_p99, breached for 5 consecutive minutes
ticket    = 1.5 × measured p99, sustained for 30 minutes
```

| Setting | Value | Where it came from |
|:--|--:|:--|
| Measured p99 | 350 ms | Steady-state run, 6 users |
| `slo_p99_ms` | **700** | 2 × 350 |
| `ticket_p99_ms` | 525 | 1.5 × 350 |
| `target_rps` | 25 | ~half the measured knee |
| `measured_knee_rps` | 52 | Ramp sweep |

**That discipline earned its keep.** The plan estimated p50 of 4–8 ms; measured
p50 is ~130 ms. The model is 1 ms and `ColumnTransformer.transform` is 64 ms.
Chosen after seeing results, the rule would have been fitted to that behaviour
and hidden the finding entirely.

**`for: 5m` on the page** survives a GC pause and scrape jitter. A single slow
scrape is not an incident, and paging on one teaches people to dismiss the alert.

### A caveat that is recorded, not buried

The load generator ran on **the same machine as the service**, so every latency
number includes contention between the measurement and the thing measured.
Repeat runs varied **2–3×** with machine state.

- **Trustworthy:** the shape — where the knee is, and that throughput inverts
  past it. The sweep ran back to back under comparable conditions.
- **Not trustworthy:** the absolute numbers as a characterisation. They are an
  upper bound.

`configs/thresholds.yaml` carries `remeasure_required: true`. The SLO is
deliberately loose for this reason: a tight SLO derived from a contended
measurement pages on healthy behaviour, which is worse than no SLO.

### Saturation

**Provenance: `MEASURED`.** The ramp sweep found the knee between 20 and 40
concurrent users, with an unambiguous signature:

| Users | Median ms | req/s |
|--:|--:|--:|
| 10 | 68 | 33.5 |
| 20 | 110 | **52.3** |
| 40 | 500 | 47.6 ⚠ |

Past the knee **throughput inverts** — work queues rather than completing.
Adding load makes the service slower *and* less productive.

The alert fires at **42 req/s (80% of the knee)**, deliberately *before* that
regime. An alert at the knee arrives after the damage.

---

## 3. Drift detection

### Effect sizes, not p-values

This is the decision that most affects whether drift alerting is usable.

With ~20,000 rows in a monitoring window, a Kolmogorov–Smirnov test returns
p < 0.05 for shifts far too small to act on. **Statistical significance scales
with n; practical significance does not.** Alerting on K-S p-values at this
sample size guarantees a permanently red dashboard, which trains the operator to
ignore it.

So the primary measure is **PSI**, with Jensen–Shannon distance secondary. Both
are effect sizes and neither has this failure mode.

### Thresholds calibrated against an empirical null

The credit-risk convention (PSI < 0.1 stable, 0.1–0.2 moderate, > 0.2
significant) is defensible but generic — it says nothing about how much *this*
dataset's features naturally churn.

So Phase 6 measures that:

1. Split the training period into 20 consecutive equal windows — periods already
   accepted as stable, by having trained on them.
2. Compute PSI per feature between adjacent windows.
3. That distribution **is** this dataset's normal churn.
4. `threshold[feature] = clamp(percentile_99(null_psi[feature]), 0.10, 0.25)`

Every threshold then answers "why that number?" with **"because this feature
moved that much between stable training windows only 1% of the time"** — a
per-feature, data-derived answer.

The clamp stops a noisy feature setting an absurdly permissive bar and a frozen
one setting a hair-trigger. Floor and ceiling are the `STANDARD` credit-risk
values.

### Alert conditions

| Condition | Rule | Why |
|:--|:--|:--|
| Data drift | ≥ 3 features breaching, **2 consecutive windows** | With ~45 features at a 99th-percentile null threshold, ~0.45 features breach per window by chance. Requiring 3 puts the false-alarm rate into the low percent. |
| Prediction drift | mean score outside the training bootstrap 99% CI, **or** alert rate moving > 25% relative | Two signals because they fail differently. The alert rate is what an operator feels first, because it drives downstream workload. |
| Delayed-label drift | rolling PR-AUC below test PR-AUC by > 2 bootstrap SE | Expressed in SEs, not points: a 0.02 drop means something entirely different at n=500 than at n=5000. |
| Label pipeline stalled | no matured label in the maturation window plus slack | The most common way monitoring dies silently. |

**Two-window confirmation is what kills flapping.** A single-window blip is
noise, and paging on noise is how a pager gets ignored.

---

## 4. Alert design

13 alerts across 5 groups. **4 page, 9 ticket** — most alerts should not wake
anyone, because a pager that fires constantly gets silenced, and then the one
that mattered is silenced too.

Exactly **two severity levels**:

- **page** — act now; the budget is burning fast enough that waiting until
  morning breaches it
- **ticket** — act in working hours; something is degrading but the budget
  survives the night

A third level always becomes one nobody reads.

### Every alert links to a runbook

Each carries a `runbook_url` pointing at the section of `docs/RUNBOOK.md` that
says what to **do**. An alert with no documented response is a notification, and
notifications train people to ignore pages. This is enforced by a test, not by
convention.

### Alert on symptoms, not causes

"p99 exceeds the SLO" is actionable. "CPU is at 80%" is not — 80% CPU is fine if
latency is fine and irrelevant if it is not.

### The alerts that watch the monitoring

The most valuable group, because silent monitoring failure looks exactly like
health:

- **`ServiceDown`** uses `up`, which comes from Prometheus rather than the
  application, so it still fires when the app is too broken to expose metrics —
  precisely when every application-level alert goes quiet.
- **`NoOutcomesRecorded`** watches the label pipeline. Without matured labels,
  rolling PR-AUC freezes at its last value and looks healthy *because* nothing
  is updating it.
- **`PredictionLogWriteFailures`** — a failed write deliberately does not fail
  the request (losing a log line is a monitoring gap; failing the request is an
  outage), so the gap has to be visible some other way.
- **`NoTrafficReceived`** — with zero traffic every `rate()` is legitimately
  zero and nothing else can fire.
- **`ServingFromLocalFallback`** is a ticket, not a page: the fallback working
  is the design. But it means the registry is unreachable, so **promotion and
  rollback would silently have no effect** — worth knowing before you need them.

---

## 5. Dashboards

Three, provisioned from committed JSON. A dashboard configured through the UI
lives only in a Grafana volume: not committed, not reviewable, and gone with
`make clean`.

They are **generated** by `scripts/build_dashboards.py` from
`configs/thresholds.yaml`, so a threshold change moves the red band
automatically. CI regenerates and diffs them, failing if the committed JSON has
gone stale.

### Golden signals — readable in ten seconds

Four tiles on the top row, answering in order: **is anyone using it, is it fast,
is it failing, is it near capacity.** Someone who has never seen this service
should be able to say "fine" or "not fine" without scrolling or reading a legend.

Detail sits below for anyone who wants it.

### Colour never carries meaning alone

Every tile shows its **value and its name as text**, with a unit. A red square
that does not say what is red, or by how much, communicates nothing to someone
who has not memorised the layout — and nothing at all to a colourblind viewer.

The status palette is validated for CVD separation and contrast against both
light and dark surfaces rather than picked by eye.

### No dual-axis panels, anywhere

Two y-scales on one panel is the most common charting mistake: it lets any two
series be made to look correlated by choosing scales. Two measures of different
magnitude get two panels. Enforced by a test.

### UTC, always

An incident spans time zones, and correlating a dashboard against UTC log
timestamps should not require arithmetic.

---

## 6. What is not verified yet

Stated plainly rather than implied:

- **No screenshots.** Docker is not installed on the development machine, so
  Grafana has never rendered these dashboards. The JSON is validated
  structurally and by test, but "readable in ten seconds" is a claim about
  something nobody has looked at yet.
- **Alert rules have never fired against real data.** CI validates every PromQL
  expression with `promtool` — which is genuine verification that they parse and
  are well formed, and is where a typo would otherwise ship silently and simply
  never fire. But firing behaviour is unproven.
- **Drift metrics do not exist yet.** The drift dashboard reads metrics the
  Phase 6 monitoring job will export. It is built now so the metric contract is
  fixed before the job is written, rather than the dashboard being shaped around
  whatever the job happened to emit.
- **Latency numbers need re-measuring off-box.** See the caveat in §2.
