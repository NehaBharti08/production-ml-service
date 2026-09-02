# Retraining Policy

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

Every number in this document is read from [`configs/thresholds.yaml`](../configs/thresholds.yaml)
and executed by [`src/mlservice/retraining/`](../src/mlservice/retraining/). Nothing here is
documentation-only: if the document and the code disagree, the code is what runs, and the
tests in [`tests/unit/test_gates.py`](../tests/unit/test_gates.py) and
[`tests/unit/test_retraining.py`](../tests/unit/test_retraining.py) fail.

---

## 1. The shape of the policy

```
  triggers  ─────►  retrain  ─────►  gates  ─────►  promote  ─────►  rollback
  (when)            (what)           (whether)      (how)            (undo)

  4 triggers        full pipeline    6 gates        alias flip       alias flip
  any one fires     from raw data    ALL must pass  + audit record   + audit record
```

Two asymmetries are deliberate and worth stating up front, because they are the
parts most often got backwards:

**Triggering is permissive; promoting is strict.** Any one trigger starts a retrain,
but every gate must pass to ship the result. Training a model is cheap and reversible;
serving one is neither. Making the entrance narrow and the exit wide would be exactly
the wrong way round.

**Rollback is not gated.** Promotion is guarded by six gates; rollback has none. A
safety mechanism that can be blocked by the checks it exists to escape is not a safety
mechanism. `promote(gates_passed=False)` raises; `rollback()` always runs.

---

## 2. Triggers — when to retrain

Implemented in [`trigger.py`](../src/mlservice/retraining/trigger.py). All four are
evaluated on every check, even after one fires — a short-circuit would leave the record
saying "we retrained on drift" when the truth was "we retrained on drift, and
performance was also degrading". Those are different incidents.

| Trigger | Condition | Why |
|---|---|---|
| **Scheduled** | Model older than 30 days | Decay can be slow enough that no single window breaches while the cumulative shift is real. A schedule bounds that. |
| **Drift** | Data-drift alert **confirmed** across 2 consecutive windows | Leading indicator. Says the input population moved — a reason to look, not proof the model got worse. |
| **Performance** | Rolling PR-AUC more than 2 bootstrap SEs below baseline, on matured labels | Lagging ground truth. The only trigger that observes actual performance. |
| **Manual** | `--manual --by <name>` | An automated system that cannot be driven by hand is one people work around. The requester is recorded. |

### Ordering of trust

Drift is a *leading* signal and performance is the *lagging* one. A system that retrains
on drift alone will retrain on population changes the model handles perfectly well —
the input distribution moving is not the same as the model failing. So a performance
trigger firing is stronger evidence than a drift trigger firing, and the two are
reported separately rather than collapsed into one "retrain needed" boolean.

### Deliberately *not* triggers

Listed in `NOT_TRIGGERS` in the code and asserted by a test, so removing one is a
failure rather than a quiet loosening:

- **A single feature breaching in a single window.** With 43 features at a 99th-percentile
  threshold, ~0.4 features breach per window by chance alone. This is noise.
- **Prediction drift without a corresponding data-drift signal.** Scores moving without
  inputs moving usually means the serving path changed, not the world.
- **A latency or error-rate alert.** Those are serving problems. Retraining does not fix
  a slow container, and shipping a new model during an incident adds a variable.
- **An upstream schema break.** That needs a pipeline fix. Retraining on broken data
  bakes the break into the model — the one response that makes it permanent.

### The insufficient-labels guard

The performance trigger checks label *sufficiency before* it checks degradation. Order
matters: without that, a label pipeline delivering twelve rows of noise can trip a
retrain, which is an upstream failure being laundered into a model decision. When labels
are insufficient the trigger reports *why* rather than silently not firing.

```bash
uv run mlservice retrain check          # evaluate all four
uv run mlservice retrain check --json   # machine-readable, for CI
```

The command prints how many drift windows it evaluated. **Zero windows is reported
explicitly**, because "I examined nothing" and "I examined the evidence and found no
drift" must never render as the same reassuring line.

---

## 3. Promotion gates — whether to ship

Implemented in [`gates.py`](../src/mlservice/retraining/gates.py). A challenger is
promoted **only if all six pass**. Every gate is unit-tested with a challenger
constructed to fail *that gate and no other* — a bad model that trips five gates proves
nothing about which gate caught it.

`evaluate_promotion()` runs **every** gate even after one fails. Short-circuiting would
hide the other problems, and the operator reading the blocked-promotion log needs the
whole picture, not the first thing that happened to be checked.

### 3.1 Performance — non-inferiority, not improvement

> challenger PR-AUC ≥ incumbent PR-AUC − **0.005**

Non-inferiority rather than strict improvement, because demanding improvement on every
refresh blocks legitimate freshness updates: a model retrained on newer data that
performs identically is a *good* outcome, and a gate that rejects it forces the model to
go stale. The margin is one-sided — the challenger may be better without limit.

### 3.2 Calibration — the thesis of the project

> Brier ≤ incumbent × **1.02**  **and**  ECE ≤ **0.05**

**This gate is why the repository exists.** A challenger that ranks patients better but
whose probabilities are systematically wrong is a *regression* for a health-adjacent use
case, and almost every promotion pipeline in the wild would ship it because AUC went up.

If a model says 30% it should be right about 30% of the time. When that breaks, everyone
downstream who reasoned about the number — a threshold, a triage cut-off, a human
reading "moderate risk" — is now wrong in a way no discrimination metric will show.

`test_blocks_a_better_ranking_but_worse_calibrated_model` is the test that pins this. It
constructs exactly that challenger — better PR-AUC, worse Brier — and asserts it is
refused.

Both conditions are required. Brier is relative (has calibration degraded against what
we already accepted?) and ECE is absolute (is it acceptable at all?). A challenger can
pass one and fail the other, and either failure is disqualifying.

### 3.3 Subgroup — relative to the incumbent, not to an invented constant

> worst subgroup gap (n ≥ **500**) must not widen by more than **20%** relative to the
> incumbent's existing gap

Measured against the incumbent rather than an absolute fairness threshold, because no
absolute number here is defensible — and an invented one is worse than none, since it
launders a guess as a standard. What *is* defensible: "this model does not make the
existing disparity materially worse."

The n ≥ 500 floor exists because subgroup metrics on small groups are mostly sampling
noise, and a gate driven by noise blocks good models at random.

This gate does not claim the model is fair. The incumbent's gaps are reported openly in
the [model card](MODEL_CARD.md), including where they are unflattering. The gate only
prevents them getting worse.

### 3.4 Behavioural — 100% of invariance and directional tests

> **all** Phase 4 behaviour tests pass — no partial credit

Catches a corrupted feature pipeline that every aggregate metric sails past. A model can
post an excellent PR-AUC while `number_inpatient` is silently mapped to the wrong
column; the directional test that asserts *more prior inpatient visits must not decrease
predicted risk* catches it, and the metric does not.

The pass rate is 1.0 rather than 0.95 because these encode clinical priors. "95% of our
clinical assumptions hold" is not a thing to be relaxed about.

### 3.5 Operational — can it actually serve?

> artifact loads · feature schema hash matches the serving contract · canary p99 within SLO

A model that scores well and cannot be loaded is not a model. The schema hash check is
the important one: it fails when the challenger expects a different feature contract
than the API sends, which is otherwise discovered in production as a 500 per request.

### 3.6 Data quality — was it trained on good data?

> data-quality suite passes · no feature's missingness up more than **10 percentage points**

The missingness check catches the upstream break that would otherwise be trained *into*
the model. A feature going from 2% to 40% missing means something broke in the pipeline;
retraining on it teaches the model that the broken state is normal.

```bash
uv run mlservice retrain gates \
  --challenger reports/challenger.json \
  --incumbent  reports/champion.json \
  --out        reports/gate_decision.json
```

Exit code 0 promotes, 1 blocks. The failing gate is named in the log with the numbers
that produced it — at 3am "the calibration gate failed" is not enough; the operator needs
the challenger's ECE, the incumbent's, and the limit, without re-running anything.

---

## 4. Promotion — how to ship

Promotion is a **single MLflow registry alias flip**. `champion` moves to the new
version; the API loads by alias, so the change takes effect on the next model load with
no deploy.

One atomic operation, not a multi-step state machine that can be interrupted halfway and
leave the system in a state nobody designed.

The gate decision is a **required argument** to `mlservice retrain promote`, not an
optional flag:

```bash
uv run mlservice retrain promote \
  --version 3 --decision reports/gate_decision.json \
  --trigger drift --approver neha
```

There is deliberately no `--force`. A promote command with a bypass flag is a promote
command with no gates, and the flag would be used at precisely the moment judgment is
worst — during an incident, under time pressure.

### The audit trail

Every promotion and rollback appends to `reports/promotion_history.ndjson`:
timestamp, action, from-version, to-version, trigger, gate result, approver. Append-only,
for the same reason the prediction log is — the history of what was deployed when is
evidence, and evidence that can be rewritten is not evidence.

**`from_version` is captured before the flip, not looked up afterwards.** Once the alias
moves, "what was it before" is no longer answerable from the registry, and reconstructing
it from run history is exactly the archaeology you do not want to be doing during an
incident. This is also what makes rollback a lookup instead of a search.

```bash
uv run mlservice retrain history -n 10
```

---

## 5. Rollback — how to undo

Two levers, because there are two layers that can be wrong:

| Layer | Mechanism | When |
|---|---|---|
| Model | `mlservice retrain rollback --reason "..."` | The model is bad. Alias flip, no deploy. |
| Container | `kubectl rollout undo deployment/readmission-api` | The image is bad — dependency, config, serving code. |

Budget: **120 seconds** to rollback. Not because 120 is magic, but because a rollback
path that takes longer than that stops being the first thing you reach for during an
incident, and a rollback nobody reaches for is not a control.

Rollback reads the target from the promotion history rather than the registry, because
the registry only knows where the alias points *now*.

### The rollback path is tested, not asserted

```bash
uv run mlservice retrain verify-rollback
```

This runs a real cycle against the registry: seed the alias to the oldest version,
promote the latest, roll back, and assert the alias both **moved** and **returned**. The
original alias target is restored on the way out, so running it never changes what is
serving.

The "and moved" half is not pedantry. The first implementation promoted whichever version
the registry happened to return first — which was the one already serving — and recorded
`promote 2 → 2, rollback 2 → 2, VERIFIED: True`. It passed while proving nothing. A
verification that cannot fail is decoration, so the current version fails explicitly on a
degenerate cycle, and `test_the_alias_genuinely_moved_in_between` pins the property.

**Verified output on the real registry (2026-08-29):**

```
seed      alias -> 1     champion is now: 1
promote   1 -> 2         champion is now: 2
rollback  2 -> 1         champion is now: 1

VERIFIED: True  --  alias moved 1 -> 2 and back to 1
```

---

## 6. Canary rollout

| Setting | Value | Why |
|---|---|---|
| Traffic share | 10% | Enough signal, bounded blast radius |
| Hold | **1000 predictions**, not a wall-clock duration | A time-based hold on a low-traffic demo promotes on almost no evidence |
| Auto-rollback | on any breach | Breach conditions listed in `thresholds.yaml` |

Holding by volume rather than by time is the decision worth defending: "wait 30 minutes"
and "wait for 1000 predictions" are the same thing at production traffic and wildly
different at demo traffic, where the first promotes after seeing almost nothing.

---

## 7. What is not verified

Stated plainly rather than left for a reader to discover:

- **The canary is designed and configured, not exercised.** Weighted traffic splitting
  needs the kind cluster, and Docker is not installed on this machine. The manifests are
  written and the thresholds are set; no canary rollout has actually run.
- **`kubectl rollout undo` has not been demonstrated.** Same reason. The *registry* half
  of rollback — the model-level lever — is genuinely verified above.
- **The scheduled trigger runs on a simulated clock.** The dataset spans 1999–2008 and
  has no timestamp column (see [ADR 0004](DECISIONS/0004-temporal-split-proxy.md)); the
  30-day cadence is evaluated against registry creation timestamps.
- **No retraining has been triggered end-to-end by drift in a live system.** The trigger
  fires correctly against real replay evidence — `retrain check` reports
  `RETRAIN: YES — fired: drift` from the induced-drift replay windows — but the
  full trigger → retrain → gate → promote loop has not run unattended.

---

## Related

- [ADR 0002 — Calibration as a deployment gate](DECISIONS/0002-calibration-as-deployment-gate.md)
- [ADR 0008 — Promotion gates and the rollback path](DECISIONS/0008-promotion-gates-and-rollback.md)
- [MONITORING.md](MONITORING.md) — where the drift and performance signals come from
- [RUNBOOK.md](RUNBOOK.md) — what to do when one of these fires
