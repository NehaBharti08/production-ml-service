# Kubernetes rollout and rollback — captured run

> **NOT FOR CLINICAL USE.** This document describes an engineering demonstration of ML operations. Nothing here is clinically validated or fit to inform patient care.

Output from an actual run, not a description of one.

| | |
|:--|:--|
| **Date** | 2026-09-04 |
| **Cluster** | kind v0.33.0, Kubernetes v1.37.0, single node |
| **Host** | Windows 11, Docker Desktop 4.89.0, 8 GiB to the Docker VM |
| **Script** | [`scripts/k8s_rollback_demo.sh`](../scripts/k8s_rollback_demo.sh) |

**The result in one line:** a broken deployment reached the cluster and served
**zero** requests.

Not because anyone was watching. Because readiness is a traffic gate, liveness
is a restart trigger, and they are wired to endpoints that answer those two
different questions.

---

## 1. Baseline

```console
$ kubectl get pods -l app=readmission-api
NAME                               READY   STATUS    RESTARTS   AGE
readmission-api-5bb855dd66-589g5   1/1     Running   0          15s
readmission-api-5bb855dd66-jzlln   1/1     Running   0          15s

$ curl -s http://localhost:18080/v1/model
  loaded              : True
  source              : local_fallback
  decision_threshold  : 0.10106382978723404
  feature_schema_hash : 06f5f0b873ca95f6
```

The threshold is the **trained** 0.1011, not the 0.5 config placeholder. That is
not incidental — see [§6](#6-what-this-run-found).

## 2. Deploy a broken change

```console
$ kubectl set env deployment/readmission-api \
    MLSERVICE_MODEL__LOCAL_FALLBACK=/app/models/champion/does-not-exist.joblib
deployment.apps/readmission-api env updated
```

Broken where it matters: the process starts and **liveness passes**, but the
model never loads so **readiness stays 503 forever**. A container that crashes
outright is caught by anything; one that runs happily while unable to do its job
is the case that needs a readiness probe wired to something real.

## 3. The rollout stalls — and the stall is the containment

```console
$ kubectl rollout status deployment/readmission-api --timeout=120s
Waiting for deployment "readmission-api" rollout to finish: 1 out of 2 new replicas have been updated...
error: timed out waiting for the condition
```

**This timeout is the system working.** `maxUnavailable: 0` means the old pods
are not removed until a new one becomes Ready. It never does, so the rollout
halts with the healthy version still serving.

```console
$ kubectl get pods -l app=readmission-api
NAME                               READY   STATUS    RESTARTS   AGE
readmission-api-55d98997bc-42wfl   0/1     Running   0          136m   <-- broken, never Ready
readmission-api-5bb855dd66-589g5   1/1     Running   0          137m
readmission-api-5bb855dd66-jzlln   1/1     Running   0          137m
```

The broken pod is `Running` — the process is alive — and `0/1` Ready. Note
`RESTARTS 0`: liveness is *not* killing it, because liveness points at
`/health/live`, which is correctly still 200. Had liveness been wired to
`/health/ready`, this pod would be in a crash loop, and the incident would look
like an outage instead of a blocked rollout.

## 4. Traffic during the failure

```console
$ for _ in $(seq 1 12); do curl -s -o /dev/null -w '%{http_code}' http://localhost:18080/health/ready; done
  12/12 probes returned 200 DURING the failed rollout
```

Counted, not sampled once. "It answered when I checked" is an anecdote.

And the mechanism behind it, not merely the symptom:

```console
$ kubectl get endpoints readmission-api
  ready endpoint: 10.244.0.5
  ready endpoint: 10.244.0.6
```

Two endpoints, both healthy pods. **The broken pod's IP is absent.** That
absence is *why* no traffic reached it — kube-proxy never had a route to it.

## 5. Rollback

```console
$ kubectl rollout undo deployment/readmission-api
deployment.apps/readmission-api rolled back

$ kubectl rollout status deployment/readmission-api
deployment "readmission-api" successfully rolled out

$ kubectl get pods -l app=readmission-api
readmission-api-55d98997bc-42wfl   0/1   Terminating   0   137m
readmission-api-5bb855dd66-589g5   1/1   Running       0   137m
readmission-api-5bb855dd66-jzlln   1/1   Running       0   137m
```

**The two healthy pods are 137 minutes old and were never restarted.** The
rollback did not disturb serving at all; it only removed a pod that had never
received a request.

```console
$ curl -s http://localhost:18080/v1/model
  loaded            : True
  decision_threshold: 0.10106382978723404
```

---

## 6. What this run found

Running this for the first time exposed three real defects. That is the argument
for running things rather than describing them.

**1. The manifests could never have worked.** Both pods came up `0/1` Ready and
stayed there. The image contains no model — `.dockerignore` excludes `models/`
by design — and the `serve` dependency group excludes `mlflow`, so loading from
the registry raises `ImportError` by design too. Every route to a model was
closed, while `configmap.yaml` claimed the API "falls back to the artifact baked
into the image at build time". Nothing was ever baked in.

The fix is a mount, not a bake, and the reason is Phase 7: **promotion is an
alias flip with no deploy.** A model inside the image makes every promotion a
rebuild and redeploy. So `kind-cluster.yaml` maps the host artifact onto the
node and the Deployment mounts it read-only — a `hostPath`, which is the
kind-specific stand-in for the PersistentVolume or init container a real cluster
would use.

**2. `hostPort: 8080` cannot bind on this machine.** kind failed with:

```
bind: An attempt was made to access a socket in a way forbidden by its access permissions
```

which reads like a permissions problem and is not one. On Windows with
Hyper-V/WSL2, WinNAT reserves blocks of ports; 8080 falls inside `8011-8110`
here. Nothing was listening — the OS simply refuses the bind.

```console
$ netsh interface ipv4 show excludedportrange protocol=tcp
      8011        8110
```

Changed to **18080**, with the diagnostic recorded in the manifest so the next
person spends a minute on it rather than an hour.

**3. The demo script referenced a build arg that never existed.** It called
`docker build --build-arg BREAK_MODEL_LOAD=1`, which nothing implements. Replaced
with a bad env var — a more realistic incident anyway, since a config change is
the most common way a working deployment breaks, and `rollout undo` treats
config and image changes identically.

---

---

## 7. Canary rollout — captured run

Run on the same cluster, 2026-09-05. **The canary caught a regression that
latency and error-rate monitoring could not see.**

### Weighting without a service mesh

Both Deployments carry `app: readmission-api`, so the one Service
load-balances across every Ready pod from either. Nine stable replicas beside
one canary is 1-in-10 — the `rollout.canary.traffic_percent: 10` in
`configs/thresholds.yaml`.

The split is quantised by replica count: 10% needs 9:1, and finer weights need
more pods or a mesh. A mesh needs resources this machine does not have. Stated
rather than hidden.

**Attribution comes from the response, not from telemetry.** The canary serves
a different operating point — threshold 0.15 against the champion's 0.1011 —
so every response says which version produced it. That is also what a canary
genuinely *is*: a different model version taking real traffic.

```console
$ kubectl get pods -l app=readmission-api -L track
  9  stable  1/1 Running
  1  canary  1/1 Running
```

### Measured split

```console
  track         n   share   flagged   mean p   p99 ms
  stable      360   90.0%     16.7%   0.0699     40.8
  canary       40   10.0%      0.0%   0.0697     37.7
```

**10.0% measured against 10% configured**, n=400. An earlier 300-request run
gave 8.7%, which is 0.75 standard errors from 10% — consistent, and a reminder
that one run is a sample rather than a proof.

### The verdict

```console
  BREACH EVALUATION (configs/thresholds.yaml -> rollout.canary)
    pass  canary p99 within SLO          37.7 ms <= 700 ms
    pass  canary 5xx ratio               0.00% <= 1%
    FAIL  flagged-rate drift vs stable   16.7% -> 0.0% (-16.7 pts, limit +/-10)

  VERDICT: BREACH - flagged-rate drift vs stable
```

**This is the whole argument for canaries in one screen.** The challenger was
fast and threw no errors — it passed every operational gate. And it flagged
*nobody*: a screening model that identifies zero at-risk patients, while
latency, error rate and health checks all read green.

That is the same failure mode that slipped past CI four separate times in this
project. Here it was caught in 400 requests of real traffic, by the only check
that looks at what the model *does* rather than whether it responds.

Note `mean p` is identical across tracks (0.0699 vs 0.0697) — the scores are the
same, because the weights are the same. The entire difference lands in the
flagged rate, which is precisely the quantity downstream humans feel as
workload. Watching only score distributions would have missed this.

### Rollback

```console
$ kubectl scale deployment/readmission-api-canary --replicas=0
deployment.apps/readmission-api-canary scaled

  track         n   share
  stable      150  100.0%
  canary        0    0.0%
```

Traffic returns to stable immediately. Scaling to zero rather than deleting
keeps the Deployment and its history for the post-incident review.

### What this run found

**An HPA silently caps a replica-weighted canary.** `kubectl scale --replicas=9`
appeared to succeed and then settled at 4, because the HPA's `maxReplicas: 4`
overrode it — turning a configured 10% split into 20% with no error anywhere.
The two mechanisms both control replica count and neither knows about the
other.

The demo removes the HPA first. In a real cluster the canary controller and the
autoscaler have to be reconciled deliberately; a canary whose weight is set by
replica ratio cannot share a Deployment with an autoscaler that has its own
opinion about that ratio.

**Overlapping Deployment selectors would have scaled the canary away.** The
stable Deployment originally selected on `app: readmission-api` alone, which
also matches canary pods. Both Deployments now carry an explicit `track` label
in their selectors. A Deployment's selector is immutable, so applying this to a
running deployment needs `kubectl delete deployment` first.

### Not demonstrated

- **Automatic** rollback. The breach evaluation and the rollback are both
  scripted and both real, but nothing watches continuously and pulls the
  trigger on its own. `auto_rollback_on_breach: true` describes intent; a
  controller would be needed to honour it unattended.
- **The 1000-prediction hold.** `min_predictions_before_promote` is 1000; this
  run used 400, enough to measure the split and catch the regression but short
  of the configured promotion bar.

## What is still not demonstrated

- **Automatic** canary rollback. The canary itself is now demonstrated in
  §7 — split measured, regression caught, rollback executed — but the
  evaluate-and-roll-back loop is scripted rather than continuous.
- **Multi-node behaviour.** Single-node cluster by choice; the resource budget
  does not allow more, and scheduling is not what this demonstrates.
- **The HPA does nothing here.** It is applied and reports `<unknown>/70%`
  because kind does not ship metrics-server. Stated in `hpa.yaml` itself.

## Reproducing

```bash
docker compose down                      # kind and compose together exceed the RAM budget
bash scripts/k8s_rollback_demo.sh
```

Requires `docker`, `kind`, `kubectl`, and a trained artifact at
`models/champion/` (run `uv run mlservice train run` first).
