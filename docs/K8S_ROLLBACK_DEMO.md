# Kubernetes rollout and rollback — captured run

> **NOT A CREDIT DECISIONING SYSTEM.** This document describes an engineering
> demonstration of ML operations. Nothing here has been validated for lending,
> and none of it may decide anyone's access to credit.

Captured output from a real run, not a description of what would happen.

- **When:** 2026-09-23
- **Cluster:** kind v0.33.0, Kubernetes v1.37.0, single control-plane node
- **Serving:** `credit-default-risk`, threshold `0.20700843608046723`, feature
  schema `f5566c96d5eed4de`
- **Script:** [`scripts/k8s_rollback_demo.sh`](../scripts/k8s_rollback_demo.sh)

> **Captured on the lbfgs champion.** On 2026-09-25 the solver changed to newton-cholesky, because lbfgs stopped at a point that depended on row order and platform; the current threshold is 0.2049. The output below is left exactly as it ran. Nothing it demonstrates depends on the value: probes, endpoints and rollback behave the same, and the canary still serves a lower operating point (0.15) than the champion.

The headline result:

> **12/12 health probes returned 200 during the failed rollout**, and the
> Service endpoint list never contained the broken pod.

The point is not that `kubectl rollout undo` exists. It is that a bad
deployment is *contained* rather than propagated, because readiness and
liveness answer different questions and are wired to endpoints that answer
them.

---

## 1. Baseline

```console
$ kubectl rollout status deployment/credit-risk-api
deployment "credit-risk-api" successfully rolled out

$ curl -fsS http://localhost:18080/v1/model
{"loaded":true,"name":"credit-default-risk",
 "version":"local:1789985993277297900","stage":"local_fallback",
 "source":"local_fallback","feature_schema_hash":"f5566c96d5eed4de",
 "decision_threshold":0.20700843608046723,"loaded_seconds_ago":2.3,
 "api_version":"v1","disclaimer":"NOT A CREDIT DECISIONING SYSTEM. ..."}
```

The threshold served is the **trained** `0.20700843608046723`, not the `0.5`
config placeholder. That distinction has its own history in this repository —
the placeholder reached production six separate times in different disguises,
which is why the artifact now carries its own `metadata.json` and why the
bootstrap workflow compares it against the training run that produced it.

---

## 2. Deploy a broken change

Broken where it matters. The process starts, liveness passes, and the model
never loads — so readiness stays 503 forever.

```console
$ kubectl set env deployment/credit-risk-api \
    MLSERVICE_MODEL__LOCAL_FALLBACK=/app/models/champion/does-not-exist.joblib
deployment.apps/credit-risk-api env updated
```

A container that crashes outright is caught by anything. One that runs happily
while unable to do its job is the case that needs a readiness probe wired to
something real.

A config change rather than a second image, because a config change is the most
common way a working deployment breaks, and `rollout undo` treats the two
identically.

---

## 3. The rollout stalls — and the stall is the containment

```console
$ kubectl rollout status deployment/credit-risk-api --timeout=90s
Waiting for deployment "credit-risk-api" rollout to finish: 1 out of 2 new replicas have been updated...
error: timed out waiting for the condition

$ kubectl get pods -l app=credit-risk-api -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP
credit-risk-api-556d8c5bf6-j8bwn   1/1     Running   0          98s   10.244.0.6
credit-risk-api-556d8c5bf6-kqksm   1/1     Running   0          98s   10.244.0.5
credit-risk-api-c5bfccbdf-dx4gp    0/1     Running   0          90s   10.244.0.7
```

`maxUnavailable: 0` means the old pods are not removed until the new one is
Ready. It never becomes Ready, so the command times out. **That timeout is the
system working**, not failing.

Note `RESTARTS 0` on the broken pod. Liveness is wired to `/health/live`, which
answers "is the process alive" — and it is. Had liveness been pointed at the
readiness check, this pod would be crash-looping, and a blocked rollout would
look like an outage.

---

## 4. Traffic during the failure

Counted across the failure window, not sampled once. "It answered when I
checked" is an anecdote.

```console
$ for _ in $(seq 1 12); do curl -s -o /dev/null -w '%{http_code}' \
    http://localhost:18080/health/ready; sleep 1; done
    12/12 probes returned 200 DURING the failed rollout
```

The mechanism, not just the symptom — the broken pod is **absent from the
Service endpoints**, which is *why* no traffic reached it:

```console
$ kubectl get endpointslices -l kubernetes.io/service-name=credit-risk-api
  10.244.0.5 ready=true
  10.244.0.6 ready=true
  10.244.0.7 ready=false

$ kubectl get pods -l app=credit-risk-api \
    -o custom-columns=NAME:.metadata.name,READY:.status.containerStatuses[0].ready
NAME                               READY
credit-risk-api-556d8c5bf6-j8bwn   true
credit-risk-api-556d8c5bf6-kqksm   true
credit-risk-api-c5bfccbdf-dx4gp    false
```

---

## 5. Rollback

```console
$ kubectl rollout undo deployment/credit-risk-api
deployment.apps/credit-risk-api rolled back

$ kubectl rollout status deployment/credit-risk-api
deployment "credit-risk-api" successfully rolled out
    image is back to mlservice-api:local
    GET /health/ready -> 200

$ kubectl rollout history deployment/credit-risk-api
REVISION  CHANGE-CAUSE
2         <none>
3         <none>
```

`CHANGE-CAUSE` is empty because neither change was annotated. Left as it ran
rather than tidied: an un-annotated rollout history is what most clusters
actually look like, and pretending otherwise would misrepresent the run.

---

## 6. What this run found

**A broken model reached the cluster and served zero requests.** Not because a
human noticed, but because readiness is a traffic gate and liveness is a
restart trigger, and each was wired to the endpoint that answers its own
question.

The model-level equivalent needs no cluster at all:

```bash
uv run mlservice retrain verify-rollback
```

Two levers, two layers: the registry alias flip for a bad model,
`rollout undo` for a bad image or config.

---

## 7. Canary rollout — captured run

### Weighting without a service mesh

Both Deployments carry `app: credit-risk-api`, so the one Service
load-balances across every Ready pod from either. Nine stable replicas beside
one canary gives 1-in-10.

The canary serves a **different operating point** — threshold `0.15` against
the champion's `0.2070` — and every response carries its
`decision_threshold`. That is what makes the split measurable from the
responses alone, with no mesh telemetry.

`0.15` is *below* the champion's threshold, so the canary flags **more**
applications. In credit that is a real business change: a lower bar rejects
more borrowers.

### The HPA silently broke the split, and that is the finding

The first measurement came back at **23.0%** against a configured 10%.

```console
  configured traffic_percent : 10%
  achievable from replicas   : 20.0%  (1 canary : 4 stable)
  measured canary share      : 18.2%  (n=400)

    FAIL  the cluster cannot deliver 10%.
```

`kubectl scale --replicas=9` had been applied and had taken effect. Seven
seconds later:

```console
$ kubectl get events --field-selector involvedObject.name=credit-risk-api
70s   Normal   ScalingReplicaSet   Scaled up replica set ... from 2 to 9
63s   Normal   SuccessfulRescale   New size: 4; reason: Current number of replicas above Spec.MaxReplicas
63s   Normal   ScalingReplicaSet   Scaled down replica set ... from 9 to 4
```

**The HPA's `maxReplicas: 4` clamped it back**, and 1/(4+1) is 20%, which is
what was measured.

Two things worth keeping:

1. **Replica-ratio canary weighting is incompatible with an HPA on the stable
   deployment.** The autoscaler owns the denominator of your traffic split. A
   canary you believe is taking 10% can quietly be taking 20% — or, under load,
   something different again from minute to minute.
2. **An HPA that cannot read metrics is not inert.** This one reported
   `ScalingActive: False` with `FailedGetResourceMetric` — kind ships no
   metrics-server — and *still enforced its bounds*. A broken autoscaler is
   easy to assume is a no-op. It is not.

The demo now pins the HPA bounds for the canary window, which is what a real
team does: you do not want an autoscaler reshaping a traffic split mid-
experiment.

`scripts/canary_split.py` asks the two questions separately — *can the cluster
deliver the configured split* and *is the Service balancing as the replica
ratio implies* — because they have different causes and different fixes.
Printing "configured 10, measured 23" and exiting 0, which is what it used to
do, answers neither.

### Measured split, after pinning the HPA

```console
  track     requests    share     p50 ms   p99 ms
  stable         274    91.3%       52.2     84.3
  canary          26     8.7%       51.6     71.4

  configured traffic_percent : 10%
  achievable from replicas   : 10.0%  (1 canary : 9 stable)
  measured canary share      :  8.7%  (n=300)

    pass  replica ratio delivers the configured 10%
    pass  measured share is within 5 points of the replica ratio
```

8.7% against 10.0% at n=300. The split is a per-connection random draw, so an
exact match would be the surprising result.

### The verdict

```console
$ uv run python scripts/canary_evaluate.py 400

  stable 0.207  canary 0.15

  track         n   share   flagged   mean p   p99 ms
  stable      364   91.0%     23.9%   0.1427     83.2
  canary       36    9.0%     50.0%   0.1432    146.5

  BREACH EVALUATION (configs/thresholds.yaml -> rollout.canary)
    pass  canary p99 within SLO          146.5 ms <= 700 ms
    pass  canary 5xx ratio               0.00% <= 1%
    FAIL  flagged-rate drift vs stable   23.9% -> 50.0% (+26.1 pts, limit +/-10)

  VERDICT: BREACH - flagged-rate drift vs stable
  auto_rollback_on_breach is true -> roll the canary back
```

**Read the two middle columns together.** Mean predicted probability is
`0.1427` on stable and `0.1432` on the canary — identical to three decimal
places, because the weights *are* identical. Only the operating point differs.

And the flagged rate **doubles**, from 23.9% to 50.0%.

That is the entire argument for watching the flagged rate rather than the score
distribution. A monitor watching mean probability would have seen nothing at
all. The quantity that changed is the one a downstream human feels as workload
— here, twice as many applications pushed into manual review, or twice as many
borrowers declined.

### Rollback

```console
$ kubectl delete -f deploy/k8s/canary.yaml
deployment.apps "credit-risk-api-canary" deleted

$ uv run python scripts/canary_split.py 120
  achievable from replicas   : 0.0%  (0 canary : 9 stable)
  measured canary share      : 0.0%  (n=120)
```

Verified by measurement rather than asserted: after the rollback the canary
takes no traffic.

### Not demonstrated by the canary run

- **Automatic rollback.** `auto_rollback_on_breach` is `true` in config and the
  evaluator exits non-zero, but nothing is wired to act on that exit code. The
  rollback above was run by hand. A controller that closes the loop is not
  built, and claiming it would be a lie.
- **Gradual traffic ramp.** 10% then 0%. No 25%/50%/100% progression.
- **Finer weights than the replica count allows.** 10% needs 9:1. Anything
  finer needs more pods or a service mesh, and a mesh needs resources this
  machine does not have.

---

## What is still not demonstrated

- **Multi-node scheduling.** One control-plane node, deliberately: the demo is
  about probes and rollout history, not scheduling.
- **HPA scaling on real load.** kind ships no metrics-server, so the HPA never
  computes a metric. What this run *did* show is that it enforces its bounds
  anyway — see §7.
- **A PersistentVolume or registry-pull for the artifact.** kind mounts
  `./models` from the host. A real cluster would use a PV or an initContainer,
  the same shape as the Space downloading at boot. `hostPath` is the
  kind-specific equivalent, not a pattern to copy.
- **Aggregating prediction logs across replicas.** Per-instance NDJSON on an
  `emptyDir`. Ephemeral by design, and unsolved.

---

## Reproducing

```bash
bash scripts/k8s_rollback_demo.sh        # §1-6, end to end

# §7, after the above leaves a cluster running:
kubectl patch hpa credit-risk-api --type=merge \
  -p '{"spec":{"minReplicas":9,"maxReplicas":9}}'   # or it clamps you back
kubectl scale deployment/credit-risk-api --replicas=9
kubectl apply -f deploy/k8s/canary.yaml
uv run python scripts/canary_split.py 300
uv run python scripts/canary_evaluate.py 400
kubectl delete -f deploy/k8s/canary.yaml
```

`docker compose down` first — the resource budget treats the compose stack and
a kind cluster as mutually exclusive on this machine.
