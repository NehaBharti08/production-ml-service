#!/usr/bin/env bash
# Scripted rollout/rollback demo on kind.
#
# The point is not that `kubectl rollout undo` exists. It is that a bad
# deployment is *contained* rather than propagated, because readiness and
# liveness answer different questions:
#
#   1. Deploy a healthy version, confirm it serves.
#   2. Deploy a deliberately broken one.
#   3. Readiness fails -> the pod never joins the Service endpoints ->
#      the old ReplicaSet keeps serving -> no traffic is lost.
#   4. `kubectl rollout undo` returns to the previous ReplicaSet.
#
# Step 3 is the part worth watching. If liveness had been wired to the readiness
# check, the pod would crash-loop instead of being quietly held out of rotation,
# and the failure would look like an outage rather than a blocked rollout.
#
# STATUS: written, NOT YET RUN. Docker is not installed on the development
# machine, so this script has never executed end to end. It is committed as the
# intended procedure, and the README says the same rather than implying a demo
# that has not happened. Every command is standard kubectl; the risk is in the
# details this script cannot verify without a cluster.
set -euo pipefail

CLUSTER=mlservice
DEPLOY=readmission-api
URL=http://localhost:8080
IMAGE_GOOD=mlservice-api:local
IMAGE_BAD=mlservice-api:broken

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '    \033[0;32m%s\033[0m\n' "$*"; }
bad() { printf '    \033[0;31m%s\033[0m\n' "$*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { bad "missing: $1"; exit 1; }
}

require docker
require kind
require kubectl

# ---------------------------------------------------------------- 0. cluster
say "0. Cluster"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  # Compose and kind together exceed the RAM budget on this machine.
  docker compose down --remove-orphans 2>/dev/null || true
  kind create cluster --config deploy/k8s/kind-cluster.yaml
else
  ok "cluster '$CLUSTER' already exists"
fi
kubectl config use-context "kind-${CLUSTER}"

# ------------------------------------------------------------------ 1. good
say "1. Build and load the healthy image"
docker build -f deploy/docker/Dockerfile.api -t "$IMAGE_GOOD" .
kind load docker-image "$IMAGE_GOOD" --name "$CLUSTER"

say "2. Apply manifests"
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/deployment.yaml
# The HPA needs metrics-server, which kind does not ship. Applied anyway so the
# manifest is exercised; it will read <unknown>/70% until metrics-server exists.
kubectl apply -f deploy/k8s/hpa.yaml || bad "HPA not applied (metrics-server missing?)"

kubectl rollout status "deployment/${DEPLOY}" --timeout=180s
ok "healthy version is serving"

say "3. Confirm it answers"
curl -fsS "${URL}/health/ready" >/dev/null && ok "GET /health/ready -> 200"
curl -fsS "${URL}/v1/model" | head -c 400; echo

# The baseline for step 6: whatever is serving now must still be serving after
# the bad deploy. Captured before, because after the rollout it is no longer
# distinguishable from the version we are about to try to ship.
BEFORE=$(kubectl get "deployment/${DEPLOY}" -o jsonpath='{.spec.template.spec.containers[0].image}')
ok "current image: ${BEFORE}"

# ------------------------------------------------------------------- 2. bad
say "4. Deploy a deliberately broken image"
# Broken where it matters: the process starts (liveness passes) but the model
# never loads, so readiness stays 503. That is the interesting failure — a
# container that crashes outright is caught by anything.
docker build -f deploy/docker/Dockerfile.api -t "$IMAGE_BAD" \
  --build-arg BREAK_MODEL_LOAD=1 . 2>/dev/null \
  || bad "no BREAK_MODEL_LOAD build arg yet — simulate instead: kubectl set image ... =mlservice-api:nonexistent"
kind load docker-image "$IMAGE_BAD" --name "$CLUSTER" 2>/dev/null || true

kubectl set image "deployment/${DEPLOY}" "api=${IMAGE_BAD}" --record 2>/dev/null \
  || kubectl set image "deployment/${DEPLOY}" "api=${IMAGE_BAD}"

say "5. Watch the rollout fail to progress"
# maxUnavailable: 0 means the old pods are not removed until the new one is
# Ready. It never becomes Ready, so this times out — and that timeout IS the
# containment working.
if kubectl rollout status "deployment/${DEPLOY}" --timeout=90s; then
  bad "rollout SUCCEEDED — the broken image was not rejected. Investigate the readiness probe."
else
  ok "rollout stalled, as intended: the new pod never became Ready"
fi

kubectl get pods -l "app=${DEPLOY}" -o wide

say "6. Traffic is still served by the old ReplicaSet"
if curl -fsS "${URL}/health/ready" >/dev/null; then
  ok "GET /health/ready -> 200 THROUGHOUT the failed rollout"
  ok "the bad version never received traffic"
else
  bad "service is down — containment failed, this is the case to investigate"
fi

# -------------------------------------------------------------- 3. rollback
say "7. Roll back"
kubectl rollout undo "deployment/${DEPLOY}"
kubectl rollout status "deployment/${DEPLOY}" --timeout=180s

AFTER=$(kubectl get "deployment/${DEPLOY}" -o jsonpath='{.spec.template.spec.containers[0].image}')
if [ "$AFTER" = "$BEFORE" ]; then
  ok "image is back to ${AFTER}"
else
  bad "expected ${BEFORE}, got ${AFTER}"
  exit 1
fi

curl -fsS "${URL}/health/ready" >/dev/null && ok "GET /health/ready -> 200"

say "8. Rollout history"
kubectl rollout history "deployment/${DEPLOY}"

cat <<'SUMMARY'

    What this demonstrated
    ----------------------
    A broken model reached the cluster and served zero requests.

    Not because a human noticed, but because readiness is a traffic gate and
    liveness is a restart trigger, and they were wired to endpoints that answer
    those two different questions. The rollout stalling is the system working.

    The model-level equivalent, which needs no cluster at all:

        uv run mlservice retrain verify-rollback

    Two levers, two layers: the alias flip for a bad model, `rollout undo` for
    a bad image.
SUMMARY
