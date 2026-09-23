"""Measure the real canary traffic split through the Service."""

import collections
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

URL = "http://127.0.0.1:18080/v1/predict"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def _threshold(track: str) -> float:
    """The operating point each track serves, from the artifact the node mounts.

    Was a hardcoded 0.15 beside a champion that moved to 0.2070 in the domain
    change. A stale constant here does not fail — it silently attributes every
    response to "stable" and reports a canary carrying 0% of traffic.
    """
    meta = Path("models") / track / "metadata.json"
    if not meta.is_file():
        sys.exit(f"  no {meta} — the {track} artifact must exist to attribute responses")
    return round(json.loads(meta.read_text(encoding="utf-8"))["champion_threshold"], 4)


STABLE_THRESHOLD = _threshold("champion")
CANARY_THRESHOLD = _threshold("canary")
if STABLE_THRESHOLD == CANARY_THRESHOLD:
    sys.exit(f"  both tracks serve {CANARY_THRESHOLD} — the split is unmeasurable")

payload = Path("payload.json").read_bytes()
counts: collections.Counter = collections.Counter()
errors: collections.Counter = collections.Counter()
latency: dict[str, list] = {"canary": [], "stable": []}

for i in range(N):
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read())
    except Exception as exc:
        # One slow response must not destroy the measurement. Ten API pods on a
        # single kind node contend enough to time out occasionally, and an
        # unhandled TimeoutError here threw away 400 already-collected samples.
        # Errors are counted and reported, because a split measured against a
        # struggling cluster is a different claim from one measured cleanly.
        errors[type(exc).__name__] += 1
        continue
    elapsed = (time.perf_counter() - start) * 1000

    threshold = round(body["decision_threshold"], 4)
    who = "canary" if abs(threshold - CANARY_THRESHOLD) < 1e-9 else "stable"
    counts[who] += 1
    latency[who].append(elapsed)

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{N} sent", flush=True)

total = sum(counts.values())
print()
print(f"  {'track':8s} {'requests':>9s} {'share':>8s}   {'p50 ms':>8s} {'p99 ms':>8s}")
for who in ("stable", "canary"):
    n = counts[who]
    lat = sorted(latency[who])
    if not lat:
        print(f"  {who:8s} {n:9d} {0.0:7.1f}%          -        -")
        continue
    p50 = lat[len(lat) // 2]
    p99 = lat[min(int(len(lat) * 0.99), len(lat) - 1)]
    print(f"  {who:8s} {n:9d} {100 * n / total:7.1f}%   {p50:8.1f} {p99:8.1f}")

if errors:
    print()
    print(f"  errors: {dict(errors)} of {N} attempted")

print()
share = 100 * counts["canary"] / total if total else 0

config = yaml.safe_load(Path("configs/thresholds.yaml").read_text(encoding="utf-8"))
configured = config["rollout"]["canary"]["traffic_percent"]


def _ready(deployment: str) -> int:
    """Ready replicas, which are what the Service actually balances across."""
    out = subprocess.run(
        ["kubectl", "get", "deploy", deployment, "-o", "jsonpath={.status.readyReplicas}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return int(out.stdout.strip() or 0)


stable_n = _ready("credit-risk-api")
canary_n = _ready("credit-risk-api-canary")
achievable = 100 * canary_n / (stable_n + canary_n) if (stable_n + canary_n) else 0

print(f"  configured traffic_percent : {configured}%")
print(f"  achievable from replicas   : {achievable:.1f}%  ({canary_n} canary : {stable_n} stable)")
print(f"  measured canary share      : {share:.1f}%  (n={total})")
print()

failed = False

if canary_n == 0:
    failed = True
    print("    FAIL  no canary pods are running — there is no split to measure.")
    print("          (Expected if the canary was just rolled back.)")
elif abs(achievable - configured) > 1.0:
    failed = True
    print(f"    FAIL  the cluster cannot deliver {configured}%. Replica-ratio")
    print("          weighting quantises the split by pod count, and something")
    print("          is changing the stable replica count underneath it.")
    print("          Check the HPA: maxReplicas bounds the denominator, and it")
    print("          enforces those bounds even when it cannot read metrics at")
    print("          all -- ScalingActive: False still clamps to maxReplicas.")
else:
    print(f"    pass  replica ratio delivers the configured {configured}%")

if abs(share - achievable) > 5.0:
    failed = True
    print(f"    FAIL  measured {share:.1f}% is over 5 points from the {achievable:.1f}%")
    print("          the replica ratio implies -- the Service is not balancing")
    print("          as assumed")
else:
    print("    pass  measured share is within 5 points of the replica ratio")

sys.exit(1 if failed else 0)
