"""Evaluate a running canary against its configured breach conditions.

Answers the question a canary exists to answer: *is the challenger behaving
differently, in production, on real traffic?* Attribution comes from the
response itself — the canary serves a different operating point, so every
response says which version produced it. No mesh telemetry required.
"""

import collections
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:18080/v1/predict"
CANARY_THRESHOLD = 0.15
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

payload = Path("payload_varied.json").read_bytes()
records = json.loads(payload)

counts: collections.Counter = collections.Counter()
flagged: collections.Counter = collections.Counter()
latency: dict[str, list] = {"canary": [], "stable": []}
proba: dict[str, list] = {"canary": [], "stable": []}
errors: collections.Counter = collections.Counter()

for i in range(N):
    body = json.dumps({"features": records[i % len(records)]}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read())
    except Exception:
        errors["error"] += 1
        continue
    elapsed = (time.perf_counter() - start) * 1000

    threshold = round(result["decision_threshold"], 4)
    who = "canary" if abs(threshold - CANARY_THRESHOLD) < 1e-9 else "stable"
    counts[who] += 1
    latency[who].append(elapsed)
    proba[who].append(result["readmission_probability"])
    if result["flagged"]:
        flagged[who] += 1

    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{N} sent", flush=True)

total = sum(counts.values())
print()
print(f"  {'track':8s} {'n':>6s} {'share':>7s} {'flagged':>9s} {'mean p':>8s} {'p99 ms':>8s}")
stats = {}
for who in ("stable", "canary"):
    n = counts[who]
    if not n:
        continue
    lat = sorted(latency[who])
    p99 = lat[min(int(len(lat) * 0.99), len(lat) - 1)]
    mean_p = sum(proba[who]) / len(proba[who])
    rate = flagged[who] / n
    stats[who] = {"n": n, "p99": p99, "flagged_rate": rate, "mean_proba": mean_p}
    print(f"  {who:8s} {n:6d} {100 * n / total:6.1f}% {100 * rate:8.1f}% {mean_p:8.4f} {p99:8.1f}")

print()
print("  BREACH EVALUATION (configs/thresholds.yaml -> rollout.canary)")

SLO_P99_MS = 700.0
PAGE_5XX = 0.01

verdicts = []
if "canary" in stats:
    c = stats["canary"]
    s = stats.get("stable", c)

    ok_latency = c["p99"] <= SLO_P99_MS
    verdicts.append(
        ("canary p99 within SLO", ok_latency, f"{c['p99']:.1f} ms <= {SLO_P99_MS:.0f} ms")
    )

    err_ratio = errors["error"] / max(total + errors["error"], 1)
    ok_errors = err_ratio <= PAGE_5XX
    verdicts.append(
        ("canary 5xx ratio", ok_errors, f"{100 * err_ratio:.2f}% <= {100 * PAGE_5XX:.0f}%")
    )

    # The behavioural signal. Same weights, different operating point, so the
    # scores match and the FLAGGED RATE is where the difference lands - which
    # is exactly the quantity downstream humans feel as workload.
    delta = c["flagged_rate"] - s["flagged_rate"]
    ok_flag = abs(delta) <= 0.10
    verdicts.append(
        (
            "flagged-rate drift vs stable",
            ok_flag,
            f"{100 * s['flagged_rate']:.1f}% -> {100 * c['flagged_rate']:.1f}% "
            f"({100 * delta:+.1f} pts, limit +/-10)",
        )
    )

for name, ok, detail in verdicts:
    print(f"    {'pass' if ok else 'FAIL'}  {name:30s} {detail}")

breached = [n for n, ok, _ in verdicts if not ok]
print()
if breached:
    print(f"  VERDICT: BREACH - {', '.join(breached)}")
    print("  auto_rollback_on_breach is true -> roll the canary back")
    sys.exit(2)
print("  VERDICT: canary healthy on every configured condition")
