"""Measure the real canary traffic split through the Service."""

import collections
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:18080/v1/predict"
CANARY_THRESHOLD = 0.15
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

payload = Path("payload.json").read_bytes()
counts: collections.Counter = collections.Counter()
latency: dict[str, list] = {"canary": [], "stable": []}

for i in range(N):
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=20) as response:
        body = json.loads(response.read())
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

print()
share = 100 * counts["canary"] / total if total else 0
print("  configured traffic_percent: 10")
print(f"  measured canary share     : {share:.1f}%  (n={total})")
