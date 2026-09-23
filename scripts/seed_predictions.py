"""Drive realistic traffic at a running service.

Dashboards with no data prove nothing, and a screenshot of empty panels is
worse than no screenshot — it invites the reader to assume the panels have
never worked.

The traffic is drawn from the **reference window**, not synthesised, so the
score distribution on the Model Health dashboard is the real one the model
produces rather than a shape invented to look convincing.

A deliberate minority of requests are malformed. That is not padding: the
`validation_errors_total` panel is labelled by field precisely so a spike
concentrated on one field can be told apart from a caller sending junk, and a
panel that has never rendered a non-zero value has never been tested.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from typing import Any


def _post(url: str, payload: dict[str, Any], timeout: float = 10.0) -> tuple[int, float]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, (time.perf_counter() - start) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, (time.perf_counter() - start) * 1000
    except Exception:
        return 0, (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=400, help="single predictions")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--invalid-rate", type=float, default=0.04)
    parser.add_argument("--rps", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    random.seed(args.seed)

    from mlservice.data import features as feature_mod
    from mlservice.monitoring import reports as reports_mod

    frame = reports_mod.load_reference()
    x, _ = feature_mod.split_xy(frame)
    records = [
        {k: (None if str(v) == "nan" else v) for k, v in r.items()} for r in x.to_dict("records")
    ]
    print(f"  {len(records)} reference records available")

    delay = 1.0 / args.rps if args.rps > 0 else 0.0
    codes: dict[int, int] = {}
    by_endpoint: dict[str, dict[int, int]] = {"single": {}, "batch": {}}
    latencies: list[float] = []

    for i in range(args.n):
        record = dict(random.choice(records))

        if random.random() < args.invalid_rate:
            # A field-specific violation, so validation_errors_total{field=...}
            # has a real value to render.
            #
            # It must be a violation on a field that EXISTS. This set
            # `time_in_hospital`, which the credit schema has never had, so
            # `extra="forbid"` rejected it as an unexpected field — still a
            # 422, but labelled with a field name that appears nowhere in the
            # API. The panel exists to tell "one field is being sent wrong"
            # apart from "this caller is sending junk", and an unknown-field
            # label answers neither.
            #
            # fico_range_low is bounded [300, 900], so -5 is out of range and
            # the string is a type error — the two shapes a real caller bug
            # actually takes.
            record["fico_range_low"] = -5 if random.random() < 0.5 else "not-a-number"

        code, ms = _post(f"{args.url}/v1/predict", {"features": record})
        codes[code] = codes.get(code, 0) + 1
        by_endpoint["single"][code] = by_endpoint["single"].get(code, 0) + 1
        latencies.append(ms)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n} sent")
        if delay:
            time.sleep(delay)

    for _ in range(args.batches):
        batch = [dict(random.choice(records)) for _ in range(args.batch_size)]
        # `items` is a flat list of applications. Wrapping each one in
        # {"features": ...} — the shape the SINGLE endpoint takes — made every
        # batch request a 422, so the batch panels were empty in every
        # dashboard capture and the seeder reported success anyway, because it
        # only checks that some 200 was seen.
        code, ms = _post(f"{args.url}/v1/predict/batch", {"items": batch})
        codes[code] = codes.get(code, 0) + 1
        by_endpoint["batch"][code] = by_endpoint["batch"].get(code, 0) + 1
        latencies.append(ms)

    latencies.sort()

    def pct(p: float) -> float:
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    print()
    print("  status codes:", dict(sorted(codes.items())))
    print(f"  p50 {pct(0.50):7.1f} ms")
    print(f"  p95 {pct(0.95):7.1f} ms")
    print(f"  p99 {pct(0.99):7.1f} ms")
    # Every endpoint must have succeeded, not merely one of them. The old
    # check was `any 200 anywhere`, and under it the batch endpoint returned
    # 422 on all 15 requests while this script exited 0 and the operator went
    # on to screenshot empty batch panels.
    failed = [name for name, seen in by_endpoint.items() if seen and not seen.get(200)]
    for name, seen in by_endpoint.items():
        print(f"  {name:7s} {dict(sorted(seen.items()))}")
    if failed:
        print(f"  FAIL  no successful response from: {', '.join(sorted(failed))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
