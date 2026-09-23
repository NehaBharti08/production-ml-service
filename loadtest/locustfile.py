"""Load test scenarios.

Locust rather than k6 (the plan permits either): it installs as a Python package,
so it needs no admin rights and no separate toolchain, and it ran on this machine
while Docker was still absent.

Four scenarios, escalating. Each answers a different question:

*   ``smoke``  — does the service work at all under any concurrency?
*   ``steady`` — what is the latency profile at the target rate?
*   ``ramp``   — where is the knee, i.e. at what rate does latency depart from flat?
*   ``soak``   — does anything degrade over time (leak, log growth, fd exhaustion)?

Run them with the ``--tags`` flag, e.g.::

    uv run locust -f loadtest/locustfile.py --headless \
        --users 10 --spawn-rate 5 --run-time 60s \
        --host http://127.0.0.1:8000 --tags steady

The realistic mix matters. A load test that only hits ``/v1/predict`` measures a
service nobody runs: probes and scrapes are a large share of real traffic and
they contend for the same event loop.
"""

from __future__ import annotations

import random
from typing import Any

from locust import HttpUser, between, constant, tag, task

#: Kept in sync with the API's own example by importing it, so a schema change
#: cannot leave the load test silently sending payloads the service rejects —
#: which would measure the 422 path and report excellent latency.
try:
    from mlservice.api.schemas import EXAMPLE_FEATURES
except ImportError:  # pragma: no cover - locust may run outside the venv
    EXAMPLE_FEATURES = {
        "loan_amnt": 10000.0,
        "term": " 36 months",
        "int_rate": 11.99,
        "installment": 332.1,
        "grade": "B",
        "sub_grade": "B3",
        "purpose": "debt_consolidation",
        "initial_list_status": "w",
        "application_type": "Individual",
        "annual_inc": 65000.0,
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "addr_state": "CA",
        "dti": 18.4,
        "fico_range_low": 700.0,
        "credit_history_months": 180.0,
        "open_acc": 11.0,
        "total_acc": 24.0,
        "revol_bal": 12500.0,
        "revol_util": 43.2,
        "delinq_2yrs": 0.0,
        "inq_last_6mths": 1.0,
        "pub_rec": 0.0,
        "pub_rec_bankruptcies": 0.0,
    }

# The grades the score actually turns on, and a few states for the geographic
# spread. Used only to vary the payload; see _varied_features.
GRADES = ["A", "B", "C", "D", "E"]
STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"]


def _varied_features() -> dict[str, Any]:
    """Vary the payload per request.

    Identical payloads would let any caching — in pandas, in the encoder, or
    added later — flatter the numbers. Varying the fields that actually drive the
    score keeps the measurement honest.

    These were readmission fields (`time_in_hospital`, `number_inpatient`,
    `age`) long after the domain changed. Because they were spread on top of
    EXAMPLE_FEATURES, they survived even once that import returned credit
    fields — and the request schema sets ``extra="forbid"``, so every load-test
    request would have been a 422. The run would have measured the validation
    path and reported it as prediction latency.
    """
    grade = random.choice(GRADES)  # load shaping, not crypto
    return {
        **EXAMPLE_FEATURES,
        "grade": grade,
        "sub_grade": f"{grade}{random.randint(1, 5)}",
        "addr_state": random.choice(STATES),
        "loan_amnt": float(random.randrange(1_000, 35_000, 500)),
        "int_rate": round(random.uniform(5.5, 28.0), 2),
        "dti": round(random.uniform(0.0, 40.0), 2),
        "fico_range_low": float(random.randrange(660, 845, 5)),
        "annual_inc": float(random.randrange(25_000, 200_000, 1_000)),
    }


class SteadyUser(HttpUser):
    """The realistic mix: mostly single predictions, with probes and scrapes."""

    weight = 10
    wait_time = between(0.05, 0.2)

    @tag("smoke", "steady", "ramp", "soak")
    @task(20)
    def predict(self) -> None:
        with self.client.post(
            "/v1/predict",
            json={"features": _varied_features(), "client_id": "loadtest"},
            name="POST /v1/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
            elif "prediction_id" not in response.text:
                # A 200 without a prediction_id would mean the response shape
                # changed; a status-only check would call that a success.
                response.failure("200 without prediction_id")

    @tag("smoke", "steady", "ramp", "soak")
    @task(2)
    def readiness(self) -> None:
        self.client.get("/health/ready", name="GET /health/ready")

    @tag("smoke", "steady", "ramp", "soak")
    @task(1)
    def scrape_metrics(self) -> None:
        """Prometheus scrapes every 15s in production; it contends for the loop."""
        self.client.get("/metrics", name="GET /metrics")


class BatchUser(HttpUser):
    """Batch callers, which are where the throughput actually is.

    Measured in Phase 3: per-item cost falls from ~86 ms at batch 1 to ~0.24 ms
    at batch 200, because the sklearn transform overhead is per *call*.
    """

    weight = 1
    wait_time = between(0.5, 1.5)

    @tag("smoke", "steady", "ramp", "soak")
    @task
    def predict_batch(self) -> None:
        size = random.choice([10, 25, 50])
        with self.client.post(
            "/v1/predict/batch",
            json={"items": [_varied_features() for _ in range(size)], "client_id": "loadtest"},
            name=f"POST /v1/predict/batch (n={size})",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")


class ValidationErrorUser(HttpUser):
    """A small share of malformed requests, as any real caller produces.

    Included because the 422 path runs the error handler and metric recording,
    and a service that is fast until someone sends bad input is not fast.
    """

    weight = 1
    wait_time = constant(1)

    @tag("smoke", "steady", "soak")
    @task
    def invalid_payload(self) -> None:
        with self.client.post(
            "/v1/predict",
            json={"features": {**EXAMPLE_FEATURES, "age": "75"}},  # bare number
            name="POST /v1/predict (invalid)",
            catch_response=True,
        ) as response:
            # A 422 is the CORRECT outcome here. Reporting it as a failure would
            # make the error rate meaningless.
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"expected 422, got {response.status_code}")
