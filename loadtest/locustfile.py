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
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 5,
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "diag_1": "Circulatory",
        "diag_2": "Diabetes",
        "diag_3": "Circulatory",
        "number_diagnoses": 9,
        "max_glu_serum": "NotMeasured",
        "A1Cresult": "NotMeasured",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes",
    }

AGE_BANDS = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]


def _varied_features() -> dict[str, Any]:
    """Vary the payload per request.

    Identical payloads would let any caching — in pandas, in the encoder, or
    added later — flatter the numbers. Varying the fields that actually drive the
    score keeps the measurement honest.
    """
    return {
        **EXAMPLE_FEATURES,
        "age": random.choice(AGE_BANDS),  # load shaping, not crypto
        "number_inpatient": random.randint(0, 8),
        "number_emergency": random.randint(0, 4),
        "time_in_hospital": random.randint(1, 14),
        "num_medications": random.randint(5, 40),
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
