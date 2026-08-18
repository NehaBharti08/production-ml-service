"""Integration tests against a *running* service.

Distinct from the contract suite, which exercises the app in-process with
TestClient. These go over real HTTP to a real process, which is the only way to
observe things TestClient cannot reproduce:

*   middleware ordering as ASGI actually applies it
*   response headers as they leave the server
*   the prediction log written by a separate process
*   metrics accumulating across requests in one long-lived process
*   startup and readiness sequencing

Point them at a service with ``MLSERVICE_TEST_BASE_URL``::

    uv run uvicorn mlservice.api.main:app --port 8000 &
    MLSERVICE_TEST_BASE_URL=http://127.0.0.1:8000 uv run pytest tests/integration -m integration

They **skip** rather than fail when no service is reachable, so a fresh clone and
the default CI path are not blocked. That is a deliberate trade: a skipped test
is visible in the report (`-rs`), whereas a failing one that everybody learns to
ignore is worse than none.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASE_URL = os.environ.get("MLSERVICE_TEST_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="module")
def http() -> Any:
    """An httpx client against a reachable service, or skip."""
    import httpx

    try:
        client = httpx.Client(base_url=BASE_URL, timeout=30.0)
        response = client.get("/health/live")
        response.raise_for_status()
    except Exception:  # pragma: no cover - environment dependent
        pytest.skip(
            f"no service reachable at {BASE_URL} — start one, or set MLSERVICE_TEST_BASE_URL"
        )
    yield client
    client.close()


@pytest.fixture(scope="module")
def features() -> dict[str, Any]:
    from mlservice.api.schemas import EXAMPLE_FEATURES

    return dict(EXAMPLE_FEATURES)


def _require_ready(http: Any) -> None:
    if http.get("/health/ready").status_code != 200:
        pytest.skip("service is running but not ready (no model loaded)")


class TestServiceIsUp:
    def test_liveness_responds(self, http: Any) -> None:
        assert http.get("/health/live").json()["status"] == "alive"

    def test_readiness_reports_a_definite_state(self, http: Any) -> None:
        """Either ready with a model, or 503 with a reason. Never ambiguous."""
        response = http.get("/health/ready")
        assert response.status_code in (200, 503)
        body = response.json()
        if response.status_code == 503:
            assert body["checks"].get("error")
        else:
            assert body["checks"]["model_loaded"] is True


class TestRealHttpBehaviour:
    def test_prediction_round_trip(self, http: Any, features: dict[str, Any]) -> None:
        _require_ready(http)
        response = http.post("/v1/predict", json={"features": features})
        assert response.status_code == 200

        body = response.json()
        assert 0.0 <= body["readmission_probability"] <= 1.0
        assert body["flagged"] == (body["readmission_probability"] >= body["decision_threshold"])

    def test_threshold_is_not_the_placeholder(self, http: Any) -> None:
        """Regression: the API once served config's 0.5 instead of the trained
        0.1011, so a patient above the model's own operating point came back
        unflagged. In-process tests used a stub threshold and could not have
        caught it."""
        _require_ready(http)
        body = http.get("/v1/model").json()
        assert body["decision_threshold"] != 0.5, (
            "serving the 0.5 placeholder — the threshold is not being read from the trained model"
        )

    def test_response_headers_survive_the_real_server(
        self, http: Any, features: dict[str, Any]
    ) -> None:
        """TestClient can mask middleware/ASGI ordering issues."""
        _require_ready(http)
        response = http.post(
            "/v1/predict",
            json={"features": features},
            headers={"X-Request-ID": "integration-trace"},
        )
        assert response.headers["X-Request-ID"] == "integration-trace"
        assert float(response.headers["X-Response-Time-Ms"]) >= 0

    def test_validation_error_shape_over_the_wire(self, http: Any) -> None:
        response = http.post("/v1/predict", json={"features": {"age": "75"}})
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.json()["errors"]


class TestMetricsAccumulate:
    def test_counters_increase_across_requests(self, http: Any, features: dict[str, Any]) -> None:
        """Only observable in a long-lived process; TestClient starts fresh."""
        _require_ready(http)

        def predictions_total() -> float:
            total = 0.0
            for line in http.get("/metrics").text.splitlines():
                if line.startswith("model_predictions_total{"):
                    total += float(line.rsplit(" ", 1)[1])
            return total

        before = predictions_total()
        for _ in range(3):
            http.post("/v1/predict", json={"features": features})
        assert predictions_total() >= before + 3

    def test_endpoint_labels_name_real_routes(self, http: Any, features: dict[str, Any]) -> None:
        """Regression: every metric was once labelled endpoint="unmatched"."""
        _require_ready(http)
        http.post("/v1/predict", json={"features": features})
        assert 'endpoint="/v1/predict"' in http.get("/metrics").text


class TestOutcomeLifecycle:
    def test_predict_then_record_outcome(self, http: Any, features: dict[str, Any]) -> None:
        """The full delayed-label path, across process boundaries."""
        _require_ready(http)
        prediction_id = http.post("/v1/predict", json={"features": features}).json()[
            "prediction_id"
        ]

        response = http.post(
            "/v1/outcomes",
            json={
                "prediction_id": prediction_id,
                "readmitted_within_30_days": True,
                "source": "integration-test",
            },
        )
        assert response.status_code == 200
        assert response.json()["recorded"] is True


class TestBatchOverHttp:
    def test_batch_returns_one_result_per_item(self, http: Any, features: dict[str, Any]) -> None:
        _require_ready(http)
        response = http.post("/v1/predict/batch", json={"items": [features] * 5})
        assert response.status_code == 200
        assert response.json()["count"] == 5

    def test_batch_is_cheaper_per_item_than_singles(
        self, http: Any, features: dict[str, Any]
    ) -> None:
        """Asserts the property the batch endpoint exists for.

        The sklearn transform cost is per *call*, so per-item latency must fall
        substantially with batch size. A regression that made batching
        row-by-row would leave every other test passing.
        """
        _require_ready(http)
        single = http.post("/v1/predict", json={"features": features}).json()["latency_ms"]
        batch = http.post("/v1/predict/batch", json={"items": [features] * 25}).json()
        per_item = batch["latency_ms"] / batch["count"]

        assert per_item < single / 2, (
            f"batching is not amortising per-call cost: {per_item:.1f} ms/item "
            f"vs {single:.1f} ms for a single prediction"
        )


class TestDisclaimerIsEverywhere:
    """A health-adjacent service must say what it is on every surface."""

    def test_in_prediction_responses(self, http: Any, features: dict[str, Any]) -> None:
        _require_ready(http)
        body = http.post("/v1/predict", json={"features": features}).json()
        assert "NOT FOR CLINICAL USE" in body["disclaimer"].upper()

    def test_in_model_metadata(self, http: Any) -> None:
        assert "NOT FOR CLINICAL USE" in http.get("/v1/model").json()["disclaimer"].upper()

    def test_in_the_ui(self, http: Any) -> None:
        response = http.get("/")
        if response.status_code != 200:
            pytest.skip("UI not mounted in this deployment")
        assert "NOT FOR CLINICAL USE" in response.text.upper()
