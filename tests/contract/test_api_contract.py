"""API contract: the shape callers depend on.

These are the tests that fail when a change would break an integration. They
assert the *contract* — field names, status codes, error shape — not behaviour,
which is what the behaviour suite in Phase 4 covers.

A stub model is injected rather than loading the real artifact. The contract must
hold regardless of which model is serving, and a test that needs a trained model
present cannot run on a fresh clone or in CI.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlservice.api.model_loader import LoadedModel, ModelStore, get_store
from mlservice.api.schemas import EXAMPLE_FEATURES

pytestmark = pytest.mark.contract


class _StubPipeline:
    """Returns a fixed probability. Deterministic, and needs no artifact."""

    def __init__(self, probability: float = 0.42) -> None:
        self.probability = probability

    def predict_proba(self, frame: Any) -> list[list[float]]:
        return [[1 - self.probability, self.probability] for _ in range(len(frame))]


def _stub_model(probability: float = 0.42, threshold: float = 0.1011) -> LoadedModel:
    return LoadedModel(
        pipeline=_StubPipeline(probability),
        name="readmission-risk",
        version="test-1",
        stage="champion",
        source="registry",
        feature_schema_hash="testhash",
        decision_threshold=threshold,
        loaded_at=time.time(),
    )


@pytest.fixture
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """TestClient with a stub model and an isolated prediction log."""
    from mlservice.api import main
    from mlservice.api.routes import predict as predict_route
    from mlservice.monitoring import prediction_log

    # Redirect the log so tests never touch real monitoring data.
    monkeypatch.setattr(
        predict_route,
        "_writer",
        prediction_log.PredictionLogWriter(tmp_path / "predictions.ndjson"),
    )

    store = ModelStore()
    store._model = _stub_model()  # bypass loading; the contract is model-agnostic

    app = main.create_app()
    app.dependency_overrides[get_store] = lambda: store

    # raise_server_exceptions=False so the 500 handler's response shape is
    # asserted rather than the exception propagating into the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def unready_client(tmp_path: Any) -> Any:
    """A client whose store holds no model, for the 503 paths."""
    from mlservice.api import main

    app = main.create_app()
    app.dependency_overrides[get_store] = lambda: ModelStore()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestPredictContract:
    def test_returns_the_documented_fields(self, client: Any) -> None:
        r = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES})
        assert r.status_code == 200
        body = r.json()
        for field in (
            "prediction_id",
            "request_id",
            "readmission_probability",
            "flagged",
            "decision_threshold",
            "model",
            "latency_ms",
            "disclaimer",
        ):
            assert field in body, f"contract field missing: {field}"

    def test_model_block_identifies_provenance(self, client: Any) -> None:
        """`source` is what an operator reads first during an incident."""
        body = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES}).json()
        for field in ("name", "version", "stage", "source", "feature_schema_hash"):
            assert field in body["model"]

    def test_flagged_follows_the_threshold_not_a_hardcoded_half(self, client: Any) -> None:
        """0.42 against a 0.1011 threshold must flag — the Phase 3 bug."""
        body = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES}).json()
        assert body["readmission_probability"] == pytest.approx(0.42)
        assert body["decision_threshold"] == pytest.approx(0.1011)
        assert body["flagged"] is True

    def test_every_response_carries_the_disclaimer(self, client: Any) -> None:
        """A JSON-only consumer must still be told this is not a clinical tool."""
        body = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES}).json()
        assert "NOT FOR CLINICAL USE" in body["disclaimer"].upper()

    def test_prediction_ids_are_unique_across_calls(self, client: Any) -> None:
        ids = {
            client.post("/v1/predict", json={"features": EXAMPLE_FEATURES}).json()["prediction_id"]
            for _ in range(5)
        }
        assert len(ids) == 5


class TestRequestIdPropagation:
    def test_inbound_id_is_echoed_back(self, client: Any) -> None:
        r = client.post(
            "/v1/predict",
            json={"features": EXAMPLE_FEATURES},
            headers={"X-Request-ID": "caller-supplied"},
        )
        assert r.headers["X-Request-ID"] == "caller-supplied"
        assert r.json()["request_id"] == "caller-supplied"

    def test_an_id_is_minted_when_absent(self, client: Any) -> None:
        r = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES})
        assert r.headers.get("X-Request-ID")
        assert r.json()["request_id"] not in ("", "unknown", None)

    def test_server_timing_header_is_present(self, client: Any) -> None:
        r = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES})
        assert float(r.headers["X-Response-Time-Ms"]) >= 0


class TestErrorContract:
    def test_validation_error_names_the_field(self, client: Any) -> None:
        bad = {**EXAMPLE_FEATURES, "time_in_hospital": 500}
        r = client.post("/v1/predict", json={"features": bad})
        assert r.status_code == 422

        body = r.json()
        assert body["title"] == "Validation failed"
        assert body["request_id"]
        fields = [e["field"] for e in body["errors"]]
        assert "features.time_in_hospital" in fields

    def test_unknown_field_is_rejected(self, client: Any) -> None:
        """Silently ignoring it would let a caller believe it was used."""
        bad = {**EXAMPLE_FEATURES, "not_a_feature": 1}
        assert client.post("/v1/predict", json={"features": bad}).status_code == 422

    def test_age_as_a_bare_number_is_rejected_with_a_useful_message(self, client: Any) -> None:
        bad = {**EXAMPLE_FEATURES, "age": "75"}
        r = client.post("/v1/predict", json={"features": bad})
        assert r.status_code == 422
        assert "band" in str(r.json()["errors"]).lower()

    def test_errors_use_the_problem_content_type(self, client: Any) -> None:
        r = client.post("/v1/predict", json={"features": {**EXAMPLE_FEATURES, "age": "75"}})
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_missing_model_returns_503_not_500(self, unready_client: Any) -> None:
        """The request was fine; the service is not ready.

        A load balancer retries a 503 elsewhere, and a 500 would be counted
        against the error-rate SLO for something that is not an error.
        """
        r = unready_client.post("/v1/predict", json={"features": EXAMPLE_FEATURES})
        assert r.status_code == 503
        assert r.json()["title"] == "Model not loaded"


class TestBatchContract:
    def test_returns_one_prediction_per_item(self, client: Any) -> None:
        r = client.post("/v1/predict/batch", json={"items": [EXAMPLE_FEATURES] * 3})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert len(body["predictions"]) == 3
        assert len({p["prediction_id"] for p in body["predictions"]}) == 3

    def test_shares_one_batch_id(self, client: Any) -> None:
        body = client.post("/v1/predict/batch", json={"items": [EXAMPLE_FEATURES] * 2}).json()
        assert body["batch_id"]

    def test_empty_batch_is_rejected(self, client: Any) -> None:
        assert client.post("/v1/predict/batch", json={"items": []}).status_code == 422

    def test_oversized_batch_returns_413(self, client: Any) -> None:
        """configs/test.yaml sets max_batch_size to 10 so this stays cheap."""
        r = client.post("/v1/predict/batch", json={"items": [EXAMPLE_FEATURES] * 50})
        assert r.status_code == 413


class TestHealthContract:
    def test_liveness_ignores_model_state(self, unready_client: Any) -> None:
        """Liveness must not depend on anything external.

        If it did, a registry outage would restart every replica in a loop —
        turning a degraded dependency into a total outage.
        """
        r = unready_client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_readiness_fails_without_a_model(self, unready_client: Any) -> None:
        r = unready_client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"

    def test_readiness_passes_with_a_model(self, client: Any) -> None:
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["checks"]["model_loaded"] is True


class TestOutcomeContract:
    def test_accepts_an_outcome_for_a_prediction(self, client: Any) -> None:
        pid = client.post("/v1/predict", json={"features": EXAMPLE_FEATURES}).json()[
            "prediction_id"
        ]
        r = client.post(
            "/v1/outcomes",
            json={"prediction_id": pid, "readmitted_within_30_days": True, "source": "test"},
        )
        assert r.status_code == 200
        assert r.json()["recorded"] is True

    def test_unknown_prediction_id_is_accepted(self, client: Any) -> None:
        """Deliberate: validating against the log would be O(n) per request and
        would reject legitimate late outcomes after rotation. Phase 6 counts
        unmatched IDs, which is where a systematic mismatch should surface."""
        r = client.post(
            "/v1/outcomes",
            json={"prediction_id": "never-seen-before", "readmitted_within_30_days": False},
        )
        assert r.status_code == 200


class TestMetaContract:
    def test_model_endpoint_reports_provenance(self, client: Any) -> None:
        body = client.get("/v1/model").json()
        assert body["loaded"] is True
        assert body["source"] == "registry"
        assert "NOT FOR CLINICAL USE" in body["disclaimer"].upper()

    def test_metrics_endpoint_serves_prometheus_text(self, client: Any) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "http_requests_total" in r.text

    def test_endpoint_labels_use_the_route_template(self, client: Any) -> None:
        """Labels must name the route, not be bucketed as 'unmatched'.

        This regressed once: the route template was read before `call_next`, so
        every metric was labelled 'unmatched' and per-endpoint dashboards would
        have been useless.
        """
        client.post("/v1/predict", json={"features": EXAMPLE_FEATURES})
        text = client.get("/metrics").text
        assert 'endpoint="/v1/predict"' in text


class TestOpenApiStability:
    def test_schema_generates(self, client: Any) -> None:
        assert client.get("/openapi.json").status_code == 200

    def test_documented_paths_are_present(self, client: Any) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/v1/predict",
            "/v1/predict/batch",
            "/v1/outcomes",
            "/v1/model",
            "/health/live",
            "/health/ready",
        ):
            assert path in paths, f"documented path missing: {path}"

    def test_endpoints_are_versioned(self, client: Any) -> None:
        """Health and metrics are infrastructure; everything else is versioned."""
        paths = client.get("/openapi.json").json()["paths"]
        unversioned = [
            p
            for p in paths
            if not p.startswith("/v1/") and not p.startswith("/health/") and p != "/metrics"
        ]
        assert not unversioned, f"unversioned API paths: {unversioned}"
