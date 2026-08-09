"""Structured logging: correlation IDs and redaction.

Both properties are load-bearing beyond Phase 0. The prediction log written in
Phase 3, and everything reading it in Phases 5-7, assumes a request_id ties an
inbound request to its prediction, its latency and any error along the way.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mlservice.logging_ import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_id,
    new_request_id,
    request_context,
)

pytestmark = pytest.mark.unit


class TestRequestId:
    def test_minted_when_absent(self) -> None:
        rid = bind_request_id()
        assert rid
        assert get_request_id() == rid

    def test_inbound_id_is_preserved(self) -> None:
        """A caller's trace must survive, or cross-service correlation breaks."""
        assert bind_request_id("caller-supplied-123") == "caller-supplied-123"
        assert get_request_id() == "caller-supplied-123"

    def test_ids_are_unique(self) -> None:
        assert len({new_request_id() for _ in range(100)}) == 100

    def test_cleared_context_has_no_id(self) -> None:
        bind_request_id()
        clear_request_context()
        assert get_request_id() is None

    def test_context_manager_restores_previous(self) -> None:
        outer = bind_request_id("outer")
        with request_context("inner") as inner:
            assert inner == "inner"
            assert get_request_id() == "inner"
        assert get_request_id() == outer

    def test_id_appears_on_every_event(self, captured_logs: list[dict[str, Any]]) -> None:
        log = get_logger("test")
        with request_context("trace-abc"):
            log.info("first_event")
            log.info("second_event")
        assert [e["request_id"] for e in captured_logs] == ["trace-abc", "trace-abc"]

    def test_extra_bound_keys_propagate(self, captured_logs: list[dict[str, Any]]) -> None:
        log = get_logger("test")
        with request_context("trace-abc", model_version="7"):
            log.info("prediction_served")
        assert captured_logs[0]["model_version"] == "7"


class TestRedaction:
    """A safety net, not the primary control.

    The real control is not putting identifiers into log calls. But a defence
    that only works when everyone remembers is not a defence, and this is a
    health-adjacent service.
    """

    def test_redacts_configured_top_level_fields(self, captured_logs: list[dict[str, Any]]) -> None:
        get_logger("test").info("evt", patient_nbr=8222157, encounter_id=2278392)
        assert captured_logs[0]["patient_nbr"] == "<redacted>"
        assert captured_logs[0]["encounter_id"] == "<redacted>"

    def test_redacts_nested_fields(self, captured_logs: list[dict[str, Any]]) -> None:
        """Identifiers usually arrive inside a payload dict, not at top level."""
        get_logger("test").info("evt", features={"patient_nbr": 999, "age": "[70-80)"})
        assert captured_logs[0]["features"] == {"patient_nbr": "<redacted>", "age": "[70-80)"}

    def test_redacts_inside_lists(self, captured_logs: list[dict[str, Any]]) -> None:
        """Batch prediction logs a list of records."""
        get_logger("test").info("evt", batch=[{"patient_nbr": 1}, {"patient_nbr": 2}])
        assert captured_logs[0]["batch"] == [
            {"patient_nbr": "<redacted>"},
            {"patient_nbr": "<redacted>"},
        ]

    def test_leaves_non_sensitive_fields_intact(self, captured_logs: list[dict[str, Any]]) -> None:
        get_logger("test").info("evt", latency_ms=6.1, model_version="3")
        assert captured_logs[0]["latency_ms"] == 6.1
        assert captured_logs[0]["model_version"] == "3"


class TestConfiguration:
    def test_is_idempotent(self) -> None:
        configure_logging()
        configure_logging()  # must not raise or duplicate handlers

    def test_stamps_service_identity(self, captured_logs: list[dict[str, Any]]) -> None:
        """Logs must stay attributable once aggregated across containers."""
        get_logger("test").info("evt")
        assert captured_logs[0]["service"] == "mlservice"
        assert captured_logs[0]["env"] == "test"

    def test_json_output_is_parseable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What ships to a log aggregator must actually be valid JSON.

        Redirects the configured handler's stream rather than using capsys:
        the handler binds sys.stdout at configuration time, so capsys — which
        replaces sys.stdout afterwards — never sees these records.
        """
        import io
        import logging as stdlib_logging

        from mlservice.config import get_settings

        monkeypatch.setenv("MLSERVICE_LOGGING__FORMAT", "json")
        # configs/test.yaml pins WARNING to keep suite output readable; this
        # test needs an INFO record to actually reach the handler.
        monkeypatch.setenv("MLSERVICE_LOGGING__LEVEL", "INFO")
        get_settings.cache_clear()
        configure_logging(force=True)

        stream = io.StringIO()
        handler = stdlib_logging.getLogger().handlers[0]
        monkeypatch.setattr(handler, "stream", stream)

        with request_context("json-test"):
            get_logger("test").info("evt", latency_ms=1.5, patient_nbr=42)

        payload = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert payload["event"] == "evt"
        assert payload["request_id"] == "json-test"
        assert payload["latency_ms"] == 1.5
        assert payload["service"] == "mlservice"
        # Redaction must survive rendering, not just the processor chain.
        assert payload["patient_nbr"] == "<redacted>"
