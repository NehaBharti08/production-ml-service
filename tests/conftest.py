"""Shared pytest fixtures.

Two things this file guarantees, because getting either wrong makes the whole
suite untrustworthy:

*   **Tests never touch real config state.** ``get_settings`` is ``lru_cache``d,
    so a test that mutates the environment would otherwise leak its settings
    into every test that runs after it — producing failures that depend on
    ordering and vanish when run alone.
*   **Tests never touch real data.** ``MLSERVICE_ENV`` is forced to ``test``,
    which resolves ``configs/test.yaml``: no MLflow URI, no real paths.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True, scope="session")
def _force_test_env() -> Iterator[None]:
    """Pin MLSERVICE_ENV=test for the whole session, before anything imports."""
    previous = os.environ.get("MLSERVICE_ENV")
    os.environ["MLSERVICE_ENV"] = "test"
    yield
    if previous is None:
        os.environ.pop("MLSERVICE_ENV", None)
    else:
        os.environ["MLSERVICE_ENV"] = previous


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Drop cached settings and thresholds around every test.

    Both before and after: before, so a test starts from a clean resolve; after,
    so a test that set an env var cannot contaminate its successors.
    """
    from mlservice.config import get_settings, get_thresholds

    get_settings.cache_clear()
    get_thresholds.cache_clear()
    yield
    get_settings.cache_clear()
    get_thresholds.cache_clear()


@pytest.fixture(autouse=True)
def _clear_log_context() -> Iterator[None]:
    """Clear structlog contextvars so a bound request_id cannot leak."""
    import structlog

    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set MLSERVICE_* environment variables and re-resolve settings.

    Usage::

        def test_override(env):
            settings = env(MLSERVICE_API__PORT="9999")
            assert settings.api.port == 9999
    """
    from mlservice.config import get_settings

    def _set(**kwargs: str) -> Any:
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return get_settings()

    return _set


@pytest.fixture
def captured_logs() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog events as dicts, after the real processor chain.

    Deliberately built from :func:`mlservice.logging_.shared_processors` rather
    than a reimplementation, so these tests exercise the pipeline that actually
    ships — contextvar merging, service stamping and redaction included. A
    fixture that rebuilt the chain would happily pass while production logging
    was broken.

    Events are captured as dicts rather than rendered text: asserting on JSON
    string output is brittle, and what matters is what the event *carries*.
    """
    import structlog

    from mlservice.logging_ import configure_logging, shared_processors

    # Configure for real FIRST. get_logger() lazily calls configure_logging()
    # when it has not run yet, which would overwrite the capture config
    # installed below — making these tests pass or fail depending on whether an
    # earlier test in the session happened to configure logging already.
    configure_logging()

    events: list[dict[str, Any]] = []

    def _capture(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        events.append(dict(event_dict))
        raise structlog.DropEvent  # stop before rendering; keeps test output clean

    previous = structlog.get_config()
    structlog.configure(
        processors=[*shared_processors(), _capture],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # or a cached logger keeps the old chain
    )
    try:
        yield events
    finally:
        structlog.configure(**previous)


@pytest.fixture
def project_root() -> Path:
    from mlservice.config import PROJECT_ROOT

    return PROJECT_ROOT
