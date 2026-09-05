"""Structured logging with request-ID correlation.

This module is configured once, at the entrypoint of every process — API,
CLI, training job, monitoring job — so that a single log format spans the whole
system. It is built in Phase 0 rather than bolted on later for two reasons:

*   Correlation IDs are miserable to retrofit. Every call site has to be
    revisited, and the historical logs stay uncorrelated forever.
*   The prediction log (Phase 3) and everything reading it (Phases 5-7) assume
    that a ``request_id`` ties an inbound HTTP request to its prediction record,
    its latency measurement, and any error raised along the way.

Usage::

    from mlservice.logging_ import configure_logging, get_logger, bind_request_id

    configure_logging()
    log = get_logger(__name__)
    bind_request_id("abc-123")
    log.info("prediction_served", model_version="3", latency_ms=6.1)

Every log line is an event name plus key-value pairs, never an interpolated
sentence. ``"prediction_served"`` is greppable and aggregatable;
``f"Served prediction in {ms}ms"`` is neither.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from mlservice.config import get_settings

# --------------------------------------------------------------------------- #
# Request-ID propagation
# --------------------------------------------------------------------------- #

#: structlog's contextvars integration is async-safe and task-local, so
#: concurrent FastAPI requests cannot read each other's IDs.
_REQUEST_ID_KEY = "request_id"

_configured = False


def new_request_id() -> str:
    """Mint a correlation ID for a request that arrived without one."""
    return str(uuid.uuid4())


def bind_request_id(request_id: str | None = None) -> str:
    """Bind a request ID to the current context and return it.

    Passing an inbound ID (from the ``X-Request-ID`` header) preserves the
    caller's trace; passing ``None`` mints a fresh one.
    """
    rid = request_id or new_request_id()
    structlog.contextvars.bind_contextvars(**{_REQUEST_ID_KEY: rid})
    return rid


def get_request_id() -> str | None:
    """Return the request ID bound to the current context, if any."""
    value = structlog.contextvars.get_contextvars().get(_REQUEST_ID_KEY)
    return value if isinstance(value, str) else None


def clear_request_context() -> None:
    """Drop all bound context. Call at the end of a request or CLI command."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def request_context(request_id: str | None = None, **extra: Any) -> Iterator[str]:
    """Scope a request ID (plus any extra bound keys) to a block.

    Restores the previous context on exit, so nesting is safe.
    """
    previous = structlog.contextvars.get_contextvars()
    rid = bind_request_id(request_id)
    if extra:
        structlog.contextvars.bind_contextvars(**extra)
    try:
        yield rid
    finally:
        structlog.contextvars.clear_contextvars()
        if previous:
            structlog.contextvars.bind_contextvars(**previous)


# --------------------------------------------------------------------------- #
# Processors
# --------------------------------------------------------------------------- #


def _redact_sensitive(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """Replace configured sensitive keys with a marker, at any nesting depth.

    This is a safety net, not the primary control — the real control is not
    putting identifiers into log calls in the first place. But this is a
    health-adjacent service, and a defence that only works when everyone
    remembers is not a defence.
    """
    redact = set(get_settings().logging.redact_fields)
    if not redact:
        return event_dict

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ("<redacted>" if k in redact else scrub(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    for key in list(event_dict):
        if key in redact:
            event_dict[key] = "<redacted>"
        else:
            event_dict[key] = scrub(event_dict[key])
    return event_dict


def _add_service_context(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """Stamp every line with service identity, so logs stay attributable once
    they are aggregated across containers."""
    settings = get_settings()
    event_dict.setdefault("service", "mlservice")
    event_dict.setdefault("env", settings.env)
    return event_dict


def _drop_color_message(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """uvicorn duplicates its message into ``color_message``; drop the copy."""
    event_dict.pop("color_message", None)
    return event_dict


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def shared_processors() -> list[Processor]:
    """The processor chain applied to every event, before rendering.

    Exposed rather than inlined so that tests can assert against the *real*
    chain — contextvar merging, service stamping and redaction included. A test
    that reimplements this list would pass happily while the shipped pipeline
    was broken.
    """
    return [
        structlog.contextvars.merge_contextvars,  # injects request_id
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _drop_color_message,
        _add_service_context,
        _redact_sensitive,
    ]


def configure_logging(*, force: bool = False) -> None:
    """Configure structlog and route stdlib logging through it.

    Idempotent: calling it twice is a no-op unless ``force=True``. Uvicorn,
    MLflow and scikit-learn all log via stdlib ``logging``; routing them through
    the same pipeline means one format for the whole process rather than JSON
    lines interleaved with someone else's plaintext.
    """
    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    level = getattr(logging, settings.logging.level)

    shared: list[Processor] = shared_processors()

    if settings.logging.format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
        shared.append(structlog.processors.format_exc_info)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        shared.append(structlog.processors.ExceptionPrettyPrinter())

    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge: stdlib records get the same processors and the same renderer.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Logs go to STDERR, machine-readable output to STDOUT.
    #
    # This is not stylistic. `mlservice retrain check --json > triggers.json`
    # captured the log stream alongside the JSON and json.load failed with
    # "Extra data" - which broke the retraining workflow on its first real run.
    # A --json flag whose output cannot be redirected is a --json flag that does
    # not work, and the whole point of it is automation.
    #
    # Containers are unaffected: Docker and Kubernetes collect both streams, and
    # every log collector treats stderr as a normal source. Diagnostics on
    # stderr is the convention precisely so a program's actual output stays
    # pipeable.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; strip them so it propagates to root.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    # These are chatty at INFO and say nothing operationally useful.
    for name in ("urllib3", "botocore", "matplotlib", "git", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use.

    Auto-configuring means a module that logs during import cannot silently
    drop its output because nobody had called ``configure_logging`` yet.
    """
    if not _configured:
        configure_logging()
    return structlog.stdlib.get_logger(name)


__all__ = [
    "bind_request_id",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "request_context",
    "shared_processors",
]
