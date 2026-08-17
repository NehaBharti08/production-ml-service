"""Error responses that tell the caller what to fix.

Follows RFC 9457 (problem details) so every failure has the same shape whatever
raised it. A caller integrating against this API should never have to parse two
different error formats depending on which layer rejected them.

Validation errors name **the field and the constraint**. FastAPI's default 422
body is a nested structure with a ``loc`` array that callers routinely misread;
this flattens it into something a human can act on without consulting the spec.

Every error carries the ``request_id``, so a bug report containing a screenshot
is enough to find the corresponding log lines.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlservice.api.model_loader import ModelNotLoadedError
from mlservice.logging_ import get_logger, get_request_id

log = get_logger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"


def problem(
    *,
    status_code: int,
    title: str,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"about:blank#{title.lower().replace(' ', '-')}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "request_id": get_request_id(),
        **extra,
    }
    if errors:
        body["errors"] = errors
    return JSONResponse(status_code=status_code, content=body, media_type=PROBLEM_CONTENT_TYPE)


def _flatten_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Turn pydantic's nested errors into flat, actionable entries."""
    flat: list[dict[str, Any]] = []
    for error in exc.errors():
        # loc is like ("body", "features", "time_in_hospital"); the leading
        # "body" is noise to the caller, who knows what they sent.
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        entry: dict[str, Any] = {
            "field": ".".join(location) or "(request)",
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "unknown"),
        }
        if "input" in error:
            # Truncated: an oversized payload echoed back in full is a log-bloat
            # and information-disclosure hazard.
            entry["received"] = str(error["input"])[:120]
        if (ctx := error.get("ctx")) and isinstance(ctx, dict):
            entry["constraint"] = {k: str(v)[:80] for k, v in ctx.items()}
        flat.append(entry)
    return flat


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so no failure escapes as an unshaped 500."""

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = _flatten_validation_errors(exc)
        # Logged per offending field so Phase 5 can count validation_errors_total
        # by field — a spike on one field is an upstream schema change, which is
        # very different from a caller sending occasional junk.
        log.warning(
            "request_validation_failed",
            path=request.url.path,
            error_count=len(errors),
            fields=[e["field"] for e in errors],
        )
        try:
            from mlservice.api import metrics

            for entry in errors:
                metrics.record_validation_error(entry["field"])
        except Exception:  # metrics must never mask the real error
            pass

        return problem(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            title="Validation failed",
            detail=(
                f"{len(errors)} field(s) failed validation. See 'errors' for the "
                "field name and the constraint each one violated."
            ),
            errors=errors,
        )

    @app.exception_handler(ModelNotLoadedError)
    async def _model_missing(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        # 503, not 500: the request was fine, the service is not ready. The
        # distinction matters because a load balancer retries a 503 elsewhere
        # and a 500 is counted against the error-rate SLO.
        log.error("model_not_loaded", path=request.url.path, error=str(exc))
        return problem(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            title="Model not loaded",
            detail=str(exc),
            retry_after_seconds=5,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The message is deliberately generic; the detail goes to the logs. An
        # exception string can contain internals a public endpoint should not
        # disclose. The request_id is how a caller and an operator connect the
        # two.
        log.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
        )
        return problem(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal server error",
            detail=(
                "The request could not be completed. Quote the request_id when reporting this."
            ),
        )


__all__ = ["PROBLEM_CONTENT_TYPE", "problem", "register_exception_handlers"]
