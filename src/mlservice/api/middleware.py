"""Request-ID propagation, timing, and access logging.

The correlation spine built in Phase 0 wired into the HTTP layer. Every request
gets an ID — honoured from the inbound header if present, minted otherwise — and
that ID appears on every log line the request produces, in the response header,
in any error body, and in the prediction log record.

Timing is measured with ``perf_counter``, not ``time()``: the wall clock can step
backwards (NTP correction, DST on a badly configured host) and produce negative
durations, which then poison a latency histogram permanently.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from mlservice.api import metrics
from mlservice.config import get_settings
from mlservice.logging_ import bind_request_id, clear_request_context, get_logger

log = get_logger(__name__)

#: Paths excluded from access logging. Prometheus scrapes /metrics every 15s and
#: probes hit /health constantly; logging them would bury real traffic at a
#: ratio of roughly 100:1 and make the log useless during an incident. They are
#: still measured — only the per-request log line is suppressed.
_QUIET_PATHS = frozenset({"/metrics", "/health/live", "/health/ready", "/favicon.ico"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        settings = get_settings()
        header = settings.logging.request_id_header

        # Honour an inbound ID so a trace survives across services; mint one
        # otherwise. Truncated because this value is echoed into a response
        # header and every log line — an unbounded caller-supplied string there
        # is a log-injection and memory hazard.
        inbound = request.headers.get(header)
        request_id = bind_request_id(inbound[:64] if inbound else None)
        request.state.request_id = request_id

        started = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            # The exception handler produces the body; this only has to make
            # sure the failure is counted, timed and logged. Re-raised at once.
            duration = time.perf_counter() - started
            failed_endpoint = self._route_template(request)
            metrics.record_request(request.method, failed_endpoint, 500, duration)
            log.exception(
                "request_failed",
                method=request.method,
                endpoint=failed_endpoint,
                duration_ms=round(duration * 1000, 3),
            )
            # Context is cleared here rather than in a `finally`, because the
            # success path must clear it only *after* the response headers are
            # set — a shared finally would wipe the request ID before it is
            # written to the response.
            clear_request_context()
            raise

        duration = time.perf_counter() - started
        # Resolved AFTER call_next, not before: the router populates
        # scope["route"] while handling the request, so reading it earlier
        # labelled every single metric "unmatched" and made per-endpoint
        # dashboards useless. Caught by inspecting real /metrics output.
        endpoint = self._route_template(request)
        metrics.record_request(request.method, endpoint, status_code, duration)

        response.headers[header] = request_id
        # Surfaced so a caller can compare their observed latency against what
        # the server measured, which separates network time from service time.
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.2f}"

        if request.url.path not in _QUIET_PATHS:
            log.info(
                "request_completed",
                method=request.method,
                endpoint=endpoint,
                path=request.url.path,
                status=status_code,
                duration_ms=round(duration * 1000, 3),
            )

        clear_request_context()
        return response

    @staticmethod
    def _route_template(request: Request) -> str:
        """The matched route pattern, not the concrete path.

        A metric label that varies per request would grow Prometheus's series
        count without bound. Using the template keeps cardinality fixed at the
        number of routes.
        """
        route = request.scope.get("route")
        template = getattr(route, "path", None)
        if template:
            return str(template)
        # Unmatched request (404): bucket them together rather than creating a
        # new series for every scanned URL a bot tries.
        return "unmatched"


__all__ = ["RequestContextMiddleware"]
