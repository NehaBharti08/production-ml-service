"""Liveness and readiness — deliberately different questions.

Conflating these is one of the most consequential mistakes in a Kubernetes
deployment, because the two probes have opposite failure semantics:

*   **Liveness** asks "is this process wedged?" A failure gets the container
    **restarted**. It must therefore not depend on anything external. If
    liveness checked the model registry, an MLflow outage would restart every
    replica in a loop — turning a degraded dependency into a total outage.

*   **Readiness** asks "should this instance receive traffic?" A failure gets it
    **removed from the load balancer** but left running. This is where the model
    check belongs: an instance with no working model should not be served
    traffic, but restarting it will not help if the artifact is genuinely
    missing.

That distinction is what makes the Phase 7 rollback demo work: a bad model makes
readiness fail, traffic drains away, and the deployment is rolled back — without
a restart storm obscuring what happened.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Response, status

from mlservice.api.model_loader import ModelStore, get_store
from mlservice.api.schemas import HealthResponse
from mlservice.logging_ import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness — is the process running?",
)
async def liveness() -> HealthResponse:
    """Returns 200 while the event loop can serve a request.

    Deliberately checks **nothing** external. Any dependency here would let an
    external outage trigger a restart loop, which is strictly worse than serving
    degraded.
    """
    return HealthResponse(
        status="alive",
        checks={"uptime_seconds": round(time.time() - _STARTED_AT, 1)},
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness — should this instance receive traffic?",
    responses={503: {"description": "Not ready: no usable model is loaded."}},
)
async def readiness(response: Response, store: ModelStore = Depends(get_store)) -> HealthResponse:
    """503 until a model is loaded *and* has passed a canary inference.

    Returning 200 here while unable to score would mean the load balancer sends
    traffic that is guaranteed to fail — the exact situation readiness exists to
    prevent.
    """
    checks: dict[str, Any] = {"model_loaded": store.is_loaded}

    if not store.is_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        checks["error"] = store.last_error or "model has not been loaded"
        log.warning("readiness_failed", **checks)
        return HealthResponse(status="not_ready", checks=checks)

    model = store.model
    checks.update(
        {
            "model_version": model.version,
            "model_source": model.source,
            "feature_schema_hash": model.feature_schema_hash,
            "loaded_seconds_ago": round(time.time() - model.loaded_at, 1),
        }
    )
    return HealthResponse(status="ready", checks=checks)


__all__ = ["router"]
