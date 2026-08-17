"""Model and service metadata.

Exists so an operator can answer "what is actually serving right now?" without
shell access to the container. During an incident that question comes first, and
``model_source`` in particular distinguishes "serving from the registry" from
"serving the baked-in fallback because MLflow is down" — two very different
situations that look identical from the outside.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mlservice.api import metrics
from mlservice.api.model_loader import ModelStore, get_store
from mlservice.config import get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["meta"])


@router.get("/v1/model", summary="What model is serving, and its provenance")
async def model_metadata(store: ModelStore = Depends(get_store)) -> dict[str, Any]:
    settings = get_settings()

    if not store.is_loaded:
        return {
            "loaded": False,
            "error": store.last_error,
            "disclaimer": settings.api.disclaimer,
        }

    model = store.model
    return {
        "loaded": True,
        "name": model.name,
        "version": model.version,
        "stage": model.stage,
        # The field an operator reads first: "registry" or "local_fallback".
        "source": model.source,
        "feature_schema_hash": model.feature_schema_hash,
        "decision_threshold": model.decision_threshold,
        "loaded_seconds_ago": round(time.time() - model.loaded_at, 1),
        "api_version": settings.api.version,
        # Carried here as well as in every prediction response. A consumer
        # inspecting the service should not have to make a prediction to be told
        # this is not a clinical tool.
        "disclaimer": settings.api.disclaimer,
    }


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus exposition from the project's own registry.

    Uses the dedicated registry rather than the global default, so the output
    contains this service's metrics and not whatever any imported library
    happened to register.
    """
    return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)
