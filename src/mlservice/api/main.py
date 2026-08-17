"""FastAPI application factory.

Startup and shutdown are handled in a lifespan context rather than event hooks,
which is the supported approach in modern FastAPI and — more usefully — gives a
single place where the ordering is visible.

**Startup deliberately does not fail on a missing model.** The process starts,
readiness reports 503, and the failure is visible through the probe and
``/v1/model``. Crashing instead would produce a CrashLoopBackOff whose real cause
is buried in restart logs, and would prevent an operator from querying the
service to ask *why* it cannot load a model. Degrade visibly, do not die
silently.

**Graceful shutdown** is uvicorn's ``--timeout-graceful-shutdown``, which stops
accepting connections and lets in-flight requests finish. For it to work at all,
uvicorn must be PID 1 and receive SIGTERM directly — hence the exec-form CMD in
Dockerfile.api. Wrapped in a shell it would not, and in-flight predictions would
be dropped on every deploy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from mlservice import __version__
from mlservice.api import metrics
from mlservice.api.errors import register_exception_handlers
from mlservice.api.middleware import RequestContextMiddleware
from mlservice.api.model_loader import get_store
from mlservice.api.routes import health, meta, outcomes, predict
from mlservice.config import PROJECT_ROOT, get_settings
from mlservice.logging_ import configure_logging, get_logger

log = get_logger(__name__)

UI_DIR = PROJECT_ROOT / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 — FastAPI passes it
    settings = get_settings()
    configure_logging(force=True)

    log.info(
        "service_starting",
        version=__version__,
        env=settings.env,
        api_version=settings.api.version,
        tracking_uri=settings.mlflow.tracking_uri or "(unset)",
    )

    store = get_store()
    try:
        model = store.load()
        metrics.set_model_loaded(
            True,
            name=model.name,
            version=str(model.version),
            source=model.source,
            schema_hash=model.feature_schema_hash,
        )
    except Exception as exc:  # start degraded, do not crash
        metrics.set_model_loaded(False)
        log.error(
            "startup_model_load_failed",
            error=str(exc)[:400],
            consequence="serving 503 on /health/ready until a model is available",
        )

    yield

    # Nothing to close: the log writer opens and closes per append (with fsync)
    # precisely so there is no buffered state to lose here.
    writer_stats = predict.get_writer().stats if predict._writer else {}
    log.info("service_stopping", prediction_log=writer_stats)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.api.title,
        version=__version__,
        lifespan=lifespan,
        description=(
            f"**{settings.api.disclaimer}**\n\n"
            "Predicts 30-day hospital readmission risk. This service exists to "
            "demonstrate ML *operations* — monitoring, drift detection, "
            "calibration-gated retraining and rollback. See `/v1/model` for what "
            "is currently serving."
        ),
        docs_url="/docs",
        redoc_url=None,
        openapi_tags=[
            {"name": "prediction", "description": "Scoring endpoints."},
            {"name": "outcomes", "description": "Late-arriving observed outcomes."},
            {"name": "health", "description": "Liveness and readiness — different questions."},
            {"name": "meta", "description": "What is serving, and metrics."},
        ],
    )

    # Order matters: this must be the outermost middleware so the request ID is
    # bound before anything else can log, and the timing covers the whole
    # request rather than a fraction of it.
    app.add_middleware(RequestContextMiddleware)

    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=False,  # no cookies or auth; nothing to protect
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=[settings.logging.request_id_header, "X-Response-Time-Ms"],
        )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(outcomes.router)
    app.include_router(meta.router)

    _mount_ui(app)

    log.info("app_created", routes=len(app.routes))
    return app


def _mount_ui(app: FastAPI) -> None:
    """Serve the single-page UI, if present.

    Guarded because the UI directory is excluded from some build contexts. A
    missing UI must degrade to an API-only service, not prevent startup.
    """
    index = UI_DIR / "index.html"
    if not index.is_file():
        log.warning("ui_not_found", path=str(UI_DIR), consequence="serving API only")

        @app.get("/", include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse("/docs")

        return

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def _index() -> FileResponse:
        return FileResponse(index)


app = create_app()


__all__ = ["app", "create_app", "lifespan"]
