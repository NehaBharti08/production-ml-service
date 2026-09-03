"""MLflow tracking and model registry access.

Tracking must never be able to break training. If the MLflow server is
unreachable, this falls back to a local file store and logs the substitution
loudly — a failed experiment-tracking call is not a reason to lose a training
run. The same principle governs model loading in Phase 3: the API keeps a local
fallback artifact so serving does not depend on the registry being up.

The fallback is also what makes Phase 2 runnable before Docker is installed:
the compose MLflow server does not exist yet, and waiting for it would block
work that does not actually need it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from mlservice.config import PROJECT_ROOT, get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)

#: Local store used when no server is reachable. Gitignored — an MLflow
#: artifact store must never enter version control.
#:
#: SQLite rather than the plain './mlruns' file store: MLflow 3.x put the
#: filesystem backend into maintenance mode and refuses it by default, and the
#: file store never supported the model registry anyway. SQLite is the
#: documented local replacement and gives us registry support offline, which is
#: what makes the Phase 7 promotion/rollback work testable without a server.
LOCAL_STORE = PROJECT_ROOT / "mlruns"
LOCAL_DB = LOCAL_STORE / "mlflow.db"
LOCAL_ARTIFACTS = LOCAL_STORE / "artifacts"

_SERVER_TIMEOUT_SECONDS = 3


def server_reachable(uri: str, timeout: int = _SERVER_TIMEOUT_SECONDS) -> bool:
    """Cheap liveness probe so a dead server degrades instead of hanging.

    The DNS lookup is done separately and first, because **urlopen's timeout
    does not cover name resolution**. Configs point at compose service names
    like ``http://mlflow:5000``; off the compose network that name does not
    resolve, and the lookup blocked for far longer than the socket timeout —
    observed as a 15-second delay on every service start in Phase 3, and as a
    multi-minute hang when the registry was queried from a script.

    Resolving in a daemon thread bounds it: if the lookup has not returned by
    the deadline the host is treated as unreachable and the thread is abandoned
    rather than waited on.
    """
    if not uri or not uri.startswith("http"):
        return False

    import socket
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(uri)
    host, port = parsed.hostname, parsed.port or 80
    if not host:
        return False

    resolved: list[bool] = []

    def _resolve() -> None:
        try:
            socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
            resolved.append(True)
        except OSError:
            resolved.append(False)

    thread = threading.Thread(target=_resolve, daemon=True)
    thread.start()
    thread.join(timeout)

    if not resolved or not resolved[0]:
        log.debug("registry_host_unresolvable", host=host, timeout_s=timeout)
        return False

    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as r:
            return bool(200 <= r.status < 300)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def resolve_tracking_uri() -> tuple[str, bool]:
    """Return the tracking URI to use, and whether it is the real server."""
    configured = get_settings().mlflow.tracking_uri

    if server_reachable(configured):
        return configured, True

    LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Forward slashes: SQLAlchemy URIs are not Windows path syntax.
    fallback = f"sqlite:///{LOCAL_DB.resolve().as_posix()}"
    log.warning(
        "mlflow_server_unreachable_using_local_store",
        configured=configured or "(unset)",
        fallback=fallback,
        consequence="runs are tracked locally; start the compose stack to use the server",
    )
    return fallback, False


def setup_tracking(experiment: str | None = None) -> tuple[str, bool]:
    """Point MLflow at a working store and select the experiment."""
    settings = get_settings()
    uri, is_server = resolve_tracking_uri()

    mlflow.set_tracking_uri(uri)
    name = experiment or settings.mlflow.experiment_name

    # With the local SQLite backend the DB holds only metadata, so the artifact
    # location has to be set explicitly at experiment-creation time — it cannot
    # be changed afterwards.
    if not is_server and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=LOCAL_ARTIFACTS.resolve().as_uri())

    mlflow.set_experiment(name)

    log.info("mlflow_configured", uri=uri, experiment=name, server=is_server)
    return uri, is_server


@contextmanager
def run(name: str, tags: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start an MLflow run, tagged so it is identifiable months later."""
    settings = get_settings()
    base_tags = {
        "phase": "2",
        "dataset": settings.data.dataset_name,
        "env": settings.env,
        # Recorded on every run so a model can always be traced back to the
        # exact split policy that produced its training data.
        "split_policy": "chronological-encounter_id-proxy+censoring-buffer",
        **(tags or {}),
    }
    with mlflow.start_run(run_name=name, tags=base_tags) as active:
        log.info("mlflow_run_started", run_name=name, run_id=active.info.run_id)
        yield active


def register(model_uri: str, name: str, alias: str | None = None) -> Any:
    """Register a model version and optionally point an alias at it.

    Aliases rather than the deprecated stage transitions: an alias is a movable
    pointer, so promotion and rollback are both a single atomic re-point rather
    than a multi-step state change that can be interrupted halfway.
    """
    client = MlflowClient()
    version = mlflow.register_model(model_uri=model_uri, name=name)

    if alias:
        client.set_registered_model_alias(name, alias, version.version)
        log.info("model_alias_set", model=name, alias=alias, version=version.version)

    log.info("model_registered", model=name, version=version.version, uri=model_uri)
    return version


def load_by_alias(name: str, alias: str) -> Any:
    """Load the model an alias currently points at."""
    uri = f"models:/{name}@{alias}"
    log.info("model_loading", uri=uri)
    return mlflow.sklearn.load_model(uri)


def current_version(name: str, alias: str) -> str | None:
    """Which version an alias points at, or None if unset.

    Used by the Phase 7 rollback path to record what the alias pointed at
    *before* a promotion, so the previous version can be restored.
    """
    try:
        # str() rather than returning mlflow's untyped attribute directly:
        # MLflow reports versions as strings, but the object is untyped, so
        # without this the return value is Any and the annotation is a fiction.
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    except Exception:
        return None


def save_local_fallback(pipeline: Any, path: Path | None = None) -> Path:
    """Persist the champion beside the code as the API's fallback artifact.

    Gitignored. Baked into the serving image at build time so the API can start
    and serve when the registry is unreachable — availability should not depend
    on the tracking server, and the compose file deliberately does not make the
    API wait for MLflow.
    """
    import joblib

    target = path or (get_settings().paths.models / "champion" / "model.joblib")
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    log.info("local_fallback_saved", path=str(target), size_kb=round(target.stat().st_size / 1e3))
    return target


__all__ = [
    "LOCAL_STORE",
    "current_version",
    "load_by_alias",
    "register",
    "resolve_tracking_uri",
    "run",
    "save_local_fallback",
    "server_reachable",
    "setup_tracking",
]
