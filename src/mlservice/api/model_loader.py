"""Model loading: registry first, baked-in artifact as fallback.

Availability must not depend on the tracking server. The compose file
deliberately does not make the API wait for MLflow to be healthy, and this is
the code that makes that safe — if the registry is unreachable, the service
loads the artifact baked into the image and says so loudly in the logs and in
``/v1/model``.

The distinction is recorded per prediction (``model_source``) rather than only
logged at startup. During an incident, "which model actually produced this
score" is the first question, and a startup log line scrolled past an hour ago
does not answer it for a specific prediction.

Readiness is deliberately coupled to a **canary inference**, not merely to the
object being non-None. A model that loads but cannot score — a mismatched
sklearn version, a corrupted artifact — would otherwise pass readiness and then
fail every real request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlservice.config import get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)


@dataclass
class LoadedModel:
    pipeline: Any
    name: str
    version: str
    stage: str
    source: str  # "registry" | "local_fallback"
    feature_schema_hash: str
    decision_threshold: float
    loaded_at: float

    def predict_proba(self, row: dict[str, Any]) -> float:
        """Score one record. Returns P(readmitted within 30 days)."""
        frame = pd.DataFrame([row])
        return float(self.pipeline.predict_proba(frame)[0][1])

    def predict_proba_batch(self, rows: list[dict[str, Any]]) -> list[float]:
        """Score many records in one call.

        Batched rather than looped: the per-call overhead of sklearn's input
        validation dominates the arithmetic for a linear model, so a loop over
        500 rows costs far more than one 500-row frame.
        """
        frame = pd.DataFrame(rows)
        return [float(p[1]) for p in self.pipeline.predict_proba(frame)]


class ModelNotLoadedError(RuntimeError):
    """No model is available — the service must report itself unready."""


class ModelStore:
    """Holds the active model and knows how to (re)load it.

    Thread-safe because uvicorn serves concurrent requests and a hot reload
    (Phase 7 promotion) swaps the model underneath them. Readers take a
    reference to the immutable :class:`LoadedModel` rather than reading through
    the store, so a swap mid-request cannot produce a half-updated view.
    """

    def __init__(self) -> None:
        self._model: LoadedModel | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def model(self) -> LoadedModel:
        model = self._model  # single read; never dereference twice
        if model is None:
            raise ModelNotLoadedError(self._last_error or "model has not been loaded")
        return model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------ loading

    def load(self) -> LoadedModel:
        """Load from the registry, falling back to the local artifact."""
        settings = get_settings()

        with self._lock:
            # Every source records why it declined, so a failure names ALL the
            # reasons rather than whichever exception happened to fire last.
            #
            # This mattered: the container reported "No module named 'mlflow'"
            # when the decisive problem was an absent artifact. The registry
            # error was stale — it was simply the only one anything recorded,
            # because a source returning None set nothing. That message sent
            # the investigation at the wrong layer entirely.
            reasons: list[str] = []

            for attempt in (self._load_from_registry, self._load_from_local):
                try:
                    model = attempt()
                except Exception as exc:  # try the next source
                    reason = f"{attempt.__name__}: {str(exc)[:200]}"
                    log.warning(
                        "model_load_attempt_failed",
                        source=attempt.__name__,
                        error=str(exc)[:300],
                    )
                    reasons.append(reason)
                    self._last_error = str(exc)[:300]
                    continue

                if model is None:
                    reasons.append(f"{attempt.__name__}: declined (see logs)")
                    continue

                # A model that loads but cannot score is not a loaded model.
                self._canary_inference(model)
                self._model = model
                self._last_error = None
                log.info(
                    "model_loaded",
                    source=model.source,
                    version=model.version,
                    threshold=model.decision_threshold,
                    schema_hash=model.feature_schema_hash,
                )
                return model

            detail = "; ".join(reasons) or "no source was attempted"
            self._last_error = detail[:300]
            raise ModelNotLoadedError(
                f"no model could be loaded. Reasons: {detail}. "
                f"Expected local artifact at {self._fallback_path(settings)} "
                "(the serving image bakes it in at build time; if it is missing, "
                "the image was built without one). "
                "Otherwise start MLflow and set mlflow.tracking_uri."
            )

    def _load_from_registry(self) -> LoadedModel | None:
        from mlservice.models import registry

        settings = get_settings()
        uri = settings.mlflow.tracking_uri

        # Skip fast when the URI is empty (the HF Space case) or unreachable,
        # rather than letting mlflow burn its internal retry budget on startup.
        if not uri or not registry.server_reachable(uri):
            log.info("registry_unreachable_skipping", configured=uri or "(unset)")
            return None

        try:
            import mlflow
        except ImportError as exc:
            # Expected in the serving image: the `serve` dependency group
            # deliberately excludes mlflow to keep the CVE surface small. This
            # is a design decision, not a broken install — so it is reported as
            # a declined source rather than an error, and the local artifact is
            # the intended path from here.
            raise RuntimeError(
                "mlflow is not installed in this image (serve group excludes it) "
                "— the local artifact is the intended source here"
            ) from exc

        mlflow.set_tracking_uri(uri)
        pipeline = registry.load_by_alias(settings.model.name, settings.model.serving_alias)
        version = registry.current_version(settings.model.name, settings.model.serving_alias)

        return LoadedModel(
            pipeline=pipeline,
            name=settings.model.name,
            version=version or "unknown",
            stage=settings.model.serving_alias,
            source="registry",
            feature_schema_hash=self._schema_hash(),
            decision_threshold=self._decision_threshold(),
            loaded_at=time.time(),
        )

    def _load_from_local(self) -> LoadedModel | None:
        import joblib

        settings = get_settings()
        path = self._fallback_path(settings)
        if not path.is_file():
            log.info("local_fallback_absent", path=str(path))
            return None

        pipeline = joblib.load(path)
        return LoadedModel(
            pipeline=pipeline,
            name=settings.model.name,
            version=f"local:{path.stat().st_mtime_ns}",
            stage="local_fallback",
            source="local_fallback",
            feature_schema_hash=self._schema_hash(),
            decision_threshold=self._decision_threshold(),
            loaded_at=time.time(),
        )

    @staticmethod
    def _fallback_path(settings: Any) -> Path:
        configured = settings.model.local_fallback
        if configured:
            path = Path(str(configured))
            return path if path.is_absolute() else Path(settings.paths.models.parent) / path
        return Path(settings.paths.models) / "champion" / "model.joblib"

    @staticmethod
    def _training_summary() -> dict[str, Any]:
        """The model's serving contract: threshold and feature schema hash.

        **Read from a sidecar beside the artifact first, and only then from
        reports/.** The order is the whole fix.

        ``reports/training_summary.json`` is a build-time report that does not
        travel with the model. The container mounts ``models/`` and nothing
        else, so inside it that file does not exist — and the API fell back to
        the config placeholder of 0.5 while the model had been tuned to 0.1011.

        Nothing failed. The service returned 200 with a plausible probability
        and ``flagged: false`` for every patient, because scores cluster near
        0.1 and almost nothing clears 0.5. A screening model that never flags
        anyone, reporting itself perfectly healthy. It was found by making one
        real request to the container, and by nothing else — not by CI, not by
        the unit tests, not by the warning this code already logged.

        The structural answer is that an artifact must carry its own contract.
        ``registry.save_local_fallback`` now writes ``metadata.json`` next to
        ``model.joblib``, so wherever the model goes — a mount, an HF download,
        a volume — its threshold goes with it.
        """
        import json

        from mlservice.config import get_settings

        settings = get_settings()
        candidates = [
            # Beside the artifact. Travels with the model, so this is the one
            # that is right in a container.
            ModelStore._fallback_path(settings).parent / "metadata.json",
            # The build-time report. Correct on a dev machine, absent in a
            # container — kept so local workflows are unaffected.
            #
            # Resolved through settings rather than PROJECT_ROOT: hardcoding the
            # repository root made this unreachable by configuration and, more
            # to the point, untestable — the first version of the regression
            # test below silently read THIS repository's real summary and passed
            # a case that should have failed.
            settings.paths.reports / "training_summary.json",
        ]

        for path in candidates:
            if not path.is_file():
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(loaded, dict) and loaded:
                return loaded
        return {}

    @classmethod
    def _decision_threshold(cls) -> float:
        """The operating threshold, taken from the TRAINED MODEL, not config.

        This is not an environment setting. Phase 2 selects it on validation for
        a stated recall target, and it is a property of the fitted model in the
        same way its coefficients are — pairing a model with someone else's
        threshold produces a service that is confidently wrong.

        The bug this prevents was real: ``configs/base.yaml`` still carried the
        0.5 placeholder while the trained threshold was 0.1011, so the API
        returned ``flagged: false`` for a patient the model scored *above* its
        own operating point. Nothing raised, because both values were
        individually valid.

        Config remains the fallback for a deployment with no training summary
        (the HF Space builds one in), and a disagreement is logged rather than
        silently resolved.
        """
        settings = get_settings()
        configured = settings.model.decision_threshold
        trained = cls._training_summary().get("champion_threshold")

        if trained is None:
            log.warning(
                "decision_threshold_from_config",
                threshold=configured,
                reason="no training summary found",
                consequence="verify this matches the threshold the model was tuned for",
            )
            return float(configured)

        trained = float(trained)
        if abs(trained - configured) > 1e-9:
            log.warning(
                "decision_threshold_overrides_config",
                trained=trained,
                configured=configured,
                resolution="using the trained value",
                note="the threshold belongs to the model, not the environment",
            )
        return trained

    @classmethod
    def _schema_hash(cls) -> str:
        """Hash of the training feature contract.

        Recorded per prediction so Phase 6 can tell whether two windows are even
        comparable. Falls back to "unknown" rather than failing the load — a
        missing hash degrades drift analysis, but refusing to serve over it would
        be the worse trade.
        """
        return str(cls._training_summary().get("feature_schema_hash", "unknown"))

    @staticmethod
    def _canary_inference(model: LoadedModel) -> None:
        """Score one synthetic record to prove the model actually works.

        Uses the schema example rather than a stored patient record: a canary
        that needs real data cannot run in a fresh container, and this must work
        before any traffic arrives.
        """
        from mlservice.api.schemas import EXAMPLE_FEATURES, PatientFeatures

        probability = model.predict_proba(PatientFeatures(**EXAMPLE_FEATURES).to_model_row())
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"canary inference returned {probability}, expected a probability")
        log.info("canary_inference_ok", probability=round(probability, 6))


#: One store per process. The API reads it through a dependency so tests can
#: substitute a stub without patching import-time state.
store = ModelStore()


def get_store() -> ModelStore:
    return store


__all__ = [
    "LoadedModel",
    "ModelNotLoadedError",
    "ModelStore",
    "get_store",
    "store",
]
