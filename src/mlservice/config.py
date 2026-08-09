"""Layered configuration for the ML service.

Resolution order, lowest precedence first:

    1. Field defaults declared below
    2. ``configs/base.yaml``
    3. ``configs/{MLSERVICE_ENV}.yaml``   (local | hf | test)
    4. Environment variables, ``MLSERVICE_`` prefixed, ``__`` for nesting
    5. ``.env`` file, same naming

Nested fields use a double underscore, so ``api.port`` is overridden by
``MLSERVICE_API__PORT=9000``.

Two rules this module exists to enforce:

*   **Fail fast.** A misconfigured service must refuse to start rather than
    serve predictions with, say, the wrong decision threshold. Every field is
    typed and validated at import of :func:`get_settings`, not at first use.
*   **One source of truth for operational numbers.** Thresholds live in
    ``configs/thresholds.yaml`` and are loaded by :func:`get_thresholds`. They
    are kept separate from application settings because Phase 6 *regenerates*
    them from data (empirical null calibration) and they need to be diffable in
    isolation when that happens.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #

# config.py -> mlservice -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

Environment = Literal["local", "hf", "test"]


def _current_env() -> Environment:
    raw = os.environ.get("MLSERVICE_ENV", "local").strip().lower()
    if raw not in ("local", "hf", "test"):
        raise ValueError(
            f"MLSERVICE_ENV must be one of 'local', 'hf', 'test' — got {raw!r}. "
            "Refusing to start rather than guess which environment this is."
        )
    return raw  # type: ignore[return-value]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level, got {type(loaded)}")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """Feeds ``base.yaml`` deep-merged with the environment overlay into pydantic."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ARG002
        # Unused: we supply the whole mapping in __call__ instead of per-field.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        base = _read_yaml(CONFIG_DIR / "base.yaml")
        overlay = _read_yaml(CONFIG_DIR / f"{_current_env()}.yaml")
        return _deep_merge(base, overlay)


# --------------------------------------------------------------------------- #
# Settings sections
# --------------------------------------------------------------------------- #


class _Section(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` matters more than it looks. A mistyped key — ``prot``
    for ``port``, ``treshold`` for ``threshold`` — would otherwise be silently
    discarded and the service would run on defaults while the operator believed
    their override had taken effect. That is exactly the class of failure that
    shows up at 2am as "but I *set* that".

    This covers both YAML keys and *nested* environment variables: the env
    source assembles ``MLSERVICE_API__*`` into a mapping for ``api``, so an
    unknown subkey reaches validation and is rejected here.

    One gap remains, and it is structural. A **root-level** env typo such as
    ``MLSERVICE_DEBUGG`` maps to no field at all, so pydantic-settings never
    reads it and validation never sees it — the override silently does nothing.
    :func:`stray_env_vars` covers that case, reported by ``mlservice doctor``.
    """

    model_config = ConfigDict(extra="forbid")


class PathsSettings(_Section):
    """All filesystem locations, resolved absolute against the project root.

    Relative values in YAML are resolved here so that the service behaves
    identically whether it is launched from the repo root, from a container
    WORKDIR, or from a pytest rootdir.
    """

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    data_reference: Path = Path("data/reference")
    data_monitoring: Path = Path("data/monitoring")
    models: Path = Path("models")
    reports: Path = Path("reports")
    logs: Path = Path("logs")

    @model_validator(mode="after")
    def _resolve(self) -> PathsSettings:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, (PROJECT_ROOT / value).resolve())
        return self


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    #: JSON in containers and CI; console renderer is far easier to read locally.
    format: Literal["json", "console"] = "json"
    #: Header carrying an inbound correlation ID. Honoured if present, minted if not.
    request_id_header: str = "X-Request-ID"
    #: Never log these, at any level. Extended as the schema grows.
    redact_fields: list[str] = Field(default_factory=lambda: ["patient_nbr", "encounter_id"])


class DataSettings(_Section):
    """Dataset identity and the split policy.

    ``time_proxy_column`` is named as a *proxy* on purpose. The Diabetes 130
    dataset has no timestamp; Phase 1 must verify that ordering by this column
    actually carries time signal before any split built on it is trusted. See
    docs/DECISIONS/0004-temporal-split-proxy.md.
    """

    dataset_name: str = "diabetes_130_us_hospitals"
    source_url: str = ""
    target_column: str = "readmitted"
    #: Binary task: readmitted within 30 days vs everything else.
    positive_label: str = "<30"
    time_proxy_column: str = "encounter_id"
    patient_id_column: str = "patient_nbr"
    #: Chronological fractions; must sum to 1.0.
    train_fraction: float = Field(default=0.60, gt=0, lt=1)
    val_fraction: float = Field(default=0.20, gt=0, lt=1)
    test_fraction: float = Field(default=0.20, gt=0, lt=1)
    #: Keep only each patient's first encounter, preventing the same patient
    #: from appearing on both sides of the split (Strack et al. protocol).
    first_encounter_only: bool = True
    #: discharge_disposition_id values meaning expired or hospice. A patient who
    #: died cannot be readmitted, so these rows have a deterministic label.
    #: Confirmed against the dataset's IDS_mapping in Phase 1 before use.
    expired_discharge_ids: list[int] = Field(default_factory=lambda: [11, 13, 14, 19, 20, 21])
    random_seed: int = 42

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> DataSettings:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train/val/test fractions must sum to 1.0, got {total:.6f}. "
                "A silent renormalisation here would quietly change every "
                "downstream metric."
            )
        return self


class ModelSettings(_Section):
    name: str = "readmission-risk"
    #: Registry alias the API loads. Flipping this alias is the rollback lever.
    serving_alias: str = "champion"
    #: Alias a challenger occupies while it is being evaluated.
    challenger_alias: str = "challenger"
    #: Fallback artifact baked into the image so the API can serve even when
    #: MLflow is unreachable. Availability should not depend on the registry.
    local_fallback: Path | None = None
    #: Placeholder. Phase 2 selects this on validation for a stated recall
    #: target; 0.5 is a default, never a decision.
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MLflowSettings(_Section):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "readmission-risk"
    registry_uri: str | None = None

    @model_validator(mode="after")
    def _default_registry_to_tracking(self) -> MLflowSettings:
        if self.registry_uri is None:
            object.__setattr__(self, "registry_uri", self.tracking_uri)
        return self


class ApiSettings(_Section):
    title: str = "Hospital Readmission Risk API"
    version: str = "v1"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    #: Seconds to let in-flight requests finish on SIGTERM before forcing exit.
    #: Must stay below the Kubernetes terminationGracePeriodSeconds.
    graceful_shutdown_seconds: int = Field(default=20, ge=1)
    max_batch_size: int = Field(default=500, ge=1)
    cors_origins: list[str] = Field(default_factory=list)
    #: Surfaced verbatim in every prediction response, /v1/model, and the UI.
    #: Non-optional by design: this is a health-adjacent service.
    disclaimer: str = (
        "NOT FOR CLINICAL USE. This is an engineering demonstration of ML "
        "operations, trained on a public 1999-2008 research dataset. It has "
        "not been clinically validated, is not a medical device, and must "
        "never inform patient care."
    )

    @field_validator("disclaimer")
    @classmethod
    def _disclaimer_is_present(cls, v: str) -> str:
        if len(v.strip()) < 40:
            raise ValueError(
                "The non-clinical disclaimer may not be blanked or trimmed to a "
                "token string. It is a requirement of this project, not a label."
            )
        return v


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLSERVICE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",  # an unknown key is a typo; typos in config cause outages
        validate_default=True,
    )

    env: Environment = "local"
    debug: bool = False

    paths: PathsSettings = Field(default_factory=PathsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins, so this reads highest-precedence first.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSettingsSource(settings_cls),
            file_secret_settings,
        )


def _known_env_var_names() -> set[str]:
    """Every ``MLSERVICE_*`` name that maps to a real field, one level deep."""
    prefix = "MLSERVICE_"
    names = {f"{prefix}ENV"}
    for field_name, field in Settings.model_fields.items():
        names.add(f"{prefix}{field_name.upper()}")
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for sub in annotation.model_fields:
                names.add(f"{prefix}{field_name.upper()}__{sub.upper()}")
    return names


def stray_env_vars() -> list[str]:
    """Return ``MLSERVICE_*`` environment variables that match no known field.

    This closes the one gap left by ``extra="forbid"``: a **root-level** name
    that matches no field, such as ``MLSERVICE_DEBUGG``, is never read by
    pydantic-settings at all. No error is raised, the override silently does
    nothing, and the operator is left believing they configured something they
    did not. (Nested typos like ``MLSERVICE_API__PROT`` *are* caught by
    validation — see :class:`_Section`.)

    Reported by ``mlservice doctor`` as a warning rather than an exception: an
    unrecognised variable is usually a typo, but it can legitimately belong to
    another tool, so refusing to boot over one would be too aggressive.
    """
    known = _known_env_var_names()
    return sorted(
        name for name in os.environ if name.startswith("MLSERVICE_") and name.upper() not in known
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated singleton settings object.

    Cached so that config is parsed once per process and every component sees
    identical values. Call ``get_settings.cache_clear()`` in tests that need to
    re-read the environment.
    """
    return Settings(env=_current_env())


# --------------------------------------------------------------------------- #
# Operational thresholds — deliberately separate
# --------------------------------------------------------------------------- #


class Thresholds(BaseModel):
    """SLOs, drift thresholds and promotion gates.

    Kept out of :class:`Settings` because these values are *derived*, not
    chosen: latency numbers come from the Phase 4 load test, and per-feature
    drift thresholds are regenerated from data by the Phase 6 empirical-null
    calibration. Isolating them keeps that regeneration a small, reviewable
    diff instead of a churn across the whole config.

    ``extra="allow"`` because the schema fills in across phases; each phase
    tightens this into typed sub-models as its section becomes real.
    """

    model_config = {"extra": "allow"}


@lru_cache(maxsize=1)
def get_thresholds() -> Thresholds:
    path = CONFIG_DIR / "thresholds.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Operational thresholds are a required, "
            "version-controlled artifact — the service will not run without them."
        )
    return Thresholds.model_validate(_read_yaml(path))


__all__ = [
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "Settings",
    "Thresholds",
    "get_settings",
    "get_thresholds",
    "stray_env_vars",
]
