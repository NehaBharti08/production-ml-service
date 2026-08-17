"""Pydantic v2 request and response models.

Field names and allowed values come from :mod:`mlservice.data.schema`, the same
module the training pipeline imports. That shared source is the point: if the API
declared its own idea of a valid record, the two definitions would drift and the
service would start scoring features the model was never trained on — with both
halves individually valid and nothing raising.

Validation is strict on purpose:

*   ``extra="forbid"`` — an unexpected field is a caller bug (usually a typo or
    a version mismatch) and returning 422 tells them so. Silently ignoring it
    means they believe they sent a feature that was discarded.
*   Numeric fields carry real bounds taken from the dataset, so a
    ``time_in_hospital`` of 500 is rejected rather than scored.
*   Categorical fields are **not** hard-restricted to observed values. The
    encoder handles unseen categories, and rejecting them would turn a
    survivable degradation into an outage the first time a hospital adds a
    specialty.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlservice.data import schema

# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #

#: A complete, plausible record. Used as the OpenAPI example AND as the startup
#: canary input, so the two cannot drift apart — a canary that scores a payload
#: the schema would reject proves nothing.
EXAMPLE_FEATURES: dict[str, Any] = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "diag_1": "Circulatory",
    "diag_2": "Diabetes",
    "diag_3": "Circulatory",
    "number_diagnoses": 9,
    "max_glu_serum": "NotMeasured",
    "A1Cresult": "NotMeasured",
    "metformin": "No",
    "insulin": "Up",
    "change": "Ch",
    "diabetesMed": "Yes",
}


class PatientFeatures(BaseModel):
    """One encounter's features, as they appear before any transformation.

    Raw values, not encoded ones. The fitted transformer travels inside the
    model artifact, so the caller sends ``age="[70-80)"`` and never needs to
    know how it is encoded.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"example": EXAMPLE_FEATURES},
    )

    # --- demographics ---
    race: str = Field(default=schema.UNKNOWN_CATEGORY, description="Caucasian, AfricanAmerican, …")
    gender: str = Field(default=schema.UNKNOWN_CATEGORY, description="Male, Female")
    age: str = Field(..., description="Ten-year band, e.g. '[70-80)'")

    # --- administrative (unordered code sets, kept categorical) ---
    admission_type_id: int = Field(..., ge=1, le=9)
    discharge_disposition_id: int = Field(..., ge=1, le=30)
    admission_source_id: int = Field(..., ge=1, le=26)
    medical_specialty: str = Field(default=schema.UNKNOWN_CATEGORY)

    # --- utilisation ---
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days; 1-14 by dataset criteria")
    num_lab_procedures: int = Field(..., ge=0, le=200)
    num_procedures: int = Field(..., ge=0, le=20)
    num_medications: int = Field(..., ge=0, le=100)
    number_outpatient: int = Field(..., ge=0, le=100)
    number_emergency: int = Field(..., ge=0, le=100)
    number_inpatient: int = Field(
        ..., ge=0, le=100, description="Prior inpatient admissions — strongest single predictor"
    )
    number_diagnoses: int = Field(..., ge=1, le=20)

    # --- diagnoses, as clinical bands ---
    diag_1: str = Field(default=schema.UNKNOWN_CATEGORY, description="Circulatory, Diabetes, …")
    diag_2: str = Field(default=schema.UNKNOWN_CATEGORY)
    diag_3: str = Field(default=schema.UNKNOWN_CATEGORY)

    # --- labs: NotMeasured is meaningful, not missing ---
    max_glu_serum: str = Field(default=schema.NOT_MEASURED_CATEGORY)
    A1Cresult: str = Field(  # dataset column name, must match exactly
        default=schema.NOT_MEASURED_CATEGORY,
        description="'>7', '>8', 'Norm', or 'NotMeasured' if the test was not ordered",
    )

    # --- medications: No / Steady / Up / Down ---
    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    acetohexamide: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    tolbutamide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    troglitazone: str = "No"
    tolazamide: str = "No"
    insulin: str = "No"
    glyburide_metformin: str = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: str = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: str = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: str = Field(default="No", alias="metformin-pioglitazone")

    # --- treatment ---
    change: str = Field(default="No", description="'Ch' if medication changed, else 'No'")
    diabetesMed: str = Field(  # noqa: N815 — must match the dataset column name exactly
        default="No", description="'Yes' if any diabetes medication prescribed"
    )

    @field_validator("age")
    @classmethod
    def _age_looks_like_a_band(cls, v: str) -> str:
        """Reject a bare number early with a message that says what is wrong.

        Sending ``age=75`` is the single most likely caller mistake, and the
        generic "unseen category" path would silently score it as unknown —
        producing a plausible-looking prediction from a discarded feature.
        """
        if not (v.startswith("[") and "-" in v):
            raise ValueError(
                f"age must be a ten-year band like '[70-80)', got {v!r}. "
                "This model was trained on banded ages, not exact ages."
            )
        return v

    def to_model_row(self) -> dict[str, Any]:
        """Feature dict keyed by the column names the model expects.

        ``by_alias=True`` restores the hyphenated medication names
        (``glyburide-metformin``); the Python attributes use underscores because
        a hyphen is not a valid identifier.
        """
        return self.model_dump(by_alias=True)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: PatientFeatures
    client_id: str | None = Field(
        default=None,
        max_length=64,
        description="Caller identifier, recorded in the prediction log.",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Bounded because an unbounded batch is a denial-of-service vector and a
    #: latency-SLO hazard: one 100k-row request would blow the p99 for every
    #: concurrent caller. The limit is configurable per environment.
    items: Annotated[list[PatientFeatures], Field(min_length=1)]
    client_id: str | None = Field(default=None, max_length=64)


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str
    source: str = Field(description="'registry' or 'local_fallback'")
    feature_schema_hash: str
    decision_threshold: float


class PredictionResponse(BaseModel):
    """A single prediction.

    ``prediction_id`` is returned so the caller can later submit the observed
    outcome against it — that join is what makes delayed-label monitoring
    possible at all, and it cannot be reconstructed after the fact.
    """

    prediction_id: str
    request_id: str
    readmission_probability: float = Field(ge=0.0, le=1.0)
    flagged: bool = Field(description="probability >= decision_threshold")
    decision_threshold: float
    model: ModelInfo
    latency_ms: float
    #: Present in every prediction response, not only the docs. A consumer that
    #: only ever sees JSON must still be told this is not a clinical tool.
    disclaimer: str


class BatchPredictionResponse(BaseModel):
    batch_id: str
    request_id: str
    count: int
    predictions: list[PredictionResponse]
    latency_ms: float
    disclaimer: str


class OutcomeRequest(BaseModel):
    """A late-arriving observed outcome, joined on ``prediction_id``."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(..., min_length=8, max_length=64)
    readmitted_within_30_days: bool
    source: str = Field(default="manual", max_length=32)


class OutcomeResponse(BaseModel):
    prediction_id: str
    recorded: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "EXAMPLE_FEATURES",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfo",
    "OutcomeRequest",
    "OutcomeResponse",
    "PatientFeatures",
    "PredictionRequest",
    "PredictionResponse",
]
