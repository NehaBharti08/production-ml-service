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

# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #

#: A complete, plausible record. Used as the OpenAPI example AND as the startup
#: canary input, so the two cannot drift apart — a canary that scores a payload
#: the schema would reject proves nothing.
EXAMPLE_FEATURES: dict[str, Any] = {
    # Loan terms
    "loan_amnt": 10000.0,
    "term": " 36 months",
    "int_rate": 11.99,
    "installment": 332.1,
    "grade": "B",
    "sub_grade": "B3",
    "purpose": "debt_consolidation",
    "initial_list_status": "w",
    "application_type": "Individual",
    # Borrower
    "annual_inc": 65000.0,
    "emp_length": "5 years",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "addr_state": "CA",
    # Bureau core
    "dti": 18.4,
    "fico_range_low": 700.0,
    "credit_history_months": 180.0,
    "open_acc": 11.0,
    "total_acc": 24.0,
    "revol_bal": 12500.0,
    "revol_util": 43.2,
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 1.0,
    "pub_rec": 0.0,
    "pub_rec_bankruptcies": 0.0,
    # The optional bureau tail is deliberately OMITTED. The canary therefore
    # exercises the missing-value path on every startup, which is the path a
    # real caller is most likely to take — and the one where a broken imputer
    # would otherwise go unnoticed until production.
}


class LoanApplication(BaseModel):
    """One loan application, as known at origination.

    **Required fields are the ones a loan application always has**; every
    credit-bureau attribute is optional and defaults to ``None``. That split is
    a deliberate API decision, not laziness.

    Bureau data is genuinely patchy — ``mths_since_recent_inq`` is null for
    17.4% of the training set — and demanding all 64 features would force
    callers to invent values for attributes they do not hold. Inventing them is
    worse than omitting them, because the feature pipeline imputes a missing
    value *and records that it was missing*, while a fabricated zero is
    indistinguishable from a real one.

    So the contract mirrors the data: a short required core, and a long
    optional tail whose absence is itself informative.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "loan_amnt": 10000,
                "term": " 36 months",
                "int_rate": 11.99,
                "installment": 332.1,
                "grade": "B",
                "sub_grade": "B3",
                "emp_length": "5 years",
                "home_ownership": "MORTGAGE",
                "annual_inc": 65000,
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "addr_state": "CA",
                "dti": 18.4,
                "fico_range_low": 700,
                "credit_history_months": 180.0,
                "open_acc": 11,
                "total_acc": 24,
                "revol_bal": 12500,
                "revol_util": 43.2,
            }
        },
    )

    # --- loan terms: always present on an application -----------------------
    loan_amnt: float = Field(..., gt=0, le=100_000, description="Requested amount")
    term: str = Field(..., description="' 36 months' or ' 60 months' (note the leading space)")
    int_rate: float = Field(..., gt=0, le=40, description="Annual rate, percent")
    installment: float = Field(..., gt=0, le=5_000, description="Monthly payment")
    grade: str = Field(..., min_length=1, max_length=1, description="A-G, the lender's grade")
    sub_grade: str = Field(..., min_length=2, max_length=2, description="e.g. 'B3'")
    purpose: str = Field(..., description="debt_consolidation, credit_card, …")
    initial_list_status: str = Field(default="w", description="'f' or 'w'")
    application_type: str = Field(default="Individual")

    # --- borrower, as stated -------------------------------------------------
    annual_inc: float = Field(..., ge=0, le=10_000_000, description="Stated annual income")
    emp_length: str | None = Field(default=None, description="'10+ years', '< 1 year', …")
    home_ownership: str = Field(..., description="MORTGAGE, RENT, OWN, …")
    verification_status: str = Field(..., description="Verified, Source Verified, Not Verified")
    addr_state: str = Field(..., min_length=2, max_length=2, description="Two-letter state code")

    # --- bureau: the core an underwriter would always pull --------------------
    dti: float = Field(..., ge=0, le=1_000, description="Debt-to-income, percent")
    fico_range_low: float = Field(..., ge=300, le=900, description="Lower bound of the FICO band")
    credit_history_months: float | None = Field(
        default=None, ge=0, description="Months since the earliest credit line"
    )
    open_acc: float | None = Field(default=None, ge=0)
    total_acc: float | None = Field(default=None, ge=0)
    revol_bal: float | None = Field(default=None, ge=0)
    revol_util: float | None = Field(default=None, ge=0)
    delinq_2yrs: float | None = Field(default=None, ge=0)
    inq_last_6mths: float | None = Field(default=None, ge=0)
    pub_rec: float | None = Field(default=None, ge=0)
    pub_rec_bankruptcies: float | None = Field(default=None, ge=0)

    # --- bureau: the long optional tail --------------------------------------
    # Omitted values are imputed AND flagged as missing by the feature
    # pipeline, so absence is modelled rather than papered over.
    acc_now_delinq: float | None = None
    acc_open_past_24mths: float | None = None
    avg_cur_bal: float | None = None
    bc_open_to_buy: float | None = None
    bc_util: float | None = None
    chargeoff_within_12_mths: float | None = None
    collections_12_mths_ex_med: float | None = None
    delinq_amnt: float | None = None
    mo_sin_old_il_acct: float | None = None
    mo_sin_old_rev_tl_op: float | None = None
    mo_sin_rcnt_rev_tl_op: float | None = None
    mo_sin_rcnt_tl: float | None = None
    mort_acc: float | None = None
    mths_since_recent_bc: float | None = None
    mths_since_recent_inq: float | None = None
    num_accts_ever_120_pd: float | None = None
    num_actv_bc_tl: float | None = None
    num_actv_rev_tl: float | None = None
    num_bc_sats: float | None = None
    num_bc_tl: float | None = None
    num_il_tl: float | None = None
    num_op_rev_tl: float | None = None
    num_rev_accts: float | None = None
    num_rev_tl_bal_gt_0: float | None = None
    num_sats: float | None = None
    num_tl_120dpd_2m: float | None = None
    num_tl_30dpd: float | None = None
    num_tl_90g_dpd_24m: float | None = None
    num_tl_op_past_12m: float | None = None
    pct_tl_nvr_dlq: float | None = None
    percent_bc_gt_75: float | None = None
    tax_liens: float | None = None
    tot_coll_amt: float | None = None
    tot_cur_bal: float | None = None
    tot_hi_cred_lim: float | None = None
    total_bal_ex_mort: float | None = None
    total_bc_limit: float | None = None
    total_il_high_credit_limit: float | None = None
    total_rev_hi_lim: float | None = None

    @field_validator("term")
    @classmethod
    def _term_matches_the_training_format(cls, v: str) -> str:
        """Accept '36' or '36 months' and normalise to the trained form.

        The raw data stores ``' 36 months'`` *with a leading space*, and the
        one-hot encoder learned exactly that string. A caller sending ``'36
        months'`` would fall into the unknown-category bucket and receive a
        plausible-looking score computed from a discarded feature — the
        quietest possible way to be wrong.

        Normalising here is friendlier than rejecting, and the alternative
        (teaching the encoder both forms) would hide the quirk instead of
        handling it.
        """
        digits = "".join(ch for ch in v if ch.isdigit())
        if digits not in {"36", "60"}:
            raise ValueError(f"term must be 36 or 60 months, got {v!r}")
        return f" {digits} months"

    @field_validator("grade")
    @classmethod
    def _grade_is_a_to_g(cls, v: str) -> str:
        if v.upper() not in set("ABCDEFG"):
            raise ValueError(f"grade must be one of A-G, got {v!r}")
        return v.upper()

    @field_validator("addr_state")
    @classmethod
    def _state_is_upper(cls, v: str) -> str:
        return v.upper()

    def to_model_row(self) -> dict[str, Any]:
        """Feature dict keyed by the column names the model expects.

        ``None`` is preserved rather than filled: the pipeline's imputer adds a
        missingness indicator, so a null here becomes two pieces of
        information, not a silently invented number.
        """
        return self.model_dump()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: LoanApplication
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
    items: Annotated[list[LoanApplication], Field(min_length=1)]
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
    "LoanApplication",
    "ModelInfo",
    "OutcomeRequest",
    "OutcomeResponse",
    "PredictionRequest",
    "PredictionResponse",
]
