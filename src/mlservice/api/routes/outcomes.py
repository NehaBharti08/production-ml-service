"""Late-arriving outcome ingestion.

The endpoint that makes delayed-label monitoring possible. A 30-day readmission
label cannot exist at prediction time, so it arrives here later and is joined on
``prediction_id``.

Outcomes are **appended**, never written back into the prediction record.
Mutating the original would mean rewriting the log file, which destroys the
append-only guarantee and loses the fact that the label arrived later — itself
information Phase 6 needs to reason about maturation lag.
"""

from __future__ import annotations

from fastapi import APIRouter

from mlservice.api import metrics
from mlservice.api.schemas import OutcomeRequest, OutcomeResponse
from mlservice.logging_ import get_logger
from mlservice.monitoring import prediction_log

log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["outcomes"])


@router.post(
    "/outcomes",
    response_model=OutcomeResponse,
    summary="Record an observed outcome against a prediction",
)
async def record_outcome(payload: OutcomeRequest) -> OutcomeResponse:
    """Append an observed outcome for a previously returned ``prediction_id``.

    The prediction ID is deliberately **not** validated against the log. Doing so
    would require scanning it on every call — O(n) per request against a file
    that grows forever — and would reject legitimate late outcomes after log
    rotation. Unmatched IDs simply fail to join in Phase 6 and are counted there,
    which is the right place to notice a systematic mismatch.
    """
    ok = prediction_log.append_outcome(
        prediction_id=payload.prediction_id,
        outcome_label=int(payload.readmitted_within_30_days),
        source=payload.source,
    )
    if ok:
        metrics.outcomes_recorded_total.inc()

    log.info(
        "outcome_recorded",
        prediction_id=payload.prediction_id,
        outcome=int(payload.readmitted_within_30_days),
        source=payload.source,
        written=ok,
    )

    return OutcomeResponse(
        prediction_id=payload.prediction_id,
        recorded=ok,
        message=(
            "Outcome appended; it will be joined on prediction_id by the monitoring job."
            if ok
            else "Outcome could not be written — see server logs."
        ),
    )
