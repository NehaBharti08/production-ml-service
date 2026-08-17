"""Prediction endpoints.

Every prediction writes one prediction-log record and returns its
``prediction_id``. That ID is the join key for a later observed outcome, and it
cannot be reconstructed afterwards — which is why it is returned rather than
kept internal.

Latency is measured twice: total, and inference only. When p99 rises, the first
question is whether the model got slower or the layer around it did, and a single
number cannot answer it.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from mlservice.api import metrics
from mlservice.api.model_loader import ModelStore, get_store
from mlservice.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from mlservice.config import get_settings
from mlservice.logging_ import get_logger, get_request_id
from mlservice.monitoring import prediction_log

log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["prediction"])

#: One writer per process, so the append lock is shared across requests.
_writer: prediction_log.PredictionLogWriter | None = None


def get_writer() -> prediction_log.PredictionLogWriter:
    global _writer
    if _writer is None:
        _writer = prediction_log.PredictionLogWriter()
    return _writer


def _model_info(model: object) -> ModelInfo:
    return ModelInfo(
        name=model.name,  # type: ignore[attr-defined]
        version=model.version,  # type: ignore[attr-defined]
        stage=model.stage,  # type: ignore[attr-defined]
        source=model.source,  # type: ignore[attr-defined]
        feature_schema_hash=model.feature_schema_hash,  # type: ignore[attr-defined]
        decision_threshold=model.decision_threshold,  # type: ignore[attr-defined]
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score one encounter",
    responses={
        422: {"description": "Validation failed; the response names each field."},
        503: {"description": "No model loaded."},
    },
)
async def predict(
    payload: PredictionRequest,
    store: ModelStore = Depends(get_store),
) -> PredictionResponse:
    settings = get_settings()
    started = time.perf_counter()

    model = store.model  # raises ModelNotLoadedError -> 503
    row = payload.features.to_model_row()

    inference_started = time.perf_counter()
    probability = model.predict_proba(row)
    inference_s = time.perf_counter() - inference_started

    total_s = time.perf_counter() - started
    label = int(probability >= model.decision_threshold)

    record = prediction_log.build_record(
        request_id=get_request_id() or "unknown",
        features_raw=row,
        predicted_proba=probability,
        decision_threshold=model.decision_threshold,
        model_name=model.name,
        model_version=model.version,
        model_stage=model.stage,
        model_source=model.source,
        feature_schema_hash=model.feature_schema_hash,
        latency_ms_total=total_s * 1000,
        latency_ms_inference=inference_s * 1000,
        api_version=settings.api.version,
        client_id=payload.client_id,
    )
    metrics.record_log_write(get_writer().write(record))
    metrics.record_prediction(model.version, probability, label, inference_s)

    # Note what is NOT logged here: no feature values. They are in the
    # prediction log, which is access-controlled as a data store; an aggregated
    # application log is not the place for a patient record.
    log.info(
        "prediction_served",
        prediction_id=record.prediction_id,
        model_version=model.version,
        probability=round(probability, 6),
        flagged=bool(label),
        latency_ms=round(total_s * 1000, 3),
    )

    return PredictionResponse(
        prediction_id=record.prediction_id,
        request_id=record.request_id,
        readmission_probability=round(probability, 6),
        flagged=bool(label),
        decision_threshold=model.decision_threshold,
        model=_model_info(model),
        latency_ms=round(total_s * 1000, 3),
        disclaimer=settings.api.disclaimer,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score many encounters in one call",
    responses={
        413: {"description": "Batch exceeds the configured maximum."},
        422: {"description": "Validation failed; the response names each field."},
        503: {"description": "No model loaded."},
    },
)
async def predict_batch(
    payload: BatchPredictionRequest,
    store: ModelStore = Depends(get_store),
) -> BatchPredictionResponse:
    settings = get_settings()
    started = time.perf_counter()

    # Checked here rather than as a Pydantic constraint because the limit is
    # configurable per environment (100 on the public HF Space, 500 locally) and
    # a schema bound would be baked in at import time.
    if len(payload.items) > settings.api.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"batch of {len(payload.items)} exceeds the maximum of "
                f"{settings.api.max_batch_size}. Split the request — an "
                "unbounded batch would blow the latency SLO for every "
                "concurrent caller."
            ),
        )

    model = store.model
    rows = [item.to_model_row() for item in payload.items]

    inference_started = time.perf_counter()
    probabilities = model.predict_proba_batch(rows)
    inference_s = time.perf_counter() - inference_started

    # Attributed evenly across items. The true per-item cost is not separable in
    # a vectorised call, and pretending otherwise would put a fabricated number
    # in the log.
    per_item_inference_ms = (inference_s * 1000) / len(rows)
    total_s = time.perf_counter() - started
    per_item_total_ms = (total_s * 1000) / len(rows)

    batch_id = str(uuid.uuid4())
    responses: list[PredictionResponse] = []
    records: list[prediction_log.PredictionRecord] = []

    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        label = int(probability >= model.decision_threshold)
        record = prediction_log.build_record(
            request_id=get_request_id() or "unknown",
            features_raw=row,
            predicted_proba=probability,
            decision_threshold=model.decision_threshold,
            model_name=model.name,
            model_version=model.version,
            model_stage=model.stage,
            model_source=model.source,
            feature_schema_hash=model.feature_schema_hash,
            latency_ms_total=per_item_total_ms,
            latency_ms_inference=per_item_inference_ms,
            api_version=settings.api.version,
            client_id=payload.client_id,
            batch_id=batch_id,
            batch_index=index,
        )
        records.append(record)
        metrics.record_prediction(model.version, probability, label, inference_s / len(rows))

        responses.append(
            PredictionResponse(
                prediction_id=record.prediction_id,
                request_id=record.request_id,
                readmission_probability=round(probability, 6),
                flagged=bool(label),
                decision_threshold=model.decision_threshold,
                model=_model_info(model),
                latency_ms=round(per_item_total_ms, 3),
                disclaimer=settings.api.disclaimer,
            )
        )

    # One fsync for the whole batch: measured at ~400 ms of a 490 ms 200-item
    # request when done per record. read_records tolerates a truncated tail.
    written = get_writer().write_many(records)
    for _ in range(len(records)):
        metrics.record_log_write(written > 0)

    log.info(
        "batch_prediction_served",
        batch_id=batch_id,
        count=len(rows),
        model_version=model.version,
        flagged_count=sum(r.flagged for r in responses),
        records_written=written,
        latency_ms=round(total_s * 1000, 3),
    )

    return BatchPredictionResponse(
        batch_id=batch_id,
        request_id=get_request_id() or "unknown",
        count=len(responses),
        predictions=responses,
        latency_ms=round(total_s * 1000, 3),
        disclaimer=settings.api.disclaimer,
    )


__all__ = ["get_writer", "router"]
