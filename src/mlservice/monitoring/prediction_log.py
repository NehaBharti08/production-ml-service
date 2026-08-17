"""The prediction log — the raw material for every monitoring feature later.

This schema is the most consequential design decision in Phase 3. Phases 5, 6
and 7 all read it:

*   Phase 5 computes latency percentiles and prediction-distribution metrics.
*   Phase 6 compares feature windows against the frozen reference to detect
    drift, and joins late-arriving outcomes to measure real performance.
*   Phase 7 decides whether to retrain, and whether a challenger may be
    promoted, from what is recorded here.

So it is designed up front rather than evolved. Three properties earn their
place:

**Nullable outcome columns exist from day one.** A readmission label cannot
exist until 30 days after discharge. If the columns were added later, every
record written before the migration would be unjoinable, and the first months of
production data — exactly the baseline you want — would be useless.

**``feature_schema_hash`` is stored per record.** Drift analysis across a
feature-schema change is meaningless: the distributions are not comparable.
Without the hash the change is invisible and the drift report silently lies.

**Raw features are stored, not transformed ones.** Drift must be measured in the
space a human can reason about (``age="[70-80)"``), not in one-hot columns whose
meaning depends on an encoder version. It also means a stored record can be
replayed through a *different* model, which is what the Phase 7 shadow/canary
comparison needs.

Storage is newline-delimited JSON, one record per line. Deliberately boring:
append-only, crash-safe at line granularity, greppable during an incident, and
readable by pandas without a schema registry. Parquet would compress better but
cannot be appended to a line at a time, and a monitoring substrate that can lose
the last buffer on a crash is not one you can reason about.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlservice.config import get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1


def new_prediction_id() -> str:
    """Mint a prediction ID.

    UUIDv7 would be preferable — it sorts by creation time, which makes
    time-window queries over the log cheap. Python's stdlib only gained
    ``uuid7`` in 3.14 and this project pins 3.11, so a v4 plus the timestamp
    column does the same job with an extra sort key.
    """
    return str(uuid.uuid4())


@dataclass
class PredictionRecord:
    """One prediction, and everything later phases need to interpret it."""

    # --- identity ---
    prediction_id: str
    request_id: str
    timestamp_utc: str

    # --- model provenance: which model produced this, exactly ---
    model_name: str
    model_version: str
    model_stage: str
    model_source: str  # "registry" | "local_fallback" — matters during an incident
    feature_schema_hash: str

    # --- input, in the space drift is measured in ---
    features_raw: dict[str, Any]

    # --- output ---
    predicted_proba: float
    predicted_label: int
    decision_threshold: float

    # --- performance ---
    latency_ms_total: float
    latency_ms_inference: float

    # --- request context ---
    api_version: str
    client_id: str | None = None
    batch_id: str | None = None  # groups records from one batch call
    batch_index: int | None = None

    # --- outcome, joined later (see module docstring) ---
    outcome_label: int | None = None
    outcome_timestamp: str | None = None
    outcome_source: str | None = None

    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), default=str)


class PredictionLogWriter:
    """Append-only NDJSON writer, safe for concurrent requests.

    A lock rather than a queue: at this service's throughput the write is far
    cheaper than the model call it follows, and a background flush thread would
    add a way to lose records on shutdown for no measurable gain.

    Failures are logged and swallowed. A monitoring write must never turn a
    successful prediction into a 500 — losing one log line is a monitoring gap,
    while failing the request is an outage.
    """

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or (settings.paths.data_monitoring / "predictions.ndjson")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._written = 0
        self._failed = 0

    def write(self, record: PredictionRecord) -> bool:
        line = record.to_json_line()
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                # Flush and fsync so a container kill cannot lose the tail.
                # Without this the last records sit in the OS buffer, and the
                # records around a crash are precisely the ones worth having.
                fh.flush()
                os.fsync(fh.fileno())
            self._written += 1
            return True
        except OSError as exc:
            self._failed += 1
            log.error(
                "prediction_log_write_failed",
                error=str(exc),
                path=str(self.path),
                prediction_id=record.prediction_id,
                consequence="prediction served but not recorded — monitoring gap",
            )
            return False

    def write_many(self, records: list[PredictionRecord]) -> int:
        """Append many records with a single fsync. Returns the count written.

        Measured, not assumed: a 200-item batch spent ~400 ms of its ~490 ms in
        200 separate fsync calls, while the model transform cost ~85 ms total.
        Per-record durability was the dominant cost of batch prediction.

        Crash semantics are unchanged in any way that matters. fsync per record
        guarantees "every record before the crash is durable"; fsync per batch
        guarantees "every record before the last batch is durable, and the final
        batch may be truncated". :func:`read_records` already tolerates a
        truncated tail — it is the expected state after an unclean shutdown — so
        this trades a bounded, already-handled loss for a large latency win.
        """
        if not records:
            return 0

        lines = "".join(record.to_json_line() + "\n" for record in records)
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(lines)
                fh.flush()
                os.fsync(fh.fileno())  # once, not per record
            self._written += len(records)
            return len(records)
        except OSError as exc:
            self._failed += len(records)
            log.error(
                "prediction_log_batch_write_failed",
                error=str(exc),
                path=str(self.path),
                record_count=len(records),
                consequence="predictions served but not recorded — monitoring gap",
            )
            return 0

    @property
    def stats(self) -> dict[str, int]:
        return {"written": self._written, "failed": self._failed}


def build_record(
    *,
    request_id: str,
    features_raw: dict[str, Any],
    predicted_proba: float,
    decision_threshold: float,
    model_name: str,
    model_version: str,
    model_stage: str,
    model_source: str,
    feature_schema_hash: str,
    latency_ms_total: float,
    latency_ms_inference: float,
    api_version: str,
    client_id: str | None = None,
    batch_id: str | None = None,
    batch_index: int | None = None,
) -> PredictionRecord:
    """Construct a record with a fresh ID and timestamp."""
    return PredictionRecord(
        prediction_id=new_prediction_id(),
        request_id=request_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage,
        model_source=model_source,
        feature_schema_hash=feature_schema_hash,
        features_raw=features_raw,
        predicted_proba=round(float(predicted_proba), 6),
        predicted_label=int(predicted_proba >= decision_threshold),
        decision_threshold=round(float(decision_threshold), 6),
        latency_ms_total=round(float(latency_ms_total), 3),
        latency_ms_inference=round(float(latency_ms_inference), 3),
        api_version=api_version,
        client_id=client_id,
        batch_id=batch_id,
        batch_index=batch_index,
    )


# --------------------------------------------------------------------------- #
# Reading — used by the monitoring job and the outcome join
# --------------------------------------------------------------------------- #


def read_records(path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Read the log, skipping malformed lines rather than failing.

    A truncated final line is expected after an unclean shutdown. Refusing to
    read the whole log because of it would take monitoring down for a reason
    that does not matter — but the count is logged, because a *rising* count is
    a real signal.
    """
    settings = get_settings()
    target = path or (settings.paths.data_monitoring / "predictions.ndjson")
    if not target.is_file():
        return []

    records: list[dict[str, Any]] = []
    malformed = 0
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                malformed += 1
            if limit and len(records) >= limit:
                break

    if malformed:
        log.warning("prediction_log_malformed_lines", count=malformed, path=str(target))
    return records


def append_outcome(
    prediction_id: str,
    outcome_label: int,
    source: str = "manual",
    path: Path | None = None,
) -> bool:
    """Record a late-arriving outcome as a separate NDJSON entry.

    Appended rather than rewritten in place. Mutating a prediction record would
    mean rewriting the file, which destroys the append-only guarantee, risks the
    whole log on a crash, and loses the fact that the label arrived later than
    the prediction — which is itself information Phase 6 needs.

    The join happens at read time, keyed on ``prediction_id``.
    """
    settings = get_settings()
    target = path or (settings.paths.data_monitoring / "outcomes.ndjson")
    target.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "prediction_id": prediction_id,
        "outcome_label": int(outcome_label),
        "outcome_timestamp": datetime.now(UTC).isoformat(),
        "outcome_source": source,
        "schema_version": SCHEMA_VERSION,
    }
    try:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True
    except OSError as exc:
        log.error("outcome_write_failed", error=str(exc), prediction_id=prediction_id)
        return False


def join_outcomes(
    predictions_path: Path | None = None, outcomes_path: Path | None = None
) -> list[dict[str, Any]]:
    """Join predictions with any matured outcomes, on ``prediction_id``.

    Last-write-wins on duplicate outcomes for the same prediction: a corrected
    label should supersede an earlier one rather than being ignored.
    """
    settings = get_settings()
    outcomes_target = outcomes_path or (settings.paths.data_monitoring / "outcomes.ndjson")

    outcomes: dict[str, dict[str, Any]] = {}
    if outcomes_target.is_file():
        for entry in read_records(outcomes_target):
            outcomes[entry["prediction_id"]] = entry

    joined = []
    for record in read_records(predictions_path):
        outcome = outcomes.get(record["prediction_id"])
        if outcome:
            record = {
                **record,
                "outcome_label": outcome["outcome_label"],
                "outcome_timestamp": outcome["outcome_timestamp"],
                "outcome_source": outcome["outcome_source"],
            }
        joined.append(record)

    matured = sum(1 for r in joined if r.get("outcome_label") is not None)
    log.info("outcomes_joined", predictions=len(joined), matured=matured)
    return joined


__all__ = [
    "SCHEMA_VERSION",
    "PredictionLogWriter",
    "PredictionRecord",
    "append_outcome",
    "build_record",
    "join_outcomes",
    "new_prediction_id",
    "read_records",
]
