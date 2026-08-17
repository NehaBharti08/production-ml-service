"""The prediction log — the substrate Phases 5-7 all read.

Tested carefully because a defect here is silent and unrecoverable: a missing or
malformed record is not detected at write time, and the data it would have
carried cannot be reconstructed later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlservice.monitoring import prediction_log

pytestmark = pytest.mark.unit


def _record(**overrides: object) -> prediction_log.PredictionRecord:
    defaults = {
        "request_id": "req-1",
        "features_raw": {"age": "[70-80)", "number_inpatient": 1},
        "predicted_proba": 0.1234,
        "decision_threshold": 0.1011,
        "model_name": "readmission-risk",
        "model_version": "3",
        "model_stage": "champion",
        "model_source": "registry",
        "feature_schema_hash": "abc123",
        "latency_ms_total": 12.5,
        "latency_ms_inference": 1.2,
        "api_version": "v1",
    }
    defaults.update(overrides)
    return prediction_log.build_record(**defaults)  # type: ignore[arg-type]


class TestRecordSchema:
    def test_label_is_derived_from_the_threshold(self) -> None:
        """The label must follow the threshold, never a hardcoded 0.5.

        This is the bug that shipped briefly in Phase 3: config carried the 0.5
        placeholder while the trained threshold was 0.1011, so a patient scored
        above the model's own operating point was reported as not flagged.
        """
        assert _record(predicted_proba=0.12, decision_threshold=0.1011).predicted_label == 1
        assert _record(predicted_proba=0.09, decision_threshold=0.1011).predicted_label == 0

    def test_outcome_fields_exist_and_start_null(self) -> None:
        """Present from day one so late labels join without a migration.

        Added later, every record written before the change would be
        unjoinable — losing exactly the earliest production data that makes the
        best monitoring baseline.
        """
        record = _record()
        assert record.outcome_label is None
        assert record.outcome_timestamp is None
        assert record.outcome_source is None

    def test_timestamp_is_timezone_aware_utc(self) -> None:
        """A naive timestamp makes cross-window comparison ambiguous."""
        assert _record().timestamp_utc.endswith("+00:00")

    def test_serialises_to_one_json_line(self) -> None:
        line = _record().to_json_line()
        assert "\n" not in line
        assert json.loads(line)["schema_version"] == prediction_log.SCHEMA_VERSION

    def test_raw_features_are_preserved_not_encoded(self) -> None:
        """Drift must be measured in the space a human can reason about."""
        record = _record(features_raw={"age": "[70-80)", "race": "Caucasian"})
        assert json.loads(record.to_json_line())["features_raw"]["age"] == "[70-80)"

    def test_ids_are_unique(self) -> None:
        assert len({_record().prediction_id for _ in range(200)}) == 200


class TestWriter:
    def test_appends_rather_than_overwrites(self, tmp_path: Path) -> None:
        writer = prediction_log.PredictionLogWriter(tmp_path / "p.ndjson")
        for _ in range(3):
            assert writer.write(_record())
        assert len(prediction_log.read_records(tmp_path / "p.ndjson")) == 3

    def test_batch_write_produces_identical_records(self, tmp_path: Path) -> None:
        """write_many trades fsync-per-record for fsync-per-batch, nothing else."""
        single = tmp_path / "single.ndjson"
        batch = tmp_path / "batch.ndjson"
        records = [_record(request_id=f"r{i}") for i in range(5)]

        one = prediction_log.PredictionLogWriter(single)
        for record in records:
            one.write(record)
        assert prediction_log.PredictionLogWriter(batch).write_many(records) == 5

        assert [r["request_id"] for r in prediction_log.read_records(single)] == [
            r["request_id"] for r in prediction_log.read_records(batch)
        ]

    def test_empty_batch_is_a_noop(self, tmp_path: Path) -> None:
        assert prediction_log.PredictionLogWriter(tmp_path / "p.ndjson").write_many([]) == 0

    def test_write_failure_does_not_raise(self, tmp_path: Path) -> None:
        """A monitoring write must never turn a served prediction into a 500.

        Losing a log line is a monitoring gap; failing the request is an outage.
        """
        writer = prediction_log.PredictionLogWriter(tmp_path / "p.ndjson")
        writer.path = tmp_path / "no-such-dir" / "p.ndjson"  # parent absent
        assert writer.write(_record()) is False
        assert writer.stats["failed"] == 1

    def test_tracks_counts(self, tmp_path: Path) -> None:
        writer = prediction_log.PredictionLogWriter(tmp_path / "p.ndjson")
        writer.write(_record())
        writer.write_many([_record(), _record()])
        assert writer.stats["written"] == 3


class TestReading:
    def test_tolerates_a_truncated_final_line(self, tmp_path: Path) -> None:
        """The expected state after an unclean shutdown.

        Refusing to read the whole log because of it would take monitoring down
        for a reason that does not matter — and write_many's crash semantics
        depend on this tolerance.
        """
        path = tmp_path / "p.ndjson"
        path.write_text(
            _record().to_json_line() + "\n" + '{"prediction_id": "trunc', encoding="utf-8"
        )
        assert len(prediction_log.read_records(path)) == 1

    def test_missing_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert prediction_log.read_records(tmp_path / "absent.ndjson") == []

    def test_respects_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "p.ndjson"
        prediction_log.PredictionLogWriter(path).write_many([_record() for _ in range(10)])
        assert len(prediction_log.read_records(path, limit=4)) == 4


class TestOutcomeJoin:
    def test_joins_on_prediction_id(self, tmp_path: Path) -> None:
        predictions = tmp_path / "p.ndjson"
        outcomes = tmp_path / "o.ndjson"

        record = _record()
        prediction_log.PredictionLogWriter(predictions).write(record)
        assert prediction_log.append_outcome(record.prediction_id, 1, "test", outcomes)

        joined = prediction_log.join_outcomes(predictions, outcomes)
        assert joined[0]["outcome_label"] == 1
        assert joined[0]["outcome_source"] == "test"

    def test_unmatured_predictions_keep_a_null_outcome(self, tmp_path: Path) -> None:
        """Most records are unmatured at any moment — that is normal, not an error."""
        predictions = tmp_path / "p.ndjson"
        outcomes = tmp_path / "o.ndjson"

        matched, unmatched = _record(), _record()
        prediction_log.PredictionLogWriter(predictions).write_many([matched, unmatched])
        prediction_log.append_outcome(matched.prediction_id, 1, "test", outcomes)

        joined = prediction_log.join_outcomes(predictions, outcomes)
        matured = [r for r in joined if r["outcome_label"] is not None]
        assert len(joined) == 2
        assert len(matured) == 1

    def test_a_corrected_outcome_supersedes_the_earlier_one(self, tmp_path: Path) -> None:
        """Last write wins: a correction must not be silently ignored."""
        predictions = tmp_path / "p.ndjson"
        outcomes = tmp_path / "o.ndjson"

        record = _record()
        prediction_log.PredictionLogWriter(predictions).write(record)
        prediction_log.append_outcome(record.prediction_id, 0, "first", outcomes)
        prediction_log.append_outcome(record.prediction_id, 1, "corrected", outcomes)

        joined = prediction_log.join_outcomes(predictions, outcomes)
        assert joined[0]["outcome_label"] == 1
        assert joined[0]["outcome_source"] == "corrected"

    def test_outcomes_are_appended_not_written_into_the_prediction(self, tmp_path: Path) -> None:
        """The prediction log stays append-only and immutable.

        Rewriting a record in place would risk the whole file on a crash and
        would erase the fact that the label arrived later — which Phase 6 needs
        in order to reason about maturation lag.
        """
        predictions = tmp_path / "p.ndjson"
        record = _record()
        prediction_log.PredictionLogWriter(predictions).write(record)
        before = predictions.read_bytes()

        prediction_log.append_outcome(record.prediction_id, 1, "test", tmp_path / "o.ndjson")
        assert predictions.read_bytes() == before
