"""Retraining triggers, promotion and rollback.

Two things are tested here that are easy to get wrong and expensive to get
wrong quietly:

1.  **A trigger must not fire on noise, and must fire on evidence.** Both
    directions matter. A trigger that never fires is a pipeline nobody notices
    is dead; a trigger that always fires ships models on evidence nobody read.

2.  **Rollback must actually move the alias.** The first version of
    :func:`~mlservice.retraining.promote.verify_rollback_path` promoted the
    version that was already serving, recorded ``2 -> 2`` and reported success.
    The test below pins the property that verification was missing: after a
    promote-then-rollback the alias is back where it started, *having been
    somewhere else in between*.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mlservice.monitoring import drift as drift_mod
from mlservice.retraining import promote as promote_mod
from mlservice.retraining import trigger as trigger_mod

pytestmark = pytest.mark.unit


# =============================================================================
# Triggers
# =============================================================================


class TestScheduledTrigger:
    def test_fires_when_the_model_is_older_than_the_refresh_window(self) -> None:
        now = datetime(2026, 8, 29, tzinfo=UTC)
        result = trigger_mod.scheduled_trigger(now - timedelta(days=31), now=now)
        assert result.fired
        assert result.detail["age_days"] == 31

    def test_does_not_fire_inside_the_window(self) -> None:
        now = datetime(2026, 8, 29, tzinfo=UTC)
        assert not trigger_mod.scheduled_trigger(now - timedelta(days=29), now=now).fired

    def test_boundary_day_thirty_fires(self) -> None:
        """Exactly 30 days is due. Off-by-one here silently doubles the cadence."""
        now = datetime(2026, 8, 29, tzinfo=UTC)
        assert trigger_mod.scheduled_trigger(now - timedelta(days=30), now=now).fired

    def test_no_training_history_fires(self) -> None:
        """An unknown last-trained date is a reason to train, not to assume fresh."""
        result = trigger_mod.scheduled_trigger(None)
        assert result.fired
        assert result.detail["last_trained"] is None


class TestDriftTrigger:
    def test_fires_only_on_a_confirmed_alert(self) -> None:
        assert trigger_mod.drift_trigger({"confirmed": True, "reason": "sustained"}).fired

    def test_does_not_fire_on_an_unconfirmed_single_window(self) -> None:
        """The two-window confirmation is the entire anti-flapping mechanism."""
        assert not trigger_mod.drift_trigger({"confirmed": False, "breaching_counts": [9]}).fired

    def test_does_not_fire_on_an_empty_alert(self) -> None:
        """No evidence is not evidence of drift — and must not start a retrain."""
        assert not trigger_mod.drift_trigger({}).fired


class TestPerformanceTrigger:
    def test_fires_on_measured_degradation(self) -> None:
        result = trigger_mod.performance_trigger(
            {"sufficient_labels": True, "breaching": True, "drop_in_standard_errors": 3.1}
        )
        assert result.fired

    def test_does_not_fire_without_enough_matured_labels(self) -> None:
        """Judging performance on a handful of labels is guessing with extra steps."""
        result = trigger_mod.performance_trigger(
            {"sufficient_labels": False, "breaching": True, "n_matured": 12}
        )
        assert not result.fired
        assert "insufficient matured labels" in result.reason

    def test_insufficient_labels_beats_a_breach_flag(self) -> None:
        """Ordering matters: the label-count check must dominate.

        Otherwise a label pipeline delivering 12 rows of noise can trip a
        retrain, which is the pipeline failing *upstream* being laundered into
        a model decision.
        """
        result = trigger_mod.performance_trigger(
            {"sufficient_labels": False, "breaching": True, "n_matured": 12}
        )
        assert not result.fired


class TestEvaluateTriggers:
    def test_any_single_trigger_is_sufficient(self) -> None:
        now = datetime(2026, 8, 29, tzinfo=UTC)
        decision = trigger_mod.evaluate_triggers(
            last_trained=now - timedelta(days=1), now=now, manual=True, requested_by="neha"
        )
        assert decision.retrain
        assert [t.name for t in decision.fired] == ["manual"]

    def test_all_triggers_are_evaluated_even_once_one_fires(self) -> None:
        """No short-circuit: the record must show the full state at decision time.

        "We retrained on drift, and performance was also degrading" is a
        materially different story from "we retrained on drift".
        """
        now = datetime(2026, 8, 29, tzinfo=UTC)
        decision = trigger_mod.evaluate_triggers(
            last_trained=now - timedelta(days=1),
            now=now,
            drift_alert={"confirmed": True},
            label_drift={"sufficient_labels": True, "breaching": True},
            manual=True,
        )
        assert len(decision.triggers) == 4
        assert {t.name for t in decision.fired} == {"drift", "performance", "manual"}

    def test_nothing_fires_on_an_empty_world(self) -> None:
        now = datetime(2026, 8, 29, tzinfo=UTC)
        decision = trigger_mod.evaluate_triggers(last_trained=now - timedelta(days=1), now=now)
        assert not decision.retrain
        assert decision.reason == "no trigger fired"


def test_not_triggers_are_documented_in_code() -> None:
    """The excluded conditions are part of the policy, not a comment.

    They are asserted here so deleting one is a test failure rather than a
    quiet loosening of the retraining policy.
    """
    assert any("single feature" in item for item in trigger_mod.NOT_TRIGGERS)
    assert any("latency" in item for item in trigger_mod.NOT_TRIGGERS)


# =============================================================================
# Reading drift evidence
# =============================================================================


def _report(n_breaching: int) -> dict[str, Any]:
    return {"n_breaching": n_breaching, "breaching_features": ["f"] * n_breaching}


class TestBreachingCounts:
    def test_reads_the_serialised_key(self) -> None:
        assert drift_mod.breaching_counts([_report(0), _report(5)]) == [0, 5]

    def test_raises_on_a_missing_key_rather_than_scoring_zero(self) -> None:
        """The regression test for a bug that made drift undetectable.

        The trigger read ``report.get("breaching", [])``, but ``to_dict``
        emits ``n_breaching``. Every window scored zero, so a replay whose own
        alert state said ``confirmed: True`` produced "no confirmed drift".
        A default on a key that must exist converts a schema mismatch into a
        confident wrong answer.
        """
        with pytest.raises(KeyError):
            drift_mod.breaching_counts([{"breaching": ["a", "b", "c"]}])


class TestLoadReports:
    def test_reads_windows_out_of_replay_runs(self, tmp_path: Path, env: Any) -> None:
        """Replay windows count as evidence; they are stored inline, not as files."""
        env(MLSERVICE_PATHS__REPORTS=str(tmp_path))
        (tmp_path / "replay_induced.json").write_text(
            json.dumps(
                {
                    "windows": [
                        {"drift_origin": "induced", "drift": _report(7)},
                        {"drift_origin": "induced", "drift": _report(8)},
                    ]
                }
            ),
            encoding="utf-8",
        )
        reports = drift_mod.load_reports()
        assert drift_mod.breaching_counts(reports) == [7, 8]

    def test_preserves_drift_origin(self, tmp_path: Path, env: Any) -> None:
        """Induced-vs-real must survive the round trip.

        Losing it here is how an honest demo turns into a misleading claim two
        layers up the stack.
        """
        env(MLSERVICE_PATHS__REPORTS=str(tmp_path))
        (tmp_path / "replay_x.json").write_text(
            json.dumps({"windows": [{"drift_origin": "induced", "drift": _report(4)}]}),
            encoding="utf-8",
        )
        assert drift_mod.load_reports()[0]["drift_origin"] == "induced"

    def test_no_reports_is_an_empty_list_not_an_error(self, tmp_path: Path, env: Any) -> None:
        env(MLSERVICE_PATHS__REPORTS=str(tmp_path))
        assert drift_mod.load_reports() == []


# =============================================================================
# Promotion and rollback
# =============================================================================


class _FakeClient:
    """Stands in for MlflowClient. Holds one alias, like the real registry."""

    def __init__(self, alias_points_at: str | None = None) -> None:
        self.alias: str | None = alias_points_at
        self.calls: list[str] = []

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.alias = str(version)
        self.calls.append(str(version))


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env: Any) -> _FakeClient:
    """A fake registry plus an isolated promotion history."""
    env(MLSERVICE_PATHS__REPORTS=str(tmp_path))
    client = _FakeClient(alias_points_at="1")

    import mlflow.tracking

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *a, **k: client)
    monkeypatch.setattr(promote_mod, "current_champion", lambda: client.alias)
    return client


class TestPromote:
    def test_refuses_when_the_gates_did_not_pass(self, registry: _FakeClient) -> None:
        """The gates are the whole control.

        A promote function with a bypass flag is a promote function with no
        gates, so there is deliberately no way to ask for one.
        """
        with pytest.raises(ValueError, match="gates did not pass"):
            promote_mod.promote(version="2", trigger="drift", gates_passed=False)
        assert registry.calls == []

    def test_records_the_previous_version_before_flipping(self, registry: _FakeClient) -> None:
        """Captured before, not reconstructed after.

        Once the alias moves, "what was it before" is no longer answerable from
        the registry — and an incident is the worst time to start doing
        archaeology on run history.
        """
        entry = promote_mod.promote(version="2", trigger="drift", gates_passed=True)
        assert entry.from_version == "1"
        assert entry.to_version == "2"
        assert registry.alias == "2"

    def test_appends_to_the_audit_trail(self, registry: _FakeClient) -> None:
        promote_mod.promote(version="2", trigger="scheduled", gates_passed=True, approver="neha")
        entries = promote_mod.history()
        assert len(entries) == 1
        assert entries[0]["approver"] == "neha"
        assert entries[0]["action"] == "promote"


class TestRollback:
    def test_returns_the_alias_to_the_previous_version(self, registry: _FakeClient) -> None:
        promote_mod.promote(version="2", trigger="drift", gates_passed=True)
        assert registry.alias == "2"

        entry = promote_mod.rollback(reason="canary p99 breached")
        assert entry.to_version == "1"
        assert registry.alias == "1"

    def test_the_alias_genuinely_moved_in_between(self, registry: _FakeClient) -> None:
        """The property the original verification failed to check.

        Promoting the version that is already serving and then "rolling back"
        to it satisfies *ends where it started* while proving nothing. The
        alias must have been somewhere else in the middle.
        """
        promote_mod.promote(version="2", trigger="drift", gates_passed=True)
        midpoint = registry.alias
        promote_mod.rollback(reason="verification")

        assert midpoint == "2"
        assert registry.alias == "1"
        assert midpoint != registry.alias

    def test_is_not_gated(self, registry: _FakeClient) -> None:
        """Rollback must work when everything else is broken. That is the point."""
        promote_mod.promote(version="2", trigger="drift", gates_passed=True)
        entry = promote_mod.rollback(reason="incident")
        assert entry.gates_passed is False

    def test_refuses_with_no_promotion_history(self, registry: _FakeClient) -> None:
        with pytest.raises(ValueError, match="nothing to roll back to"):
            promote_mod.rollback(reason="nothing here")

    def test_refuses_when_the_first_model_has_no_predecessor(
        self, registry: _FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An honest failure beats a confident flip to nowhere."""
        monkeypatch.setattr(promote_mod, "current_champion", lambda: None)
        promote_mod.promote(version="1", trigger="initial", gates_passed=True)

        monkeypatch.setattr(promote_mod, "current_champion", lambda: "1")
        with pytest.raises(ValueError, match="nothing to roll back to"):
            promote_mod.rollback(reason="no predecessor")

    def test_reads_history_rather_than_the_registry(self, registry: _FakeClient) -> None:
        """Chained promotions must roll back one step, not to the oldest version."""
        promote_mod.promote(version="2", trigger="drift", gates_passed=True)
        promote_mod.promote(version="3", trigger="scheduled", gates_passed=True)

        entry = promote_mod.rollback(reason="v3 was bad")
        assert entry.to_version == "2"
        assert registry.alias == "2"
