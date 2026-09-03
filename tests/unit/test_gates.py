"""Promotion gates.

The most consequential tests in the repository. A gate that cannot fail is
decoration, and a decorative gate is worse than none — it creates confidence
that nothing is checking.

So every gate is exercised with a **deliberately-bad challenger constructed to
fail that gate and no other**. That isolation matters: a bad model that trips
five gates proves nothing about which gate caught it.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mlservice.retraining import gates

pytestmark = pytest.mark.unit


def _evaluation(**overrides: Any) -> dict[str, Any]:
    """A healthy evaluation, roughly matching the real Phase 2 champion."""
    base: dict[str, Any] = {
        "pr_auc": {"point": 0.1234, "lower": 0.1110, "upper": 0.1375},
        "brier": {"point": 0.0691, "lower": 0.0660, "upper": 0.0720},
        "calibration": {"ece": 0.0140, "mce": 0.1000},
        "subgroups": {"worst_recall_gap": -0.232, "worst_group": "age=[40-50)"},
        "behavioral": {"total": 20, "passed": 20, "failures": []},
        "artifact_loads": True,
        # Not a secret: a deterministic hash of the feature column list, and the
        # operational gate compares it against the serving contract.
        "feature_schema_hash": "06f5f0b873ca95f6",  # pragma: allowlist secret
        "data_quality": {"suite_passed": True, "missingness_increase_pct_points": {}},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


@pytest.fixture
def incumbent() -> dict[str, Any]:
    return _evaluation()


class TestHealthyChallengerPromotes:
    def test_an_identical_model_passes_every_gate(self, incumbent: dict[str, Any]) -> None:
        decision = gates.evaluate_promotion(copy.deepcopy(incumbent), incumbent)
        assert decision.promote, decision.reason
        assert len(decision.gates) == 6

    def test_a_genuinely_better_model_promotes(self, incumbent: dict[str, Any]) -> None:
        better = _evaluation(
            pr_auc={"point": 0.15, "lower": 0.14, "upper": 0.16},
            brier={"point": 0.065, "lower": 0.062, "upper": 0.068},
            calibration={"ece": 0.010},
        )
        assert gates.evaluate_promotion(better, incumbent).promote


class TestPerformanceGate:
    def test_blocks_a_materially_worse_model(self, incumbent: dict[str, Any]) -> None:
        worse = _evaluation(pr_auc={"point": 0.09, "lower": 0.08, "upper": 0.10})
        result = gates.performance_gate(worse, incumbent)
        assert not result.passed
        assert "below the non-inferiority floor" in result.reason

    def test_allows_a_refresh_that_is_equally_good(self, incumbent: dict[str, Any]) -> None:
        """Non-inferiority, not strict improvement.

        Demanding improvement on every scheduled refresh would block legitimate
        freshness updates, and a gate that blocks routine work gets disabled.
        """
        same = _evaluation(pr_auc={"point": 0.1234, "lower": 0.111, "upper": 0.1375})
        assert gates.performance_gate(same, incumbent).passed

    def test_tolerates_noise_within_the_margin(self, incumbent: dict[str, Any]) -> None:
        marginally_worse = _evaluation(
            pr_auc={"point": 0.1234 - 0.004, "lower": 0.11, "upper": 0.13}
        )
        assert gates.performance_gate(marginally_worse, incumbent).passed

    def test_rejects_just_beyond_the_margin(self, incumbent: dict[str, Any]) -> None:
        beyond = _evaluation(pr_auc={"point": 0.1234 - 0.006, "lower": 0.11, "upper": 0.13})
        assert not gates.performance_gate(beyond, incumbent).passed


class TestCalibrationGate:
    """The gate that carries this project's thesis."""

    def test_blocks_a_better_ranking_but_worse_calibrated_model(
        self, incumbent: dict[str, Any]
    ) -> None:
        """The case ADR 0002 exists for.

        Higher PR-AUC, materially worse Brier. For a health-adjacent use case
        this is a regression: a probability nobody can trust cannot support a
        decision, whatever its ranking quality.
        """
        challenger = _evaluation(
            pr_auc={"point": 0.16, "lower": 0.15, "upper": 0.17},  # clearly better
            brier={"point": 0.0691 * 1.10, "lower": 0.07, "upper": 0.08},  # 10% worse
        )
        assert gates.performance_gate(challenger, incumbent).passed, (
            "setup error: this challenger should pass on performance"
        )

        result = gates.calibration_gate(challenger, incumbent)
        assert not result.passed
        assert "calibration degraded" in result.reason

        decision = gates.evaluate_promotion(challenger, incumbent)
        assert not decision.promote
        assert decision.failed[0].name == "calibration"

    def test_blocks_an_absolutely_miscalibrated_model(self, incumbent: dict[str, Any]) -> None:
        """A challenger cannot be badly calibrated just because Brier improved."""
        challenger = _evaluation(
            brier={"point": 0.060, "lower": 0.058, "upper": 0.062},  # better Brier
            calibration={"ece": 0.12},  # but well past the 0.05 ceiling
        )
        result = gates.calibration_gate(challenger, incumbent)
        assert not result.passed
        assert "not trustworthy" in result.reason

    def test_tolerates_a_2pct_brier_regression(self, incumbent: dict[str, Any]) -> None:
        """A tolerance for noise between retrains, not a licence to degrade."""
        challenger = _evaluation(brier={"point": 0.0691 * 1.019, "lower": 0.06, "upper": 0.08})
        assert gates.calibration_gate(challenger, incumbent).passed

    def test_rejects_just_past_the_tolerance(self, incumbent: dict[str, Any]) -> None:
        challenger = _evaluation(brier={"point": 0.0691 * 1.03, "lower": 0.06, "upper": 0.08})
        assert not gates.calibration_gate(challenger, incumbent).passed

    def test_reports_both_failures_when_both_fail(self, incumbent: dict[str, Any]) -> None:
        challenger = _evaluation(
            brier={"point": 0.0691 * 1.20, "lower": 0.08, "upper": 0.09},
            calibration={"ece": 0.15},
        )
        result = gates.calibration_gate(challenger, incumbent)
        assert not result.passed
        assert "AND" in result.reason


class TestSubgroupGate:
    def test_blocks_a_model_that_widens_the_worst_gap(self, incumbent: dict[str, Any]) -> None:
        """Incumbent's worst gap is -0.232; a 30% widening exceeds the 20% limit."""
        challenger = _evaluation(subgroups={"worst_recall_gap": -0.232 * 1.30})
        result = gates.subgroup_gate(challenger, incumbent)
        assert not result.passed
        assert "widened" in result.reason

    def test_allows_a_model_that_narrows_the_gap(self, incumbent: dict[str, Any]) -> None:
        challenger = _evaluation(subgroups={"worst_recall_gap": -0.15})
        assert gates.subgroup_gate(challenger, incumbent).passed

    def test_tolerates_widening_within_the_limit(self, incumbent: dict[str, Any]) -> None:
        challenger = _evaluation(subgroups={"worst_recall_gap": -0.232 * 1.15})
        assert gates.subgroup_gate(challenger, incumbent).passed

    def test_blocks_introducing_a_gap_where_none_existed(self) -> None:
        """With no incumbent disparity there is nothing to widen, so an absolute
        check applies — a challenger must not create one from nothing."""
        clean = _evaluation(subgroups={"worst_recall_gap": 0.0})
        challenger = _evaluation(subgroups={"worst_recall_gap": -0.40})
        assert not gates.subgroup_gate(challenger, clean).passed

    def test_measures_against_the_incumbent_not_an_absolute_constant(
        self, incumbent: dict[str, Any]
    ) -> None:
        """The incumbent already has a -0.232 gap. An absolute fairness constant
        would block every possible challenger, including a strictly better one —
        which is why the gate is relative."""
        slightly_better = _evaluation(subgroups={"worst_recall_gap": -0.230})
        assert gates.subgroup_gate(slightly_better, incumbent).passed


class TestBehavioralGate:
    def test_blocks_on_a_single_failure(self, incumbent: dict[str, Any]) -> None:
        """100%, not 'most'. Each test encodes a property that is either true or
        broken; there is no partial credit."""
        challenger = _evaluation(
            behavioral={
                "total": 20,
                "passed": 19,
                "failures": ["test_more_prior_inpatient_admissions_never_lowers_risk"],
            }
        )
        result = gates.behavioral_gate(challenger)
        assert not result.passed
        assert "1 of 20" in result.reason

    def test_blocks_when_the_suite_never_ran(self) -> None:
        """A model cannot be promoted on absent evidence.

        This is the more dangerous case: zero failures because zero tests ran
        looks identical to success if the gate only checks a failure count.
        """
        challenger = _evaluation(behavioral={"total": 0, "passed": 0})
        result = gates.behavioral_gate(challenger)
        assert not result.passed
        assert "did not run" in result.reason

    def test_passes_on_a_full_suite(self) -> None:
        assert gates.behavioral_gate(_evaluation()).passed


class TestOperationalGate:
    def test_blocks_an_artifact_that_does_not_load(self, incumbent: dict[str, Any]) -> None:
        challenger = _evaluation(artifact_loads=False)
        result = gates.operational_gate(challenger, incumbent)
        assert not result.passed
        assert "does not load" in result.reason

    def test_blocks_a_feature_schema_mismatch(self, incumbent: dict[str, Any]) -> None:
        """The API sends records shaped by one contract. A model expecting
        another would raise on every request, or silently score wrong columns."""
        challenger = _evaluation(feature_schema_hash="deadbeefdeadbeef")
        result = gates.operational_gate(challenger, incumbent)
        assert not result.passed
        assert "feature schema hash" in result.reason

    def test_passes_a_matching_loadable_artifact(self, incumbent: dict[str, Any]) -> None:
        assert gates.operational_gate(_evaluation(), incumbent).passed


class TestDataQualityGate:
    def test_blocks_a_failed_data_quality_suite(self) -> None:
        challenger = _evaluation(data_quality={"suite_passed": False})
        assert not gates.data_quality_gate(challenger).passed

    def test_blocks_a_missingness_jump(self) -> None:
        """A 10pp jump is usually an upstream break, not a population change.

        Retraining on it bakes the break in — and the resulting model passes
        every performance gate, because it learned the broken data faithfully.
        """
        challenger = _evaluation(
            data_quality={
                "suite_passed": True,
                "missingness_increase_pct_points": {"medical_specialty": 35.0},
            }
        )
        result = gates.data_quality_gate(challenger)
        assert not result.passed
        assert "missingness rose" in result.reason

    def test_tolerates_a_small_missingness_change(self) -> None:
        challenger = _evaluation(
            data_quality={
                "suite_passed": True,
                "missingness_increase_pct_points": {"race": 2.0},
            }
        )
        assert gates.data_quality_gate(challenger).passed


class TestTheDecision:
    def test_every_gate_runs_even_after_one_fails(self, incumbent: dict[str, Any]) -> None:
        """Short-circuiting would hide the other problems, and an operator
        fixing a blocked promotion needs the whole list rather than discovering
        the next failure after each fix."""
        catastrophic = _evaluation(
            pr_auc={"point": 0.01, "lower": 0.0, "upper": 0.02},
            brier={"point": 0.5, "lower": 0.4, "upper": 0.6},
            calibration={"ece": 0.4},
            subgroups={"worst_recall_gap": -0.9},
            behavioral={"total": 20, "passed": 3, "failures": ["a", "b"]},
            artifact_loads=False,
            data_quality={"suite_passed": False},
        )
        decision = gates.evaluate_promotion(catastrophic, incumbent)
        assert not decision.promote
        assert len(decision.gates) == 6
        assert len(decision.failed) == 6

    def test_the_reason_names_the_failing_gates(self, incumbent: dict[str, Any]) -> None:
        challenger = _evaluation(calibration={"ece": 0.3})
        decision = gates.evaluate_promotion(challenger, incumbent)
        assert "calibration" in decision.reason

    def test_one_failing_gate_blocks_promotion(self, incumbent: dict[str, Any]) -> None:
        """Every gate is independently blocking — there is no scoring or
        averaging across them."""
        for override in (
            {"pr_auc": {"point": 0.01, "lower": 0.0, "upper": 0.02}},
            {"calibration": {"ece": 0.4}},
            {"subgroups": {"worst_recall_gap": -0.9}},
            {"behavioral": {"total": 20, "passed": 10, "failures": ["x"]}},
            {"artifact_loads": False},
            {"data_quality": {"suite_passed": False}},
        ):
            challenger = _evaluation(**override)
            decision = gates.evaluate_promotion(challenger, incumbent)
            assert not decision.promote, f"{override} should have blocked promotion"
            assert len(decision.failed) >= 1

    def test_the_decision_serialises_with_its_numbers(self, incumbent: dict[str, Any]) -> None:
        """When a promotion is blocked at 3am, 'calibration failed' is not
        enough — the operator needs the numbers without re-running anything."""
        challenger = _evaluation(calibration={"ece": 0.3})
        payload = gates.evaluate_promotion(challenger, incumbent).to_dict()

        assert payload["promote"] is False
        assert "calibration" in payload["failed_gates"]
        failed = next(g for g in payload["gates"] if g["name"] == "calibration")
        assert failed["detail"]["challenger_ece"] == 0.3
        assert failed["detail"]["max_ece"] == 0.05
