"""Evaluation, calibration and selection logic.

These are the functions that decide which model ships. A bug here does not
crash anything — it promotes the wrong model quietly, which is the most
expensive kind of bug this repo can have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlservice.models import calibration, evaluate, subgroups

pytestmark = pytest.mark.unit


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


class TestInterval:
    def test_overlapping_intervals_are_not_an_improvement(self) -> None:
        """The rule that stops promotion on noise."""
        better = evaluate.Interval(0.12, 0.11, 0.14)
        worse = evaluate.Interval(0.11, 0.10, 0.13)
        assert better.overlaps(worse)
        assert not evaluate.beats(better, worse)["genuinely_better"]

    def test_separated_intervals_are_an_improvement(self) -> None:
        better = evaluate.Interval(0.20, 0.18, 0.22)
        worse = evaluate.Interval(0.11, 0.10, 0.13)
        assert not better.overlaps(worse)
        assert evaluate.beats(better, worse)["genuinely_better"]

    def test_higher_point_estimate_alone_is_not_enough(self) -> None:
        """A higher number with overlapping CIs must not count as better."""
        result = evaluate.beats(
            evaluate.Interval(0.1234, 0.1110, 0.1375),
            evaluate.Interval(0.1147, 0.1037, 0.1268),
        )
        assert result["point_difference"] > 0
        assert result["intervals_overlap"]
        assert not result["genuinely_better"]


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self, rng: np.random.Generator) -> None:
        from sklearn.metrics import average_precision_score

        y = rng.binomial(1, 0.1, 2000)
        s = rng.uniform(0, 1, 2000) * 0.5 + y * 0.3
        ci = evaluate.bootstrap_metric(y, s, average_precision_score, iterations=200)
        assert ci.lower <= ci.point <= ci.upper

    def test_is_deterministic_for_a_fixed_seed(self, rng: np.random.Generator) -> None:
        """Reproducibility matters: these numbers go into a model card."""
        from sklearn.metrics import average_precision_score

        y = rng.binomial(1, 0.1, 500)
        s = rng.uniform(0, 1, 500)
        a = evaluate.bootstrap_metric(y, s, average_precision_score, iterations=100, seed=7)
        b = evaluate.bootstrap_metric(y, s, average_precision_score, iterations=100, seed=7)
        assert (a.lower, a.upper) == (b.lower, b.upper)


class TestThresholdSelection:
    def test_meets_the_recall_target(self, rng: np.random.Generator) -> None:
        y = rng.binomial(1, 0.1, 3000)
        s = np.clip(rng.normal(0.1, 0.1, 3000) + y * 0.25, 0, 1)
        _, info = evaluate.choose_threshold(y, s, target_recall=0.5)
        assert info["target_met"]
        assert info["achieved_recall"] >= 0.5

    def test_never_returns_the_naive_half(self, rng: np.random.Generator) -> None:
        """0.5 on an imbalanced problem flags almost nobody."""
        y = rng.binomial(1, 0.08, 3000)
        s = np.clip(rng.normal(0.08, 0.08, 3000) + y * 0.2, 0, 1)
        threshold, _ = evaluate.choose_threshold(y, s, target_recall=0.5)
        assert threshold < 0.5

    def test_degrades_gracefully_when_target_is_unreachable(self) -> None:
        """A constant score cannot hit an arbitrary recall — must not crash."""
        y = np.array([0] * 90 + [1] * 10)
        s = np.full(100, 0.5)
        threshold, info = evaluate.choose_threshold(y, s, target_recall=0.99)
        assert isinstance(threshold, float)
        assert "target_met" in info


class TestCalibrationMetrics:
    def test_perfect_calibration_scores_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        prob = rng.uniform(0, 1, 20000)
        y = rng.binomial(1, prob)  # outcomes drawn from the stated probability
        report = calibration.expected_calibration_error(y, prob)
        assert report.ece < 0.02

    def test_systematic_overconfidence_is_detected(self) -> None:
        y = np.zeros(1000, dtype=int)
        prob = np.full(1000, 0.9)  # claims 90%, observes 0%
        report = calibration.expected_calibration_error(y, prob)
        assert report.ece > 0.8
        assert report.mce > 0.8

    def test_mce_exceeds_ece_when_one_bin_is_badly_wrong(self) -> None:
        """The random forest in Phase 2 had ECE 0.0139 and MCE 0.80.

        A single summary number would have hidden that, which is why both are
        reported.
        """
        prob = np.concatenate([np.full(9900, 0.05), np.full(100, 0.95)])
        y = np.concatenate([np.zeros(9900, dtype=int), np.zeros(100, dtype=int)])
        report = calibration.expected_calibration_error(y, prob)
        assert report.mce > report.ece * 5

    def test_bin_counts_sum_to_the_sample(self) -> None:
        rng = np.random.default_rng(2)
        prob = rng.uniform(0, 1, 5000)
        y = rng.binomial(1, 0.1, 5000)
        report = calibration.expected_calibration_error(y, prob)
        assert sum(report.bin_counts) == 5000

    def test_probability_of_one_lands_in_the_final_bin(self) -> None:
        """Off-by-one at the top edge silently drops the highest-risk cases."""
        report = calibration.expected_calibration_error(
            np.array([1, 1, 0]), np.array([1.0, 1.0, 0.0])
        )
        assert report.bin_counts[-1] == 2
        assert report.bin_counts[0] == 1


class TestSubgroups:
    @pytest.fixture
    def frame(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        n = 3000
        df = pd.DataFrame(
            {
                "race": rng.choice(["A", "B", "TinyGroup"], n, p=[0.6, 0.38, 0.02]),
                "gender": rng.choice(["F", "M"], n),
            }
        )
        y = rng.binomial(1, 0.1, n)
        score = np.clip(rng.normal(0.1, 0.1, n) + y * 0.2, 0, 1)
        return df, y, score

    def test_small_groups_are_reported_but_not_analysed(
        self, frame: tuple[pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        """A gap computed on 60 patients is noise presented as a finding."""
        df, y, score = frame
        report = subgroups.evaluate_subgroups(df, y, score, 0.15, ("race", "gender"))
        tiny = next(g for g in report.groups if g.group == "TinyGroup")
        assert not tiny.sufficient
        assert "n below" in tiny.note
        assert tiny not in report.analysable

    def test_worst_gap_ignores_small_groups(
        self, frame: tuple[pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        df, y, score = frame
        report = subgroups.evaluate_subgroups(df, y, score, 0.15, ("race", "gender"))
        worst = report.worst_group
        assert worst is None or worst.n >= subgroups.MIN_SUBGROUP_N

    def test_one_threshold_is_applied_to_every_group(
        self, frame: tuple[pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        """Per-group thresholds would mean treating patients differently by
        demographics — the thing this analysis exists to detect."""
        df, y, score = frame
        report = subgroups.evaluate_subgroups(df, y, score, 0.15, ("race",))
        assert report.threshold == 0.15

    def test_a_real_disparity_is_surfaced(self) -> None:
        """Construct a group the model is deliberately worse at."""
        n = 2000
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"race": ["A"] * n + ["B"] * n})
        y = np.concatenate([rng.binomial(1, 0.1, n), rng.binomial(1, 0.1, n)])
        # Group B gets uninformative scores; group A gets useful ones.
        score = np.concatenate(
            [
                np.clip(rng.normal(0.1, 0.05, n) + y[:n] * 0.4, 0, 1),
                np.clip(rng.normal(0.1, 0.05, n), 0, 1),
            ]
        )
        report = subgroups.evaluate_subgroups(df, y, score, 0.15, ("race",))
        a = next(g for g in report.groups if g.group == "A")
        b = next(g for g in report.groups if g.group == "B")
        assert a.recall > b.recall
        assert report.worst_recall_gap < 0


class TestSelection:
    def _result(self, name: str, point: float, lo: float, hi: float, rank: int):
        from mlservice.models.train import CandidateResult

        return CandidateResult(
            name=name,
            rationale="",
            complexity_rank=rank,
            threshold=0.1,
            calibration_method="isotonic",
            val_pr_auc=point,
            test={"pr_auc": {"point": point, "lower": lo, "upper": hi}},
        )

    def test_prefers_the_simpler_model_when_intervals_overlap(self) -> None:
        """The Phase 2 rule, using the actual numbers from the real run."""
        from mlservice.models.train import select_champion

        results = [
            self._result("logistic_l2", 0.1234, 0.1110, 0.1375, 1),
            self._result("random_forest_shallow", 0.1300, 0.1180, 0.1420, 3),
        ]
        champion, rationale = select_champion(results)
        assert champion.name == "logistic_l2"
        assert rationale["highest_point_estimate"] == "random_forest_shallow"

    def test_prefers_the_complex_model_when_it_genuinely_wins(self) -> None:
        from mlservice.models.train import select_champion

        results = [
            self._result("logistic_l2", 0.1234, 0.1110, 0.1375, 1),
            self._result("random_forest_shallow", 0.2500, 0.2300, 0.2700, 3),
        ]
        champion, _ = select_champion(results)
        assert champion.name == "random_forest_shallow"

    def test_baselines_are_never_selected(self) -> None:
        from mlservice.models.train import select_champion

        results = [
            self._result("baseline_prevalence", 0.9, 0.85, 0.95, 0),
            self._result("logistic_l2", 0.1234, 0.1110, 0.1375, 1),
        ]
        champion, _ = select_champion(results)
        assert champion.name == "logistic_l2"
