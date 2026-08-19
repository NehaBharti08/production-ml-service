"""Drift detection and empirical-null calibration.

The properties tested here are the ones that make drift alerting trustworthy
rather than merely present:

*   PSI behaves correctly on identical, shifted, and pathological inputs.
*   Thresholds are *derived* from data, not defaulted.
*   A single-window breach does not raise an alert.
*   Induced drift is labelled as induced everywhere it appears.

That last one is not a code-correctness property — it is an honesty property.
A monitoring demo that shows a detector firing without saying the drift was
manufactured is claiming something it has not earned, so the labelling is
enforced by test rather than left to discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlservice.monitoring import drift, null_calibration, replay

pytestmark = pytest.mark.unit


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #


class TestPSI:
    def test_identical_distributions_score_zero(self, rng: np.random.Generator) -> None:
        s = pd.Series(rng.normal(0, 1, 5000))
        assert null_calibration.population_stability_index(s, s) < 1e-9

    def test_shifted_distribution_scores_high(self, rng: np.random.Generator) -> None:
        a = pd.Series(rng.normal(0, 1, 5000))
        b = pd.Series(rng.normal(2, 1, 5000))
        assert null_calibration.population_stability_index(a, b) > 0.25

    def test_psi_grows_with_the_size_of_the_shift(self, rng: np.random.Generator) -> None:
        base = pd.Series(rng.normal(0, 1, 5000))
        scores = [
            null_calibration.population_stability_index(base, pd.Series(rng.normal(shift, 1, 5000)))
            for shift in (0.1, 0.5, 1.0, 2.0)
        ]
        assert scores == sorted(scores), f"PSI not monotonic in shift size: {scores}"

    def test_categorical_shift_is_detected(self) -> None:
        a = pd.Series(["x"] * 900 + ["y"] * 100)
        b = pd.Series(["x"] * 500 + ["y"] * 500)
        assert null_calibration.population_stability_index(a, b) > 0.2

    def test_a_new_category_registers_as_drift(self) -> None:
        """A category absent from the reference must not be silently dropped."""
        a = pd.Series(["x"] * 1000)
        b = pd.Series(["x"] * 500 + ["brand_new"] * 500)
        assert null_calibration.population_stability_index(a, b) > 0.2

    def test_disjoint_categories_stay_finite(self) -> None:
        """Without the epsilon this is infinite, and one rare value would
        become a permanent alarm."""
        a = pd.Series(["x"] * 100)
        b = pd.Series(["y"] * 100)
        psi = null_calibration.population_stability_index(a, b)
        assert np.isfinite(psi)

    def test_degenerate_constant_feature_does_not_crash(self) -> None:
        """number_emergency is 0 for most patients; quantile edges collapse."""
        a = pd.Series([0] * 1000)
        b = pd.Series([0] * 990 + [5] * 10)
        assert np.isfinite(null_calibration.population_stability_index(a, b))

    def test_skewed_numeric_uses_quantile_bins(self, rng: np.random.Generator) -> None:
        """Equal-width bins on a skewed feature put all mass in one bin and lose
        the resolution to detect anything."""
        a = pd.Series(np.concatenate([np.zeros(900), rng.integers(1, 50, 100)]))
        b = pd.Series(np.concatenate([np.zeros(600), rng.integers(1, 50, 400)]))
        assert null_calibration.population_stability_index(a, b) > 0.1


# --------------------------------------------------------------------------- #
# Empirical-null calibration
# --------------------------------------------------------------------------- #


def _stable_frame(rng: np.random.Generator, n: int = 4000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "time_in_hospital": rng.integers(1, 14, n),
            "num_lab_procedures": rng.integers(0, 100, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.integers(0, 3, n),
            "number_emergency": rng.integers(0, 2, n),
            "number_inpatient": rng.integers(0, 4, n),
            "number_diagnoses": rng.integers(1, 16, n),
            "race": rng.choice(["A", "B", "C"], n),
            "gender": rng.choice(["M", "F"], n),
            "age": rng.choice(["[60-70)", "[70-80)"], n),
        }
    )


class TestCalibration:
    def test_thresholds_are_clamped_into_the_convention(self, rng: np.random.Generator) -> None:
        result = null_calibration.calibrate(_stable_frame(rng), n_windows=10)
        for feature in result.features.values():
            assert result.floor <= feature.threshold <= result.ceiling

    def test_a_stable_feature_lands_on_the_floor(self, rng: np.random.Generator) -> None:
        """Random-but-stationary features have near-zero churn, so the 99th
        percentile falls below the floor and the clamp catches it."""
        result = null_calibration.calibrate(_stable_frame(rng), n_windows=10)
        assert any(f.clamped == "floor" for f in result.features.values())

    def test_every_feature_records_its_null_distribution(self, rng: np.random.Generator) -> None:
        """The distribution is the evidence. Keeping only the threshold would
        make the number unfalsifiable."""
        result = null_calibration.calibrate(_stable_frame(rng), n_windows=10)
        for feature in result.features.values():
            assert len(feature.psi_values) == 9  # adjacent pairs of 10 windows
            assert feature.median <= feature.p99 <= feature.maximum

    def test_refuses_windows_too_small_to_be_stable(self, rng: np.random.Generator) -> None:
        """PSI on tiny windows is noise; failing loudly beats a bad threshold."""
        with pytest.raises(ValueError, match="too few for PSI"):
            null_calibration.calibrate(_stable_frame(rng, n=500), n_windows=20)

    def test_calibration_is_deterministic(self, rng: np.random.Generator) -> None:
        frame = _stable_frame(rng)
        a = null_calibration.calibrate(frame, n_windows=10).thresholds()
        b = null_calibration.calibrate(frame, n_windows=10).thresholds()
        assert a == b


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


class TestDetection:
    def test_no_drift_between_identical_frames(self, rng: np.random.Generator) -> None:
        frame = _stable_frame(rng)
        thresholds = dict.fromkeys(frame.columns, 0.1)
        results = drift.detect_data_drift(frame, frame.copy(), thresholds)
        assert results
        assert not [r for r in results if r.breaching]

    def test_a_shifted_feature_is_flagged(self, rng: np.random.Generator) -> None:
        reference = _stable_frame(rng)
        current = reference.copy()
        current["number_inpatient"] = current["number_inpatient"] + 5

        thresholds = dict.fromkeys(reference.columns, 0.1)
        results = drift.detect_data_drift(reference, current, thresholds)
        breaching = {r.feature for r in results if r.breaching}
        assert "number_inpatient" in breaching

    def test_a_feature_without_a_calibrated_threshold_is_skipped(
        self, rng: np.random.Generator
    ) -> None:
        """Never silently defaulted — a defaulted threshold is exactly the
        arbitrary number this project avoids."""
        frame = _stable_frame(rng)
        results = drift.detect_data_drift(frame, frame.copy(), {"race": 0.1})
        assert {r.feature for r in results} == {"race"}

    def test_each_feature_is_judged_against_its_own_threshold(
        self, rng: np.random.Generator
    ) -> None:
        reference = _stable_frame(rng)
        current = reference.copy()
        current["number_inpatient"] = current["number_inpatient"] + 3

        strict = drift.detect_data_drift(reference, current, {"number_inpatient": 0.01})
        lenient = drift.detect_data_drift(reference, current, {"number_inpatient": 10.0})
        assert strict[0].breaching
        assert not lenient[0].breaching


class TestPredictionDrift:
    def test_uses_the_threshold_it_is_given_not_config(self) -> None:
        """Regression. An earlier version read settings.model.decision_threshold,
        which still holds the 0.5 placeholder — so every score fell below it,
        the alert rate was 0 in every window, and the signal was silently dead.
        The same bug hit the API in Phase 3.
        """
        rng = np.random.default_rng(0)
        reference = rng.beta(2, 20, 4000)  # scores concentrated near 0.09
        current = rng.beta(2, 20, 4000)

        result = drift.detect_prediction_drift(reference, current, decision_threshold=0.1)
        assert result["reference_alert_rate"] > 0, (
            "alert rate is zero — the decision threshold is not being applied"
        )

    def test_detects_a_shift_in_the_alert_rate(self) -> None:
        rng = np.random.default_rng(0)
        reference = rng.beta(2, 20, 4000)
        current = rng.beta(4, 20, 4000)  # shifted upward
        result = drift.detect_prediction_drift(reference, current, decision_threshold=0.1)
        assert result["current_alert_rate"] > result["reference_alert_rate"]
        assert result["breaching"]

    def test_quiet_when_the_distribution_is_unchanged(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.beta(2, 20, 4000)
        result = drift.detect_prediction_drift(scores, scores.copy(), decision_threshold=0.1)
        assert not result["breaching"]


class TestLabelDrift:
    def test_refuses_to_judge_on_too_few_labels(self) -> None:
        matured = pd.DataFrame({"outcome_label": [0, 1] * 10, "predicted_proba": [0.1, 0.9] * 10})
        result = drift.detect_label_drift(matured, baseline_pr_auc=0.12)
        assert not result["sufficient_labels"]
        assert not result["breaching"]

    def test_measures_the_drop_in_standard_errors_not_points(self) -> None:
        """A 0.02 drop means different things at n=500 and n=5000, so a fixed
        point threshold would page constantly on small windows."""
        rng = np.random.default_rng(0)
        n = 2000
        y = rng.binomial(1, 0.1, n)
        matured = pd.DataFrame(
            {
                "outcome_label": y,
                "predicted_proba": np.clip(rng.normal(0.1, 0.1, n) + y * 0.3, 0, 1),
            }
        )
        result = drift.detect_label_drift(matured, baseline_pr_auc=0.12)
        assert result["sufficient_labels"]
        assert "drop_in_standard_errors" in result
        assert result["bootstrap_se"] > 0

    def test_single_class_window_is_refused(self) -> None:
        matured = pd.DataFrame({"outcome_label": [0] * 1000, "predicted_proba": [0.1] * 1000})
        result = drift.detect_label_drift(matured, baseline_pr_auc=0.12)
        assert not result["sufficient_labels"]


class TestAlertConfirmation:
    """Two-window confirmation is what stops the pager firing on noise."""

    def _report(self, n_breaching: int) -> drift.DriftReport:
        report = drift.DriftReport(
            window_start="0",
            window_end="1",
            window_rows=5000,
            reference_rows=5000,
            feature_schema_hash="h",
            comparable=True,
        )
        report.features = [
            drift.FeatureDrift(f"f{i}", 0.5, 0.1, True, 5000, 5000) for i in range(n_breaching)
        ]
        return report

    def test_one_breaching_window_does_not_alert(self) -> None:
        """With ~43 features at a 99th-percentile threshold, roughly 0.4 breach
        per window by chance. A single window is noise."""
        assert not drift.alert_state([self._report(5)])["confirmed"]

    def test_two_consecutive_breaching_windows_alert(self) -> None:
        assert drift.alert_state([self._report(5), self._report(5)])["confirmed"]

    def test_a_quiet_window_breaks_the_streak(self) -> None:
        state = drift.alert_state([self._report(5), self._report(5), self._report(0)])
        assert not state["confirmed"]

    def test_too_few_features_does_not_alert(self) -> None:
        """One feature moving is not a population shift."""
        assert not drift.alert_state([self._report(1), self._report(1)])["confirmed"]


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #


class TestInducedDriftIsLabelled:
    """Enforced by test because it is the claim most easily overstated.

    A demo showing a detector fire, without saying the drift was manufactured,
    implies a finding it has not earned.
    """

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 3000
        return pd.DataFrame(
            {
                "encounter_id": np.arange(n),
                "age": rng.choice(["[50-60)", "[60-70)", "[70-80)", "[80-90)"], n),
                "number_inpatient": rng.integers(0, 4, n),
                "medical_specialty": rng.choice(["A", "B", "C"], n),
            }
        )

    def test_untouched_windows_are_marked_real(self, frame: pd.DataFrame) -> None:
        windows = replay.chronological_windows(frame, window_rows=1000)
        assert windows
        assert all(w.drift_origin == "real" for w in windows)
        assert all(not w.manipulations for w in windows)

    def test_manipulated_windows_are_marked_induced(self, frame: pd.DataFrame) -> None:
        windows = replay.induced_windows(
            frame, inducer="age", window_rows=1000, clean_windows=1, drifted_windows=2
        )
        assert windows[0].drift_origin == "real"
        assert all(w.drift_origin == "induced" for w in windows[1:])

    def test_every_manipulation_records_what_it_changed(self, frame: pd.DataFrame) -> None:
        windows = replay.induced_windows(
            frame, inducer="age", window_rows=1000, clean_windows=1, drifted_windows=1
        )
        manipulation = windows[-1].manipulations[0]
        assert manipulation.feature == "age"
        assert manipulation.detail
        assert manipulation.before
        assert manipulation.after
        assert manipulation.before != manipulation.after

    def test_the_report_leads_with_drift_origin(self) -> None:
        """Nobody should be able to read the artefact without seeing it."""
        result = replay.ReplayResult(drift_origin="induced", inducer="age")
        payload = result.to_dict()
        keys = list(payload)
        assert keys.index("drift_origin") < keys.index("windows")
        assert "ARTIFICIAL" in payload["honesty_note"]

    def test_a_real_replay_says_so_too(self) -> None:
        payload = replay.ReplayResult(drift_origin="real", inducer=None).to_dict()
        assert "No manipulation applied" in payload["honesty_note"]

    def test_the_inducers_actually_shift_the_feature(self, frame: pd.DataFrame) -> None:
        """A demo whose manipulation does nothing proves nothing."""
        window = frame.iloc[:1000]
        for name, inducer in replay.INDUCERS.items():
            shifted, _ = inducer(window)
            feature = {
                "age": "age",
                "utilisation": "number_inpatient",
                "specialty": "medical_specialty",
            }[name]
            psi = null_calibration.population_stability_index(window[feature], shifted[feature])
            assert psi > 0.1, f"inducer {name!r} barely moved {feature} (PSI {psi:.4f})"


class TestSchemaComparability:
    def test_mismatched_schema_hash_is_flagged_not_silently_compared(
        self, rng: np.random.Generator
    ) -> None:
        """Drift across a schema change compares distributions that are not
        comparable; the report must refuse rather than lie."""
        reference = _stable_frame(rng)
        reference.attrs["feature_schema_hash"] = "aaa"
        current = _stable_frame(rng)
        current.attrs["feature_schema_hash"] = "bbb"

        report = drift.analyse_window(reference, current)
        assert not report.comparable
        assert any("SCHEMA MISMATCH" in n for n in report.notes)

    def test_small_window_is_noted(self, rng: np.random.Generator) -> None:
        reference = _stable_frame(rng)
        report = drift.analyse_window(reference, reference.iloc[:50])
        assert any("below the" in n for n in report.notes)


class TestMaturationSimulation:
    def test_takes_the_earliest_rows_not_a_random_sample(self) -> None:
        """Labels mature oldest-first. A random sample would pretend they
        arrive uniformly, which is the one property delayed-label monitoring
        exists to handle."""
        frame = pd.DataFrame({"target": list(range(100))})
        matured = replay.simulate_maturation(frame, np.linspace(0, 1, 100), fraction=0.6)
        assert len(matured) == 60
        assert matured["outcome_label"].tolist() == list(range(60))
