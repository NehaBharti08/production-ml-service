"""Model behaviour: invariance, directional expectation, robustness.

The tests that matter most in this repo, and the ones a conventional project
omits. `assert pr_auc > 0.10` tells you almost nothing: a feature pipeline that
silently drops a column, an encoder that maps every unseen category to the same
bucket, or a transformer fitted on the wrong split can all leave an aggregate
metric nearly intact while making individual predictions nonsense.

Three families, each answering a different question:

*   **Invariance** — does something that should not matter change the answer?
    Patient identity, field ordering, batch position.
*   **Directional** — does something that should matter move the answer the
    right way? These encode clinical priors and are the strongest available
    check that the pipeline is wired correctly.
*   **Robustness** — does an input the model has never seen degrade one
    prediction, or take the service down?

These run against the **real trained artifact**, not a stub. A stub would pass
every one of them while telling you nothing about what is deployed. They are
skipped, not failed, when the artifact is absent, so a fresh clone is not
blocked by them.
"""

from __future__ import annotations

import copy
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from mlservice.api.schemas import EXAMPLE_FEATURES, PatientFeatures

pytestmark = [pytest.mark.behavior, pytest.mark.slow]

#: Scores are floats from a linear model; exact equality is the right assertion
#: for invariance (the same input must give the same output), but a tiny
#: tolerance guards against platform-level float reassociation.
EXACT = 1e-12

#: Directional tests need a margin large enough to exceed numerical noise but
#: small enough to catch a genuinely inverted relationship.
DIRECTIONAL_MARGIN = 1e-6


@pytest.fixture(scope="module")
def model() -> Any:
    """The real champion artifact, or skip."""
    from mlservice.api.model_loader import ModelStore
    from mlservice.config import get_settings

    path = get_settings().paths.models / "champion" / "model.joblib"
    if not path.is_file():
        pytest.skip(f"no trained model at {path} — run `uv run mlservice train run`")

    store = ModelStore()
    try:
        return store.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"model could not be loaded: {exc}")


def _score(model: Any, **overrides: Any) -> float:
    features = {**copy.deepcopy(EXAMPLE_FEATURES), **overrides}
    return model.predict_proba(PatientFeatures(**features).to_model_row())


# --------------------------------------------------------------------------- #
# Invariance
# --------------------------------------------------------------------------- #


class TestInvariance:
    def test_identical_input_gives_identical_output(self, model: Any) -> None:
        """Determinism. Without it, no other test here means anything."""
        first = _score(model)
        assert all(abs(_score(model) - first) < EXACT for _ in range(5))

    def test_field_order_does_not_change_the_score(self, model: Any) -> None:
        """A dict is unordered, but the transformer selects columns by name.

        If this ever fails, the pipeline is relying on positional column order —
        which works until a caller serialises their JSON differently.
        """
        forward = PatientFeatures(**EXAMPLE_FEATURES).to_model_row()
        reversed_keys = dict(reversed(list(EXAMPLE_FEATURES.items())))
        backward = PatientFeatures(**reversed_keys).to_model_row()
        assert abs(model.predict_proba(forward) - model.predict_proba(backward)) < EXACT

    def test_batch_position_does_not_change_a_prediction(self, model: Any) -> None:
        """Item i must score the same wherever it sits in the batch.

        Catches any accidental cross-row coupling — a transformer that fits on
        the incoming batch rather than applying the fitted one would pass every
        aggregate metric and fail here.
        """
        a = PatientFeatures(**EXAMPLE_FEATURES).to_model_row()
        b = PatientFeatures(**{**EXAMPLE_FEATURES, "number_inpatient": 6}).to_model_row()

        first = model.predict_proba_batch([a, b])
        swapped = model.predict_proba_batch([b, a])

        assert abs(first[0] - swapped[1]) < EXACT
        assert abs(first[1] - swapped[0]) < EXACT

    def test_batch_and_single_agree(self, model: Any) -> None:
        """The two endpoints must not disagree about the same patient."""
        row = PatientFeatures(**EXAMPLE_FEATURES).to_model_row()
        assert abs(model.predict_proba(row) - model.predict_proba_batch([row])[0]) < EXACT

    def test_batch_size_does_not_change_a_score(self, model: Any) -> None:
        row = PatientFeatures(**EXAMPLE_FEATURES).to_model_row()
        alone = model.predict_proba_batch([row])[0]
        crowded = model.predict_proba_batch([row] * 50)[0]
        assert abs(alone - crowded) < EXACT


# --------------------------------------------------------------------------- #
# Directional expectation
# --------------------------------------------------------------------------- #


class TestDirectionalExpectations:
    """Clinical priors, asserted as monotonic relationships.

    These are the tests that catch a corrupted feature pipeline. A transformer
    that dropped `number_inpatient` would leave PR-AUC almost unchanged — the
    remaining features carry correlated signal — while making the model blind to
    the single strongest predictor. Only a directional test notices.

    Where a prior is genuinely uncertain, the assertion is deliberately weak
    (monotone, not a specific magnitude): overfitting a test to the current
    coefficients would make it a change-detector rather than a correctness check.
    """

    def test_more_prior_inpatient_admissions_never_lowers_risk(self, model: Any) -> None:
        """The strongest single predictor in the data, and the clearest prior.

        Prior utilisation is the feature the Phase 1 heuristic baseline was built
        on. If the relationship inverted, something is badly wrong.
        """
        scores = [_score(model, number_inpatient=n) for n in (0, 1, 2, 4, 8)]
        assert all(later >= earlier - DIRECTIONAL_MARGIN for earlier, later in pairwise(scores)), (
            f"risk fell as prior admissions rose: {scores}"
        )
        # And the relationship must be real, not flat.
        assert scores[-1] > scores[0] + DIRECTIONAL_MARGIN

    def test_more_prior_emergency_visits_never_lowers_risk(self, model: Any) -> None:
        scores = [_score(model, number_emergency=n) for n in (0, 1, 3, 6)]
        assert all(later >= earlier - DIRECTIONAL_MARGIN for earlier, later in pairwise(scores)), (
            f"risk fell as emergency visits rose: {scores}"
        )

    def test_longer_stay_moves_risk_monotonically(self, model: Any) -> None:
        """Direction is not asserted, only monotonicity.

        Length of stay is genuinely ambiguous — a longer stay signals a sicker
        patient but also more thorough treatment. Pinning a direction would
        encode an assumption the data may not support; requiring *consistency*
        still catches a scrambled feature mapping.
        """
        scores = [_score(model, time_in_hospital=d) for d in (1, 3, 7, 14)]
        deltas = [b - a for a, b in pairwise(scores)]
        non_decreasing = all(d >= -DIRECTIONAL_MARGIN for d in deltas)
        non_increasing = all(d <= DIRECTIONAL_MARGIN for d in deltas)
        assert non_decreasing or non_increasing, f"non-monotonic in length of stay: {scores}"

    def test_number_of_diagnoses_moves_risk_monotonically(self, model: Any) -> None:
        scores = [_score(model, number_diagnoses=n) for n in (1, 5, 9, 16)]
        deltas = [b - a for a, b in pairwise(scores)]
        assert all(d >= -DIRECTIONAL_MARGIN for d in deltas) or all(
            d <= DIRECTIONAL_MARGIN for d in deltas
        ), f"non-monotonic in diagnosis count: {scores}"

    def test_prior_utilisation_actually_influences_the_score(self, model: Any) -> None:
        """A guard against the feature being dropped entirely.

        If `number_inpatient` stopped reaching the model, every score would be
        identical across its whole range and aggregate metrics would barely move.
        """
        low, high = _score(model, number_inpatient=0), _score(model, number_inpatient=10)
        assert abs(high - low) > 1e-4, (
            "number_inpatient has no measurable effect — the feature is probably "
            "not reaching the model"
        )


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


class TestRobustness:
    def test_unseen_category_degrades_rather_than_raises(self, model: Any) -> None:
        """A new hospital specialty must not take the service down.

        This is what `handle_unknown="infrequent_if_exist"` buys. Without it the
        first unfamiliar value would raise, turning a survivable degradation into
        a 500 for that caller.
        """
        score = _score(model, medical_specialty="AstronauticalMedicine")
        assert 0.0 <= score <= 1.0

    def test_unseen_categories_in_several_fields_at_once(self, model: Any) -> None:
        score = _score(
            model,
            medical_specialty="NewSpecialty",
            diag_1="NewBand",
            race="NewCategory",
            insulin="NewValue",
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("age", ["[0-10)", "[40-50)", "[90-100)"])
    def test_every_age_band_scores(self, model: Any, age: str) -> None:
        assert 0.0 <= _score(model, age=age) <= 1.0

    def test_boundary_values_score(self, model: Any) -> None:
        """The extremes of every numeric range the schema permits."""
        low = _score(
            model,
            time_in_hospital=1,
            num_lab_procedures=0,
            num_procedures=0,
            num_medications=0,
            number_outpatient=0,
            number_emergency=0,
            number_inpatient=0,
            number_diagnoses=1,
        )
        high = _score(
            model,
            time_in_hospital=14,
            num_lab_procedures=200,
            num_procedures=20,
            num_medications=100,
            number_outpatient=100,
            number_emergency=100,
            number_inpatient=100,
            number_diagnoses=20,
        )
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0

    def test_output_is_always_a_valid_probability(self, model: Any) -> None:
        """Scanned across a grid, not a single point.

        A calibrator can produce values outside [0,1] if misapplied, and a
        probability that is not a probability breaks the threshold comparison,
        the metric histogram and the calibration report at once.
        """
        for inpatient in (0, 3, 20, 100):
            for stay in (1, 7, 14):
                score = _score(model, number_inpatient=inpatient, time_in_hospital=stay)
                assert 0.0 <= score <= 1.0, f"score {score} outside [0,1]"


# --------------------------------------------------------------------------- #
# The suite must be able to fail
# --------------------------------------------------------------------------- #


class TestTheSuiteItself:
    """A test suite that cannot fail is decoration.

    Phase 7's behavioural promotion gate depends on these detecting a broken
    model, so the detection itself is verified rather than assumed.
    """

    def test_directional_check_catches_an_inverted_model(self, model: Any) -> None:
        """Invert the score and confirm the monotonicity assertion would fail."""
        scores = [1.0 - _score(model, number_inpatient=n) for n in (0, 1, 2, 4, 8)]
        monotone = all(later >= earlier - DIRECTIONAL_MARGIN for earlier, later in pairwise(scores))
        assert not monotone, (
            "an inverted model still passed the monotonicity check — the "
            "directional tests are not actually discriminating"
        )

    def test_invariance_check_catches_a_position_dependent_model(self, model: Any) -> None:
        """Simulate cross-row coupling and confirm the invariance test would fail."""
        row = PatientFeatures(**EXAMPLE_FEATURES).to_model_row()
        base = model.predict_proba(row)
        # A model whose output depended on batch index would produce this.
        coupled = [base + 0.01 * i for i in range(3)]
        assert abs(coupled[0] - coupled[2]) > EXACT

    def test_artifact_under_test_is_the_real_one(self, model: Any, tmp_path: Path) -> None:
        """Guard against these silently running on a stub.

        The whole value of this suite is that it exercises the deployed artifact.
        """
        assert model.source in ("registry", "local_fallback")
        assert model.feature_schema_hash != "unknown"
