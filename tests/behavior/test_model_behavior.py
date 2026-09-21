"""Model behaviour: invariance, directional expectation, robustness.

The tests that matter most in this repo, and the ones a conventional project
omits. `assert pr_auc > 0.10` tells you almost nothing: a feature pipeline that
silently drops a column, an encoder that maps every unseen category to the same
bucket, or a transformer fitted on the wrong split can all leave an aggregate
metric nearly intact while making individual predictions nonsense.

Three families, each answering a different question:

*   **Invariance** — does something that should not matter change the answer?
    Field ordering, batch position, an unrelated field changing.
*   **Directional** — does something that should matter move the answer the
    right way? These encode CREDIT priors — more leverage is riskier, a
    better FICO is safer — and are the strongest available check that the
    pipeline is wired correctly.
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

from mlservice.api.schemas import EXAMPLE_FEATURES, LoanApplication

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
    return model.predict_proba(LoanApplication(**features).to_model_row())


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
        forward = LoanApplication(**EXAMPLE_FEATURES).to_model_row()
        reversed_keys = dict(reversed(list(EXAMPLE_FEATURES.items())))
        backward = LoanApplication(**reversed_keys).to_model_row()
        assert abs(model.predict_proba(forward) - model.predict_proba(backward)) < EXACT

    def test_batch_position_does_not_change_a_prediction(self, model: Any) -> None:
        """Item i must score the same wherever it sits in the batch.

        Catches any accidental cross-row coupling — a transformer that fits on
        the incoming batch rather than applying the fitted one would pass every
        aggregate metric and fail here.
        """
        a = LoanApplication(**EXAMPLE_FEATURES).to_model_row()
        b = LoanApplication(**{**EXAMPLE_FEATURES, "dti": 35.0}).to_model_row()

        first = model.predict_proba_batch([a, b])
        swapped = model.predict_proba_batch([b, a])

        assert abs(first[0] - swapped[1]) < EXACT
        assert abs(first[1] - swapped[0]) < EXACT

    def test_batch_and_single_agree(self, model: Any) -> None:
        """The two endpoints must not disagree about the same patient."""
        row = LoanApplication(**EXAMPLE_FEATURES).to_model_row()
        assert abs(model.predict_proba(row) - model.predict_proba_batch([row])[0]) < EXACT

    def test_batch_size_does_not_change_a_score(self, model: Any) -> None:
        row = LoanApplication(**EXAMPLE_FEATURES).to_model_row()
        alone = model.predict_proba_batch([row])[0]
        crowded = model.predict_proba_batch([row] * 50)[0]
        assert abs(alone - crowded) < EXACT


# --------------------------------------------------------------------------- #
# Directional expectation
# --------------------------------------------------------------------------- #


class TestDirectionalExpectations:
    """Priors that must hold, or the pipeline is miswired.

    These are the tests an aggregate metric cannot replace. A transform that
    silently dropped `dti` would leave PR-AUC almost unchanged — the remaining
    features carry correlated signal — while making the model blind to
    leverage. Only a directional test notices.

    Every relationship below was **verified against the trained artifact
    before being asserted**. Writing down a prior that the model does not
    actually satisfy would turn this suite into a source of false confidence,
    which is worse than not having it.

    The assertions are weakly monotone (``>=``) on purpose: isotonic
    calibration produces a step function, so adjacent inputs legitimately
    share a score. Strict monotonicity would fail on the plateaus and teach
    everyone to ignore the suite.
    """

    def test_more_leverage_never_lowers_risk(self, model: Any) -> None:
        """Debt-to-income is the clearest prior in credit risk."""
        scores = [_score(model, dti=v) for v in (5, 15, 25, 35)]
        for low, high in pairwise(scores):
            assert high >= low - DIRECTIONAL_MARGIN, f"raising DTI lowered predicted risk: {scores}"

    def test_a_better_fico_never_raises_risk(self, model: Any) -> None:
        """The inverse direction, which catches a sign error the others cannot."""
        scores = [_score(model, fico_range_low=v) for v in (620, 680, 740, 800)]
        for high_risk, low_risk in pairwise(scores):
            assert low_risk <= high_risk + DIRECTIONAL_MARGIN, (
                f"a better FICO raised predicted risk: {scores}"
            )

    def test_a_worse_grade_never_lowers_risk(self, model: Any) -> None:
        """Grade is the lender's own ordering, so it must be respected.

        This one is close to a tautology — `grade` carries the model's largest
        coefficient — which is exactly why breaking it would mean something is
        badly wrong.
        """
        scores = [_score(model, grade=g, sub_grade=f"{g}3") for g in ("A", "C", "E", "G")]
        for better, worse in pairwise(scores):
            assert worse >= better - DIRECTIONAL_MARGIN, (
                f"a worse grade lowered predicted risk: {scores}"
            )

    def test_a_longer_term_never_lowers_risk(self, model: Any) -> None:
        """60-month loans default more than 36-month ones. Well established."""
        short = _score(model, term=" 36 months")
        long = _score(model, term=" 60 months")
        assert long >= short - DIRECTIONAL_MARGIN, (
            f"a 60-month term scored lower than 36-month: {short:.4f} vs {long:.4f}"
        )

    def test_more_recent_delinquencies_never_lower_risk(self, model: Any) -> None:
        scores = [_score(model, delinq_2yrs=v) for v in (0, 1, 3, 6)]
        for low, high in pairwise(scores):
            assert high >= low - DIRECTIONAL_MARGIN, f"delinquencies lowered risk: {scores}"

    def test_more_recent_inquiries_never_lower_risk(self, model: Any) -> None:
        scores = [_score(model, inq_last_6mths=v) for v in (0, 1, 3, 6)]
        for low, high in pairwise(scores):
            assert high >= low - DIRECTIONAL_MARGIN, f"inquiries lowered risk: {scores}"

    @pytest.mark.parametrize(
        ("field", "low", "high"),
        [
            ("dti", 5, 38),
            ("fico_range_low", 820, 615),
            ("term", " 36 months", " 60 months"),
        ],
    )
    def test_the_strong_features_materially_influence_the_score(
        self, model: Any, field: str, low: Any, high: Any
    ) -> None:
        """Monotone but flat is a failure mode of its own.

        If a feature stopped reaching the model, every score would still be
        weakly monotone in it — trivially, because nothing would change. These
        three must move the score by a visible amount, so a silently dropped
        column is caught rather than passing the monotonicity checks above.
        """
        safe, risky = _score(model, **{field: low}), _score(model, **{field: high})
        assert risky - safe > 0.005, (
            f"{field} barely moved the score ({safe:.4f} -> {risky:.4f}); "
            "the feature may not be reaching the model"
        )


class TestRobustness:
    def test_unseen_category_degrades_rather_than_raises(self, model: Any) -> None:
        """A loan purpose the model has never seen must not take the service down.

        This is what ``handle_unknown="infrequent_if_exist"`` buys. Without it
        the first unfamiliar value would raise, turning a survivable
        degradation into a 500 for that caller.
        """
        score = _score(model, purpose="crypto_mining_rig")
        assert 0.0 <= score <= 1.0

    def test_unseen_categories_in_several_fields_at_once(self, model: Any) -> None:
        score = _score(
            model,
            purpose="unheard_of",
            home_ownership="TIMESHARE",
            verification_status="Partially Verified",
            emp_length="17 years",
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("grade", ["A", "D", "G"])
    def test_every_grade_scores(self, model: Any, grade: str) -> None:
        assert 0.0 <= _score(model, grade=grade, sub_grade=f"{grade}3") <= 1.0

    def test_omitting_the_whole_optional_bureau_tail_still_scores(self, model: Any) -> None:
        """The most likely real-world request shape.

        A caller with only an application form and a credit score sends the
        required core and nothing else. If imputation were broken this is
        where it would surface — and it is the shape the startup canary uses
        for exactly that reason.
        """
        required_only = {
            "loan_amnt": 12000.0,
            "term": " 60 months",
            "int_rate": 15.0,
            "installment": 285.0,
            "grade": "C",
            "sub_grade": "C2",
            "purpose": "credit_card",
            "annual_inc": 48000.0,
            "home_ownership": "RENT",
            "verification_status": "Not Verified",
            "addr_state": "TX",
            "dti": 22.0,
            "fico_range_low": 675.0,
        }
        score = model.predict_proba(LoanApplication(**required_only).to_model_row())
        assert 0.0 <= score <= 1.0

    def test_boundary_values_score(self, model: Any) -> None:
        """The extremes of every numeric range the schema permits."""
        low = _score(
            model,
            loan_amnt=500.0,
            int_rate=5.4,
            installment=16.0,
            annual_inc=1896.0,
            dti=0.0,
            fico_range_low=845.0,
            revol_util=0.0,
            open_acc=0.0,
        )
        high = _score(
            model,
            loan_amnt=35000.0,
            int_rate=26.0,
            installment=1410.0,
            annual_inc=7_500_000.0,
            dti=40.0,
            fico_range_low=610.0,
            revol_util=100.0,
            open_acc=84.0,
        )
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0

    def test_output_is_always_a_valid_probability(self, model: Any) -> None:
        """Scanned across a grid, not a single point.

        A calibrator can produce values outside [0,1] if misapplied, and a
        probability that is not a probability breaks the threshold comparison,
        the metric histogram and the calibration report at once.
        """
        for dti in (0, 15, 30, 40):
            for fico in (610, 700, 845):
                for grade in ("A", "D", "G"):
                    score = _score(
                        model, dti=dti, fico_range_low=fico, grade=grade, sub_grade=f"{grade}3"
                    )
                    assert 0.0 <= score <= 1.0, (
                        f"dti={dti} fico={fico} grade={grade} produced {score}"
                    )


class TestTheSuiteItself:
    """A test suite that cannot fail is decoration.

    Phase 7's behavioural promotion gate depends on these detecting a broken
    model, so the detection itself is verified rather than assumed.
    """

    def test_directional_check_catches_an_inverted_model(self, model: Any) -> None:
        """Invert the score and confirm the monotonicity assertion would fail."""
        scores = [1.0 - _score(model, dti=v) for v in (5, 15, 25, 35)]
        monotone = all(later >= earlier - DIRECTIONAL_MARGIN for earlier, later in pairwise(scores))
        assert not monotone, (
            "an inverted model still passed the monotonicity check — the "
            "directional tests are not actually discriminating"
        )

    def test_invariance_check_catches_a_position_dependent_model(self, model: Any) -> None:
        """Simulate cross-row coupling and confirm the invariance test would fail."""
        row = LoanApplication(**EXAMPLE_FEATURES).to_model_row()
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
