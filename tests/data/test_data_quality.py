"""Data-quality gates that run before any training.

These assert facts about the *data*, not the code, and they are the checks that
would have caught the defects this project actually shipped. Each one exists
because something went wrong, not because a checklist suggested it.

They are slow — the raw file is 1.6 GB — so the fixtures are module-scoped and
the suite is marked ``data``. CI runs it as its own job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlservice.config import get_settings
from mlservice.data import audit, clean, schema, split

pytestmark = [pytest.mark.data, pytest.mark.slow]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    try:
        return audit.load_raw()
    except FileNotFoundError:
        pytest.skip("dataset not downloaded — run `uv run mlservice data download`")


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    frame, _ = clean.clean(raw)
    return frame


@pytest.fixture(scope="module")
def split_result(cleaned: pd.DataFrame) -> split.SplitResult:
    return split.chronological_split(cleaned)


class TestRawIntegrity:
    def test_shape_matches_what_the_audit_recorded(self, raw: pd.DataFrame) -> None:
        """A changed mirror must fail here, loudly.

        Lending Club withdrew the official download, so the source is a
        community mirror. The checksum is the primary guard; this is the second
        one, and it states the expected shape in terms a human can check.
        """
        assert len(raw) == schema.RAW_ROW_COUNT
        assert len(raw.columns) == schema.RAW_COLUMN_COUNT

    def test_only_the_known_footer_rows_lack_a_date(self, raw: pd.DataFrame) -> None:
        """Every real row must have a usable date; the footers must not.

        The split is chronological, so a row whose date fails to parse would be
        silently dropped or mis-ordered, and ``errors="coerce"`` makes that
        failure invisible. This test found 33 such rows on its first run — they
        turned out to be export footers ("Total amount funded in policy code
        1: ...") sitting in the id column with every other field empty.

        Pinned at exactly 33 rather than relaxed to "a few". If the number
        moves, the export changed, and that is worth knowing before it becomes
        a silent row loss.
        """
        parsed = clean.parse_issue_date(raw)
        unparseable = raw[parsed.isna()]
        assert len(unparseable) == 33, f"{len(unparseable)} rows lack a date, expected 33 footers"
        # A footer has one populated cell. A real loan with a broken date would
        # have many, and must not be mistaken for one.
        assert (unparseable.notna().sum(axis=1) == 1).all()

    def test_the_footer_rows_are_removed_explicitly(self, raw: pd.DataFrame) -> None:
        """Removed by a rule that names them, not incidentally by the label filter."""
        report = clean.CleaningReport(rows_in=len(raw))
        clean.drop_non_loan_rows(raw, report)
        step = next(s for s in report.steps if s["step"] == "drop_non_loan_rows")
        assert step["rows_dropped"] == 33

    def test_borrower_identity_is_genuinely_unavailable(self, raw: pd.DataFrame) -> None:
        """Pins the limitation so it cannot quietly stop being true.

        If a future mirror *did* populate member_id, deduplicating by borrower
        would become possible and the model card's limitation would be wrong.
        Better to fail here than to keep publishing a stale caveat.
        """
        assert split.BORROWER_IDENTITY_AVAILABLE is False
        assert raw["member_id"].isna().all(), "member_id is populated — revisit the split"


class TestLeakage:
    def test_post_origination_columns_are_gone(self, cleaned: pd.DataFrame) -> None:
        leaking = set(get_settings().data.post_origination_columns)
        survivors = leaking & set(cleaned.columns)
        assert not survivors, f"post-origination columns survived cleaning: {sorted(survivors)}"

    def test_the_target_source_column_is_gone(self, cleaned: pd.DataFrame) -> None:
        """The regression test for the leak that actually happened.

        ``loan_status`` is the column the label is derived from, so it predicts
        the target perfectly. It survived the first version of the pipeline
        because the data dictionary describes it as a loan attribute — which it
        is, right up until you make it the label.
        """
        assert schema.TARGET_SOURCE not in cleaned.columns

    def test_no_surviving_column_perfectly_predicts_the_target(self, cleaned: pd.DataFrame) -> None:
        """A general net, not a list of known offenders.

        Any categorical whose every level maps to a single outcome is either a
        leak or a constant. Enumerating known leaks catches the ones already
        found; this catches the next one.
        """
        offenders = []
        for column in cleaned.columns:
            if column == "target" or pd.api.types.is_numeric_dtype(cleaned[column]):
                continue
            grouped = cleaned.groupby(column, observed=True)["target"]
            if grouped.nunique().max() == 1 and cleaned[column].nunique() > 1:
                offenders.append(column)
        assert not offenders, f"columns that perfectly predict the target: {offenders}"

    def test_the_leak_would_have_been_detectable(self, raw: pd.DataFrame) -> None:
        """The demonstration itself, asserted rather than described.

        If this ever stops reaching a near-perfect AUC, either the leaking
        columns changed or the label did — and both are worth knowing.
        """
        result = audit.demonstrate_leakage(raw, sample_frac=0.05)
        assert result["roc_auc_with_leakage"] > 0.95
        assert result["recoveries_positive_default_rate"] == 1.0


class TestCensoring:
    def test_only_matured_loans_survive(self, cleaned: pd.DataFrame) -> None:
        """No loan may extend past the observation window.

        This is the rule that moves the positive rate from 0.1998 to 0.1481.
        A regression here would silently reintroduce survivorship bias.
        """
        settings = get_settings().data
        issued = clean.parse_issue_date(cleaned)
        term = clean.parse_term_months(cleaned)
        matures = issued + pd.to_timedelta(term * 30.44, unit="D")
        assert (matures <= pd.Timestamp(settings.observation_end)).all()

    def test_no_unresolved_status_survives(self, raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
        """In-flight loans are dropped, never counted as repaid."""
        settings = get_settings().data
        terminal = set(settings.loan_status_bad) | set(settings.loan_status_good)
        unresolved = set(raw[settings.target_column].dropna().unique()) - terminal
        assert unresolved, "expected some in-flight statuses in the raw data"
        assert len(cleaned) < len(raw), "cleaning removed nothing — the filter is not running"

    def test_the_positive_rate_is_what_the_model_card_claims(self, cleaned: pd.DataFrame) -> None:
        rate = float(cleaned["target"].mean())
        assert rate == pytest.approx(schema.CLEANED_POSITIVE_RATE, abs=0.005), (
            f"positive rate {rate:.4f} has moved from the documented "
            f"{schema.CLEANED_POSITIVE_RATE} — every downstream number changed with it"
        )


class TestCleanedShape:
    def test_shape_matches_the_schema(self, cleaned: pd.DataFrame) -> None:
        assert len(cleaned) == schema.CLEANED_ROW_COUNT
        assert len(cleaned.columns) == schema.CLEANED_COLUMN_COUNT

    def test_every_declared_feature_exists(self, cleaned: pd.DataFrame) -> None:
        missing = set(schema.ALL_FEATURES) - set(cleaned.columns)
        assert not missing, f"schema declares features the data lacks: {sorted(missing)}"

    def test_no_undeclared_columns_reach_the_model(self, cleaned: pd.DataFrame) -> None:
        allowed = set(schema.ALL_FEATURES) | {"target", schema.TIME_COLUMN}
        extra = set(cleaned.columns) - allowed
        assert not extra, f"columns present but undeclared: {sorted(extra)}"

    def test_no_constant_columns_survive(self, cleaned: pd.DataFrame) -> None:
        constant = [c for c in cleaned.columns if len(cleaned[c].unique()) <= 1]
        assert not constant, f"zero-information columns survived: {constant}"

    def test_no_perfectly_collinear_numeric_pairs(self, cleaned: pd.DataFrame) -> None:
        """fico_range_low/high correlated at exactly 1.0 and both shipped.

        Perfect collinearity gives a linear model two identical columns to
        split a coefficient across, inflating its variance and making the
        weights unreadable.
        """
        numeric = [c for c in schema.NUMERIC_FEATURES if c in cleaned.columns]
        corr = cleaned[numeric].corr().abs()
        np.fill_diagonal(corr.values, 0.0)
        pairs = [
            (a, b) for a in corr.index for b in corr.columns if a < b and corr.loc[a, b] > 0.999
        ]
        assert not pairs, f"perfectly collinear pairs survived: {pairs}"


class TestSplit:
    def test_the_split_is_chronological(self, split_result: split.SplitResult) -> None:
        """Train strictly before val strictly before test.

        A mis-ordered split still trains, still scores, and simply reports
        numbers that are too good — which is why this is asserted rather than
        assumed.
        """
        train = clean.parse_issue_date(split_result.train)
        val = clean.parse_issue_date(split_result.val)
        test = clean.parse_issue_date(split_result.test)
        assert train.max() < val.min()
        assert val.max() < test.min()

    def test_no_rows_are_lost(self, split_result: split.SplitResult, cleaned: pd.DataFrame) -> None:
        total = len(split_result.train) + len(split_result.val) + len(split_result.test)
        assert total == len(cleaned)

    def test_every_split_has_both_classes(self, split_result: split.SplitResult) -> None:
        for name, frame in (
            ("train", split_result.train),
            ("val", split_result.val),
            ("test", split_result.test),
        ):
            assert frame["target"].nunique() == 2, f"{name} has only one class"

    def test_the_borrower_limitation_is_recorded(self, split_result: split.SplitResult) -> None:
        """The limitation must travel with the split, not live only in prose."""
        assert any("member_id" in limitation for limitation in split_result.limitations)


class TestSeparability:
    def test_the_problem_is_not_trivially_separable(self, split_result: split.SplitResult) -> None:
        """Inverted logic: an unconstrained tree that generalises means a leak.

        This is the check that would have caught ``loan_status`` had it run
        before the schema was written rather than after.
        """
        result = audit.separability_check(split_result.train, split_result.test)
        assert not result["alarm_triggered"], (
            f"an unconstrained tree reached {result['test_roc_auc']} test ROC-AUC — "
            "something is leaking"
        )
