"""Data-quality assertions that must hold before any training runs.

These encode the Phase 1 findings as executable checks. Their job is to fail
loudly if the dataset changes, if a cleaning step regresses, or if someone
"simplifies" a step whose purpose was not obvious.

Marked ``data`` because they need the downloaded dataset. CI runs them as a
separate job; `make test` does not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlservice.data import clean, schema, split
from mlservice.data.audit import load_raw, separability_check

pytestmark = [pytest.mark.data, pytest.mark.slow]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    try:
        return load_raw()
    except FileNotFoundError:
        pytest.skip("dataset not downloaded — run `uv run mlservice data download`")


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    return clean.clean(raw)[0]


@pytest.fixture(scope="module")
def split_result(cleaned: pd.DataFrame, raw: pd.DataFrame) -> split.SplitResult:
    """Module-scoped: the split is deterministic and takes a second to build."""
    return split.chronological_split(cleaned, split.verify_time_proxy(raw))


class TestRawIntegrity:
    def test_shape_matches_expected(self, raw: pd.DataFrame) -> None:
        """Pinned so a changed upstream archive fails here, not silently later."""
        assert len(raw) == schema.RAW_ROW_COUNT
        assert len(raw.columns) == schema.RAW_COLUMN_COUNT

    def test_patient_count_matches_expected(self, raw: pd.DataFrame) -> None:
        assert raw[schema.PATIENT_ID].nunique() == schema.RAW_PATIENT_COUNT

    def test_encounter_id_is_unique(self, raw: pd.DataFrame) -> None:
        assert raw[schema.ENCOUNTER_ID].is_unique

    def test_expected_columns_all_present(self, raw: pd.DataFrame) -> None:
        missing = set(schema.expected_raw_columns()) - set(raw.columns)
        assert not missing, f"schema expects columns absent from the data: {missing}"

    def test_target_has_exactly_three_levels(self, raw: pd.DataFrame) -> None:
        assert set(raw[schema.TARGET].unique()) == {"NO", ">30", "<30"}


class TestLeakage:
    """The findings that justify the cleaning steps."""

    def test_expired_discharges_have_zero_positives(self, raw: pd.DataFrame) -> None:
        """The central leakage finding. If this ever fails, the premise changed.

        A dead patient cannot be readmitted. Zero — not merely few — positives
        is what makes the label deterministic and the leak real.
        """
        y = clean.binarise_target(raw)
        expired = raw["discharge_disposition_id"].isin(schema.EXPIRED_DISCHARGE_IDS)
        assert expired.sum() > 0, "no expired discharges found — check the code list"
        assert y[expired].sum() == 0, (
            f"{int(y[expired].sum())} expired-discharge encounters are labelled "
            "readmitted. The leakage argument rests on this being zero."
        )

    def test_hospice_is_not_deterministic(self, raw: pd.DataFrame) -> None:
        """Guards against overstating the finding.

        Hospice patients ARE readmitted. Excluding them is a clinical-relevance
        judgement, not leakage removal, and the audit says so. If this ever
        became zero the two categories would genuinely merge — but it is not.
        """
        y = clean.binarise_target(raw)
        hospice = raw["discharge_disposition_id"].isin(schema.HOSPICE_DISCHARGE_IDS)
        assert y[hospice].sum() > 0, (
            "hospice discharges show no positives — the audit's distinction "
            "between leakage and clinical judgement would no longer hold"
        )

    def test_cleaning_removes_all_excluded_discharges(self, cleaned: pd.DataFrame) -> None:
        remaining = cleaned["discharge_disposition_id"].isin(schema.EXCLUDED_DISCHARGE_IDS)
        assert not remaining.any()

    def test_no_duplicate_patients_after_cleaning(self, cleaned: pd.DataFrame) -> None:
        assert cleaned[schema.PATIENT_ID].is_unique

    def test_dropped_columns_are_gone(self, cleaned: pd.DataFrame) -> None:
        for col in schema.DROPPED_COLUMNS:
            assert col not in cleaned.columns, f"{col} should have been dropped"


class TestMissingness:
    def test_no_sentinel_survives_cleaning(self, cleaned: pd.DataFrame) -> None:
        """A surviving '?' would be silently treated as a category by encoders."""
        for col in cleaned.columns:
            if cleaned[col].dtype == object:
                n = int((cleaned[col] == schema.MISSING_SENTINEL).sum())
                assert n == 0, f"{col} still has {n} '?' values"

    def test_no_nulls_survive_cleaning(self, cleaned: pd.DataFrame) -> None:
        nulls = cleaned.isna().sum()
        assert not nulls.any(), f"null values remain: {dict(nulls[nulls > 0])}"

    def test_lab_columns_use_not_measured_category(self, cleaned: pd.DataFrame) -> None:
        """'Test not ordered' is informative and must stay an explicit category."""
        for col in schema.NOT_MEASURED_COLUMNS:
            assert schema.NOT_MEASURED_CATEGORY in set(cleaned[col].unique())


class TestTimeProxy:
    def test_proxy_verification_passes(self, raw: pd.DataFrame) -> None:
        """The gate the whole temporal claim depends on.

        If this fails, the split is an ordered holdout and every 'chronological'
        claim in the docs must be downgraded. It is a test rather than a
        notebook check precisely so that regression is impossible to miss.
        """
        result = split.verify_time_proxy(raw)
        assert result.passed, (
            f"only {result.n_trending}/{len(result.signals)} signals trend; "
            f"{split.MIN_TRENDING_SIGNALS} required. The temporal claim no "
            "longer holds and docs must be downgraded to 'ordered holdout'."
        )

    def test_payer_code_capture_increases_over_time(self, raw: pd.DataFrame) -> None:
        """The single most decisive signal: 100% missing -> ~14% missing."""
        result = split.verify_time_proxy(raw)
        signal = next(s for s in result.signals if s.column == "payer_code")
        assert signal.first_decile > 0.9
        assert signal.last_decile < 0.3
        assert signal.trends

    def test_rosiglitazone_discontinuity_is_sharp(self, raw: pd.DataFrame) -> None:
        """The dated external anchor — the Avandia collapse, 2007."""
        d = split.detect_discontinuities(raw, "rosiglitazone")
        assert d["ratio_to_typical_change"] > 3.0
        assert d["largest_drop_pct_points"] < -2.0
        assert 70 <= d["percentile_of_ordering"] <= 90


class TestSplit:
    def test_splits_do_not_overlap_in_time(self, split_result: split.SplitResult) -> None:
        assert (
            split_result.train[schema.ENCOUNTER_ID].max()
            < split_result.val[schema.ENCOUNTER_ID].min()
        )
        assert (
            split_result.val[schema.ENCOUNTER_ID].max()
            < split_result.test[schema.ENCOUNTER_ID].min()
        )

    def test_no_patient_appears_in_two_splits(self, split_result: split.SplitResult) -> None:
        train, val, test = (
            set(f[schema.PATIENT_ID])
            for f in (split_result.train, split_result.val, split_result.test)
        )
        assert not train & val
        assert not train & test
        assert not val & test

    def test_censoring_buffer_removed_a_depressed_region(
        self, split_result: split.SplitResult
    ) -> None:
        """The dropped tail must have a lower positive rate than what is kept.

        If it does not, the buffer is discarding good data for no reason and
        should be reconsidered rather than kept out of habit.
        """
        c = split_result.censoring
        assert c["positive_rate_dropped_region"] < c["positive_rate_retained"]

    def test_all_splits_contain_both_classes(self, split_result: split.SplitResult) -> None:
        for name, frame in (
            ("train", split_result.train),
            ("val", split_result.val),
            ("test", split_result.test),
        ):
            assert frame["target"].nunique() == 2, f"{name} is single-class"

    def test_split_is_reproducible(
        self, cleaned: pd.DataFrame, raw: pd.DataFrame, split_result: split.SplitResult
    ) -> None:
        again = split.chronological_split(cleaned, split.verify_time_proxy(raw))
        assert again.train[schema.ENCOUNTER_ID].tolist() == (
            split_result.train[schema.ENCOUNTER_ID].tolist()
        )


class TestSeparability:
    def test_dataset_is_not_trivially_separable(self, split_result: split.SplitResult) -> None:
        """Inverted logic: a HIGH score here is the failure.

        On the synthetic symptom-to-disease datasets an unconstrained tree
        reaches ~100% AUC. A modest score is evidence the problem is genuinely
        hard, which is what makes it worth operating a model on.
        """
        check = separability_check(split_result.train, split_result.test)
        assert not check["alarm_triggered"], (
            f"test ROC-AUC {check['test_roc_auc']} exceeds "
            f"{check['alarm_threshold']} — investigate for leakage before "
            "trusting any downstream metric."
        )
