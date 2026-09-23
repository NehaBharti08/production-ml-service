"""Cleaning: resolve the label, remove leakage, and honour right-censoring.

Three of these steps change the answer materially, and each is here because the
audit found something rather than because a tutorial said to.

**Leakage.** Thirty-one columns in this file are knowable only *after* the loan
has run. ``recoveries > 0`` means the loan defaulted; ``total_rec_prncp`` is how
much principal came back. They look like ordinary loan attributes, which is
precisely why they are enumerated in config rather than left to judgment. A
model trained with them reports near-perfect accuracy and is worthless.

**Right-censoring.** 38.9% of loans are still ``Current``. A loan issued in
December 2018 has not had time to default, so its absence of default is not
evidence of repayment. Treating ``Current`` as a negative would teach the model
that recent loans are safe — which is not a fact about lending, it is a fact
about when the file was exported.

**Unresolved is not negative.** Only terminal statuses become labels. ``Late``
and ``In Grace Period`` are dropped rather than guessed at: a loan 90 days late
will *probably* charge off, but "probably" is not a label, and encoding the
guess would bake an assumption into the ground truth where nobody could see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mlservice.config import get_settings
from mlservice.data import schema
from mlservice.logging_ import get_logger

log = get_logger(__name__)


@dataclass
class CleaningReport:
    """What cleaning actually did — the raw material for the audit document."""

    rows_in: int = 0
    rows_out: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, **detail: Any) -> None:
        self.steps.append({"step": step, **detail})
        log.info(f"clean_{step}", **detail)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out


def parse_issue_date(df: pd.DataFrame) -> pd.Series:
    """``issue_d`` ("Dec-2018") -> a real timestamp.

    Parsed with an explicit format rather than letting pandas infer. Inference
    on 2.26M rows is slow, and worse, it can silently switch interpretation
    partway through a column.
    """
    settings = get_settings()
    return pd.to_datetime(
        df[settings.data.time_column], format=settings.data.time_format, errors="coerce"
    )


def order_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort rows chronologically. The only sanctioned way to do it.

    ``issue_d`` is a **string** ("Dec-2018"), so ``sort_values(TIME_COLUMN)``
    orders it alphabetically — Apr, Aug, Dec, Feb, Jan, Jul, Jun, Mar, May,
    Nov, Oct, Sep — and the result looks chronological to every check short of
    parsing it. Two monitoring modules did precisely that, which put all 64
    drift thresholds and the whole replay on windows that were adjacent in the
    alphabet rather than in time.

    `chronological_split` never had the bug because it parsed first. The
    difference was one line, in a codebase where three call sites needed the
    same ordering — so the ordering lives here now, and the call sites cannot
    choose the wrong one.
    """
    dates = parse_issue_date(frame)
    if dates.isna().any():
        raise ValueError(
            f"{int(dates.isna().sum())} rows have an unparseable "
            f"{get_settings().data.time_column}; ordering them by time would "
            "silently place them wherever NaT happens to sort"
        )
    return (
        frame.assign(_ordered_at=dates)
        .sort_values("_ordered_at", kind="stable")
        .drop(columns="_ordered_at")
    )


def parse_term_months(df: pd.DataFrame) -> pd.Series:
    """`` 36 months`` -> 36. Note the leading space in the raw data."""
    return pd.to_numeric(
        df["term"].astype("string").str.strip().str.replace(" months", "", regex=False),
        errors="coerce",
    )


def drop_non_loan_rows(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove footer rows that are not loans at all.

    The CSV ends with 33 summary lines — text like
    ``"Total amount funded in policy code 1: 6417608175"`` sitting in the
    ``id`` column, with every other field empty. They are an artefact of how
    the file was exported, not records.

    They would be removed incidentally anyway, because their ``loan_status``
    is null and the label filter drops them. Removing them *explicitly* is the
    point: a row that is not a loan should be excluded by a rule that says so,
    and counted, rather than disappearing into a filter that exists for an
    unrelated reason. Otherwise the raw row count is quietly wrong and nobody
    can tell why.
    """
    usable = df[schema.TIME_COLUMN].notna()
    dropped = int((~usable).sum())
    out = df.loc[usable].copy()

    report.record(
        "drop_non_loan_rows",
        reason="export footer lines, not records — no issue date, no status",
        rows_dropped=dropped,
        example=(str(df.loc[~usable, "id"].dropna().iloc[0])[:80] if dropped else None),
        rows_after=len(out),
    )
    return out


def resolve_label(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Keep only terminal outcomes and turn them into a binary target.

    Nine statuses exist; five are terminal. The other four — Current, Late
    (16-30), Late (31-120), In Grace Period — describe loans still in flight.
    Dropping them costs rows and buys a label that means what it says.
    """
    settings = get_settings()
    bad = set(settings.data.loan_status_bad)
    good = set(settings.data.loan_status_good)

    status = df[settings.data.target_column]
    terminal = status.isin(bad | good)

    unresolved = status[~terminal].value_counts().to_dict()
    out = df.loc[terminal].copy()
    out["target"] = status.loc[terminal].isin(bad).astype("int8")

    report.record(
        "resolve_label",
        reason="only terminal outcomes can be labelled; in-flight loans are not negatives",
        rows_before=len(df),
        rows_after=len(out),
        rows_dropped=len(df) - len(out),
        dropped_by_status={str(k): int(v) for k, v in unresolved.items()},
        positive_rate=round(float(out["target"].mean()), 6),
    )
    return out


def drop_immature_loans(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Keep only loans whose full term has elapsed by the observation date.

    **The most consequential rule in the pipeline.** Resolution is a function of
    age, so without this the label is contaminated by export date:

        2007-2013   100.0% of loans resolved
        2016         67.5%
        2017         38.2%
        2018         11.4%   <- default rate also FALLS to 15.8%

    That falling default rate is not lending improving. It is survivorship: in a
    2018 cohort the only loans that have reached a terminal state are the ones
    that resolved fastest, and they are not representative.

    The rule is stated, not tuned — issue date plus term must be at or before
    the last observation in the file. A cut chosen to make the numbers look
    better would be exactly the kind of thing this project exists to avoid.
    """
    settings = get_settings()
    if not settings.data.require_matured_term:
        report.record("drop_immature_loans", skipped=True, reason="disabled in config")
        return df

    observation_end = pd.Timestamp(settings.data.observation_end)
    issued = parse_issue_date(df)
    term = parse_term_months(df)
    matures = issued + pd.to_timedelta(term * 30.44, unit="D")

    mature = matures <= observation_end
    kept = df.loc[mature.fillna(False)].copy()

    by_year_before = issued.dt.year.value_counts().sort_index()
    by_year_after = parse_issue_date(kept).dt.year.value_counts().sort_index()

    report.record(
        "drop_immature_loans",
        rule="issue_d + term <= observation_end",
        observation_end=str(observation_end.date()),
        rows_before=len(df),
        rows_after=len(kept),
        rows_dropped=int(len(df) - len(kept)),
        positive_rate_before=round(float(df["target"].mean()), 6),
        positive_rate_after=round(float(kept["target"].mean()), 6),
        latest_issue_kept=str(parse_issue_date(kept).max().date()),
        years_before=len(by_year_before),
        years_after=len(by_year_after),
    )
    return kept


def drop_leaking_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove every column knowable only after origination.

    Enumerated in config rather than inferred. A heuristic ("drop anything
    correlated with the target") would also drop legitimately predictive
    features, and would not explain itself to a reviewer.

    ``recoveries`` is the clearest case: it is money clawed back after a
    default, so a non-zero value *is* the label. But ``last_fico_range_high`` is
    just as fatal and far less obvious — it is the borrower's credit score
    measured after the loan has already gone wrong.
    """
    settings = get_settings()
    leaking = [c for c in settings.data.post_origination_columns if c in df.columns]
    out = df.drop(columns=leaking)

    report.record(
        "drop_leaking_columns",
        reason="knowable only after origination; several encode the outcome directly",
        columns_dropped=len(leaking),
        columns=sorted(leaking),
        columns_remaining=len(out.columns),
    )
    return out


def drop_identifier_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove ids, urls and free text.

    Three different reasons, kept distinct: identifiers carry no signal, urls
    are constant, and the free-text fields (``desc``, ``emp_title``, ``title``)
    would turn a deliberately boring tabular model into an NLP project. That is
    a legitimate thing to build — it is not what this project is demonstrating.

    ``zip_code`` goes too. It is truncated to three digits, so it is a coarse
    geography already covered by ``addr_state``, and it invites a fair-lending
    problem for no modelling gain.
    """
    settings = get_settings()
    dropped = [c for c in settings.data.identifier_columns if c in df.columns]
    out = df.drop(columns=dropped)

    report.record(
        "drop_identifier_columns",
        reason="no signal, constant, or free text outside this project's scope",
        columns=sorted(dropped),
        columns_remaining=len(out.columns),
    )
    return out


def drop_sparse_columns(
    df: pd.DataFrame, report: CleaningReport, threshold: float = 0.5
) -> pd.DataFrame:
    """Drop columns more than ``threshold`` missing.

    Lending Club added fields over the years, so a column introduced in 2015 is
    absent for every loan before it. That missingness is *structural* — it
    encodes the era, not the borrower — and under a chronological split a model
    would learn it as a proxy for time. Exactly the trap ``payer_code`` set in
    the medical version of this project.
    """
    missing = df.isna().mean()
    sparse = sorted(missing[missing > threshold].index)
    out = df.drop(columns=sparse)

    report.record(
        "drop_sparse_columns",
        reason="structurally missing by era; under a chronological split these proxy for time",
        threshold=threshold,
        columns_dropped=len(sparse),
        worst=[
            {"column": c, "missing_pct": round(float(missing[c]) * 100, 2)} for c in sparse[:10]
        ],
        columns_remaining=len(out.columns),
    )
    return out


def derive_credit_history(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """``earliest_cr_line`` -> months of credit history at origination.

    As shipped it is a date string with 691 distinct values. Treated as a
    categorical it would explode into 691 dummy columns, most of them seen a
    handful of times, and it would encode the *era* rather than the borrower.

    As a duration relative to the loan's own issue date it becomes what it
    actually means: how long this person has had credit. That is a genuine
    risk feature, and it is era-invariant, which matters under a chronological
    split.
    """
    issued = parse_issue_date(df)
    opened = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    months = ((issued - opened).dt.days / 30.44).round(1)

    out = df.drop(columns=["earliest_cr_line"])
    out["credit_history_months"] = months

    report.record(
        "derive_credit_history",
        reason="a date with 691 levels becomes one era-invariant duration",
        median_months=float(months.median()),
        negative_values=int((months < 0).sum()),
        missing=int(months.isna().sum()),
    )
    return out


def drop_target_source(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove the raw status column the label was derived from.

    **This is the target itself.** ``loan_status`` maps one-to-one onto
    ``target`` by construction, so leaving it in the feature frame hands the
    model the answer. It survived the first version of this pipeline because
    the leakage list was written from the data dictionary — which describes
    ``loan_status`` as a loan attribute, because it is one, right up until you
    make it the label.

    The lesson is that a leakage list cannot be written once: deriving a label
    creates a new leak that did not exist before.
    """
    settings = get_settings()
    column = settings.data.target_column
    if column not in df.columns:
        return df

    perfectly_predicts = bool(df.groupby(column)["target"].nunique().max() == 1)
    out = df.drop(columns=[column])

    report.record(
        "drop_target_source",
        reason="the column the label was derived from is the label",
        column=column,
        perfectly_predicts_target=perfectly_predicts,
        columns_remaining=len(out.columns),
    )
    return out


def drop_constant_and_redundant(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop zero-information and perfectly-collinear columns.

    Constants carry nothing: ``policy_code`` is 1 for every row and
    ``disbursement_method`` is "Cash" for every row. They cost a column in the
    serving contract and buy nothing.

    Redundancy is the subtler half. ``fico_range_low`` and ``fico_range_high``
    correlate at **exactly 1.0** — they are the two ends of a fixed-width band,
    so one is the other plus a constant. Keeping both gives a linear model two
    identical columns to split a coefficient across, which inflates its
    variance and makes the fitted weights unreadable. ``funded_amnt`` and
    ``funded_amnt_inv`` sit at 0.9989 against ``loan_amnt`` for the same
    reason: on a funded loan, the amount requested is the amount funded.
    """
    constant = sorted(c for c in df.columns if c != "target" and len(df[c].unique()) <= 1)
    redundant = [
        c for c in ("fico_range_high", "funded_amnt", "funded_amnt_inv") if c in df.columns
    ]

    out = df.drop(columns=constant + redundant)
    report.record(
        "drop_constant_and_redundant",
        reason="zero information, or perfectly collinear with a column we keep",
        constant=constant,
        redundant=redundant,
        kept_instead={"fico_range_high": "fico_range_low", "funded_amnt": "loan_amnt"},
        columns_remaining=len(out.columns),
    )
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return the data plus its report.

    Order matters. The label is resolved first because maturity filtering needs
    it to report the before/after positive rate; leakage is dropped before
    sparsity so that a leaking column is never retained merely for being dense.
    """
    report = CleaningReport(rows_in=len(df))

    out = drop_non_loan_rows(df, report)
    out = resolve_label(out, report)
    out = drop_immature_loans(out, report)
    out = drop_target_source(out, report)
    out = drop_leaking_columns(out, report)
    out = drop_identifier_columns(out, report)
    out = drop_sparse_columns(out, report)
    out = derive_credit_history(out, report)
    out = drop_constant_and_redundant(out, report)

    report.rows_out = len(out)
    report.record(
        "complete",
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        rows_removed=report.rows_removed,
        pct_removed=round(report.rows_removed / report.rows_in * 100, 2),
        positive_rate=round(float(out["target"].mean()), 6),
        columns=len(out.columns),
    )
    return out, report


__all__ = [
    "CleaningReport",
    "clean",
    "derive_credit_history",
    "drop_constant_and_redundant",
    "drop_identifier_columns",
    "drop_immature_loans",
    "drop_leaking_columns",
    "drop_non_loan_rows",
    "drop_sparse_columns",
    "drop_target_source",
    "parse_issue_date",
    "parse_term_months",
    "resolve_label",
]
