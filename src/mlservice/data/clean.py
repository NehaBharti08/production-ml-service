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


def parse_term_months(df: pd.DataFrame) -> pd.Series:
    """`` 36 months`` -> 36. Note the leading space in the raw data."""
    return pd.to_numeric(
        df["term"].astype("string").str.strip().str.replace(" months", "", regex=False),
        errors="coerce",
    )


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


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return the data plus its report.

    Order matters. The label is resolved first because maturity filtering needs
    it to report the before/after positive rate; leakage is dropped before
    sparsity so that a leaking column is never retained merely for being dense.
    """
    report = CleaningReport(rows_in=len(df))

    out = resolve_label(df, report)
    out = drop_immature_loans(out, report)
    out = drop_leaking_columns(out, report)
    out = drop_identifier_columns(out, report)
    out = drop_sparse_columns(out, report)

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
    "drop_identifier_columns",
    "drop_immature_loans",
    "drop_leaking_columns",
    "drop_sparse_columns",
    "parse_issue_date",
    "parse_term_months",
    "resolve_label",
]
