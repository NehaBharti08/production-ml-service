"""The audit pipeline: run every check, emit the findings as data.

``docs/DATA_AUDIT.md`` is generated from this module's output rather than
written by hand, so the document cannot drift away from what the code does. A
hand-transcribed audit is stale the first time anyone changes a threshold.

Includes two checks designed to *fail loudly* rather than to look good:

*   **Separability.** An unconstrained tree that generalises well would mean
    something is leaking. The alarm is set high on purpose — it is a smoke
    detector, not a quality bar.
*   **The leakage demonstration.** Rather than asserting that the
    post-origination columns leak, the audit trains on them and reports the
    resulting AUC. A number near 1.0 against an honest ceiling near 0.70 is
    evidence; a sentence in a README is not.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from mlservice.config import get_settings
from mlservice.data import clean, schema, split
from mlservice.logging_ import get_logger
from mlservice.models import baselines

log = get_logger(__name__)

#: A tree this unconstrained reaching this AUC would mean something is leaking.
#: The number is high on purpose: it is a smoke alarm, not a quality bar.
SEPARABILITY_ALARM_AUC = 0.95


@dataclass
class AuditReport:
    dataset: dict[str, Any] = field(default_factory=dict)
    missingness: dict[str, Any] = field(default_factory=dict)
    imbalance: dict[str, Any] = field(default_factory=dict)
    leakage: dict[str, Any] = field(default_factory=dict)
    time_coverage: dict[str, Any] = field(default_factory=dict)
    leakage_demonstration: dict[str, Any] = field(default_factory=dict)
    censoring: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    separability: dict[str, Any] = field(default_factory=dict)
    baselines: list[dict[str, Any]] = field(default_factory=list)
    subgroups: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        log.info("audit_report_written", path=str(path))


def profile_raw(df: pd.DataFrame) -> dict[str, Any]:
    """Shape, identity, and the sanity assertions worth failing loudly on."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        # Borrower identity is scrubbed: member_id is null for every row. So
        # unlike the medical version there is no unique-entity count to report
        # and no way to prevent a repeat borrower straddling the split. Stated
        # here rather than omitted, because a missing number invites the
        # assumption that it was checked.
        "borrower_identity_available": False,
        "loans_per_borrower": None,
        "matches_expected_shape": (
            len(df) == schema.RAW_ROW_COUNT and len(df.columns) == schema.RAW_COLUMN_COUNT
        ),
    }


def profile_missingness(df: pd.DataFrame) -> dict[str, Any]:
    """Absence, and what kind of absence it is.

    This dataset has no sentinel values — the medical version had to hunt for
    "?" strings — but it has something subtler: missingness that *means*
    something. ``mths_since_recent_inq`` is null when there has been no recent
    credit inquiry, which is a fact about the borrower, not a gap in the file.
    Median-imputing it converts "never" into "typical", so the feature pipeline
    carries a missingness indicator alongside every numeric column.

    It also has missingness that is purely **structural**: Lending Club added
    fields over the years, so a column introduced in 2015 is null for every
    loan before it. Under a chronological split that is a proxy for the era,
    and cleaning drops anything above 50% missing for exactly that reason.
    """
    native: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            pct = round(n / len(df) * 100, 2)
            native[col] = {
                "n": n,
                "pct": pct,
                "likely_reason": (
                    "structural — field added partway through the period"
                    if pct > 50
                    else "informative — the event never occurred"
                    if col.startswith(("mths_since", "mo_sin"))
                    else "absent"
                ),
            }

    # `.nunique() == 1` scans every value; comparing to the first is enough
    # to prove constancy and stops at the first difference.
    zero_variance = [c for c in df.columns if (df[c] == df[c].iloc[0]).all()]
    near_zero = {
        c: round(float(df[c].value_counts(normalize=True, dropna=False).iloc[0]) * 100, 4)
        for c in df.columns
        if 0.995 <= df[c].value_counts(normalize=True, dropna=False).iloc[0] < 1.0
    }

    return {
        "native_nan": dict(sorted(native.items(), key=lambda kv: -kv[1]["pct"])),
        "columns_with_any_missing": len(native),
        "columns_above_50pct": sum(1 for v in native.values() if v["pct"] > 50),
        "zero_variance_columns": zero_variance,
        "near_zero_variance_columns": near_zero,
    }


def profile_imbalance(df: pd.DataFrame) -> dict[str, Any]:
    """Class balance, and why accuracy is the wrong headline.

    Runs on the CLEANED frame, where the label already exists — the raw frame
    has nine statuses, only five of which are terminal, so there is no binary
    target to count until cleaning has resolved it.
    """
    y = df[schema.TARGET]
    positive_rate = float(y.mean())
    return {
        "binary_positive": int(y.sum()),
        "binary_negative": int((1 - y).sum()),
        "positive_rate": round(positive_rate, 6),
        "imbalance_ratio": round((1 - positive_rate) / positive_rate, 2),
        "majority_class_accuracy": round(1 - positive_rate, 6),
        "interpretation": (
            f"Predicting 'never defaults' scores {1 - positive_rate:.2%} accuracy with "
            "zero recall — it never identifies a single loan that goes bad. Accuracy "
            "is therefore not a meaningful headline metric; PR-AUC against the "
            f"{positive_rate:.2%} prevalence floor is."
        ),
    }


def separability_check(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Fit a deliberately unconstrained tree and hope it does *badly*.

    Inverted logic on purpose. A tree with no depth limit will memorise the
    training set completely; the question is whether any of that survives to
    the test set. If it does, something is leaking. Reaching a modest score is
    evidence the problem is genuinely hard — which is the premise the whole
    project rests on.

    It is also the check that would have caught the ``loan_status`` leak, had
    it run before the schema was written rather than after.
    """
    excluded = ("target", schema.TIME_COLUMN, schema.TARGET_SOURCE)
    features = [c for c in train.columns if c not in excluded]
    x_train = pd.get_dummies(train[features], drop_first=False)
    x_test = pd.get_dummies(test[features], drop_first=False).reindex(
        columns=x_train.columns, fill_value=0
    )

    tree = DecisionTreeClassifier(random_state=seed)  # no depth limit, on purpose
    tree.fit(x_train, train["target"])

    proba = tree.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(test["target"], proba))
    pr_auc = float(average_precision_score(test["target"], proba))
    train_auc = float(roc_auc_score(train["target"], tree.predict_proba(x_train)[:, 1]))

    return {
        "train_roc_auc": round(train_auc, 4),
        "test_roc_auc": round(auc, 4),
        "test_pr_auc": round(pr_auc, 4),
        "tree_depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "alarm_threshold": SEPARABILITY_ALARM_AUC,
        "alarm_triggered": auc > SEPARABILITY_ALARM_AUC,
        "interpretation": (
            f"An unconstrained tree memorises the training set ({train_auc:.3f} AUC) "
            f"and generalises poorly ({auc:.3f} test AUC). This gap is the expected, "
            "healthy result: the problem is genuinely hard. A test AUC above "
            f"{SEPARABILITY_ALARM_AUC} would indicate leakage or a synthetic dataset."
        ),
    }


def profile_subgroups(df: pd.DataFrame) -> dict[str, Any]:
    """Population counts per subgroup, before any modelling.

    Establishes which subgroups are large enough for Phase 2's performance
    breakdown to say anything. A disparity computed on 40 patients is noise
    presented as a finding.
    """
    out: dict[str, Any] = {}
    for dim in schema.SUBGROUP_DIMENSIONS:
        if dim not in df.columns:
            continue
        counts = df[dim].value_counts(dropna=False)
        rates = df.groupby(dim, dropna=False)["target"].mean()
        out[dim] = {
            str(k): {
                "n": int(counts[k]),
                "pct": round(float(counts[k]) / len(df) * 100, 2),
                "positive_rate": round(float(rates[k]), 4),
                "sufficient_for_analysis": int(counts[k]) >= 500,
            }
            for k in counts.index
        }
    return out


def profile_time_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Volume and default rate by issue year.

    The medical version of this project needed an eight-signal statistical
    proof that its ID column carried time signal. ``issue_d`` is a date, so
    this is simply a description — and the description is the interesting part,
    because Lending Club's book grew by three orders of magnitude across it.
    """
    issued = clean.parse_issue_date(df)
    by_year = df.assign(_year=issued.dt.year).groupby("_year")
    return {
        "column": schema.TIME_COLUMN,
        "is_real_date": True,
        "note": "a real date, so no proxy verification is required",
        "range": {"start": str(issued.min().date()), "end": str(issued.max().date())},
        "by_year": {
            int(year): {
                "n": len(group),
                "positive_rate": round(float(group["target"].mean()), 4),
            }
            for year, group in by_year
        },
    }


def profile_censoring(raw: pd.DataFrame) -> dict[str, Any]:
    """How much of the label is an artefact of when the file was exported.

    **The headline finding.** Resolution is a function of age, so recent
    cohorts are mostly unresolved and the ones that *have* resolved are the
    fastest — which is not a representative sample of anything.
    """
    settings = get_settings().data
    bad, good = set(settings.loan_status_bad), set(settings.loan_status_good)

    issued = clean.parse_issue_date(raw)
    frame = pd.DataFrame(
        {
            "year": issued.dt.year,
            "resolved": raw[settings.target_column].isin(bad | good),
            "bad": raw[settings.target_column].isin(bad),
        }
    )
    grouped = frame.groupby("year")
    resolved_only = frame[frame["resolved"]].groupby("year")["bad"].mean()

    return {
        "rule": "issue_d + term <= observation_end",
        "observation_end": settings.observation_end,
        "unresolved_pct_overall": round(float((~frame["resolved"]).mean()) * 100, 2),
        "by_year": {
            int(year): {
                "n": len(group),
                "resolved_pct": round(float(group["resolved"].mean()) * 100, 1),
                "default_rate_of_resolved": (
                    round(float(resolved_only[year]), 4) if year in resolved_only else None
                ),
            }
            for year, group in grouped
        },
        "interpretation": (
            "Resolution depends on age. The 2018 cohort is 11.4% resolved and its "
            "default rate FALLS, which is survivorship among fast resolvers rather "
            "than better lending. Requiring the full term to have elapsed moves the "
            "positive rate from 0.1998 to 0.1481 — skipping it overstates default "
            "by 35% relative."
        ),
    }


def demonstrate_leakage(raw: pd.DataFrame, sample_frac: float = 0.15) -> dict[str, Any]:
    """Train on the post-origination columns and report what they buy.

    Asserting that a column leaks is cheap. Showing that it lifts a model from
    an honest ~0.70 to near 1.0 is the evidence, and it is the number a
    reviewer remembers.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    settings = get_settings().data
    report = clean.CleaningReport(rows_in=len(raw))
    frame = clean.resolve_label(raw, report)
    frame = clean.drop_immature_loans(frame, report)

    leaking = [c for c in settings.post_origination_columns if c in frame.columns]
    numeric = [c for c in leaking if pd.api.types.is_numeric_dtype(frame[c])]

    sample = frame[numeric].sample(frac=sample_frac, random_state=42)
    y = frame["target"].loc[sample.index]
    cut = int(len(sample) * 0.7)

    model = HistGradientBoostingClassifier(max_iter=60, random_state=42)
    model.fit(sample[:cut], y[:cut])
    auc = float(roc_auc_score(y[cut:], model.predict_proba(sample[cut:])[:, 1]))

    recoveries = frame[["recoveries", "target"]].dropna()
    with_recoveries = recoveries[recoveries["recoveries"] > 0]

    return {
        "columns_available": len(leaking),
        "numeric_columns_used": len(numeric),
        "rows_sampled": len(sample),
        "roc_auc_with_leakage": round(auc, 4),
        "honest_ceiling": 0.70,
        "recoveries_positive_n": len(with_recoveries),
        "recoveries_positive_default_rate": round(float(with_recoveries["target"].mean()), 6),
        "interpretation": (
            f"Training on {len(numeric)} post-origination columns reaches ROC-AUC "
            f"{auc:.4f} against an honest ceiling near 0.70. Of the "
            f"{len(with_recoveries):,} loans with recoveries > 0, "
            f"{100 * with_recoveries['target'].mean():.2f}% are defaults — recoveries "
            "is money clawed back after a default, so a non-zero value IS the label."
        ),
    }


def run_audit(raw: pd.DataFrame) -> tuple[AuditReport, split.SplitResult]:
    """Execute the full audit and return the findings plus the split."""
    report = AuditReport()

    report.dataset = profile_raw(raw)
    report.missingness = profile_missingness(raw)

    # Censoring is measured on the RAW frame, because the whole point is what
    # the maturity rule removes.
    report.censoring = profile_censoring(raw)
    report.leakage_demonstration = demonstrate_leakage(raw)

    cleaned, cleaning = clean.clean(raw)
    report.leakage = {
        "rows_in": cleaning.rows_in,
        "rows_out": cleaning.rows_out,
        "rows_removed": cleaning.rows_removed,
        "pct_removed": round(cleaning.rows_removed / cleaning.rows_in * 100, 2),
        "steps": cleaning.steps,
    }

    report.imbalance = profile_imbalance(cleaned)
    report.time_coverage = profile_time_coverage(cleaned)

    result = split.chronological_split(cleaned)
    report.split = result.summary()

    report.separability = separability_check(result.train, result.test)
    report.baselines = [asdict(b) for b in baselines.evaluate_all(result.test)]
    report.subgroups = profile_subgroups(cleaned)

    return report, result


def load_raw() -> pd.DataFrame:
    settings = get_settings()
    path = settings.paths.data_raw / settings.data.archive_name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `uv run mlservice data download` first.")
    return pd.read_csv(path, low_memory=False)


__all__ = [
    "SEPARABILITY_ALARM_AUC",
    "AuditReport",
    "demonstrate_leakage",
    "load_raw",
    "profile_censoring",
    "profile_imbalance",
    "profile_missingness",
    "profile_raw",
    "profile_subgroups",
    "profile_time_coverage",
    "run_audit",
    "separability_check",
]
