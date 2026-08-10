"""The audit pipeline: run every check, emit the findings as data.

``docs/DATA_AUDIT.md`` is generated from this module's output rather than
written by hand, so the document cannot drift away from what the code does. A
hand-transcribed audit is stale the first time anyone changes a threshold.

Includes the separability sanity check, which is deliberately designed to
*fail* if the dataset is trivially separable — the failure mode that makes the
widely-circulated synthetic symptom-to-disease datasets worthless.
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
    time_proxy: dict[str, Any] = field(default_factory=dict)
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
        "unique_patients": int(df[schema.PATIENT_ID].nunique()),
        "encounters_per_patient": round(len(df) / df[schema.PATIENT_ID].nunique(), 3),
        "matches_expected_shape": (
            len(df) == schema.RAW_ROW_COUNT and len(df.columns) == schema.RAW_COLUMN_COUNT
        ),
    }


def profile_missingness(df: pd.DataFrame) -> dict[str, Any]:
    """Both kinds of absence, separated — they mean different things."""
    sentinel: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if df[col].dtype == object:
            n = int((df[col] == schema.MISSING_SENTINEL).sum())
            if n:
                sentinel[col] = {"n": n, "pct": round(n / len(df) * 100, 2)}

    native: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            native[col] = {
                "n": n,
                "pct": round(n / len(df) * 100, 2),
                "meaning": (
                    "test not ordered — a clinical decision, therefore informative"
                    if col in schema.NOT_MEASURED_COLUMNS
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
        "sentinel_missing": dict(sorted(sentinel.items(), key=lambda kv: -kv[1]["pct"])),
        "native_nan": dict(sorted(native.items(), key=lambda kv: -kv[1]["pct"])),
        "zero_variance_columns": zero_variance,
        "near_zero_variance_columns": near_zero,
    }


def profile_imbalance(df: pd.DataFrame) -> dict[str, Any]:
    counts = df[schema.TARGET].value_counts()
    y = clean.binarise_target(df)
    positive_rate = float(y.mean())
    return {
        "raw_classes": {str(k): int(v) for k, v in counts.items()},
        "binary_positive": int(y.sum()),
        "binary_negative": int((1 - y).sum()),
        "positive_rate": round(positive_rate, 6),
        "imbalance_ratio": round((1 - positive_rate) / positive_rate, 2),
        "majority_class_accuracy": round(1 - positive_rate, 6),
        "interpretation": (
            f"Predicting 'never readmitted' scores {1 - positive_rate:.2%} accuracy "
            "with zero recall. Accuracy is therefore not a meaningful headline "
            "metric; PR-AUC against the "
            f"{positive_rate:.2%} prevalence floor is."
        ),
    }


def separability_check(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Fit a deliberately unconstrained tree and hope it does *badly*.

    Inverted logic on purpose. On the synthetic symptom-to-disease datasets a
    tree like this reaches ~100% AUC, which reveals the data is trivially
    separable and worthless for demonstrating anything. Reaching a modest score
    here is evidence the problem is genuinely hard.
    """
    features = [c for c in train.columns if c not in ("target", *schema.IDENTIFIER_COLUMNS)]
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


def run_audit(raw: pd.DataFrame) -> tuple[AuditReport, split.SplitResult]:
    """Execute the full audit and return the findings plus the split."""
    report = AuditReport()

    report.dataset = profile_raw(raw)
    report.missingness = profile_missingness(raw)
    report.imbalance = profile_imbalance(raw)

    # Proxy verification runs on RAW data — it inspects the "?" sentinels and
    # native NaNs that cleaning removes.
    verification = split.verify_time_proxy(raw)
    report.time_proxy = {
        "passed": verification.passed,
        "claim": verification.claim,
        "n_trending": verification.n_trending,
        "n_signals": len(verification.signals),
        "threshold": split.MIN_TRENDING_SIGNALS,
        "criteria": {"min_abs_rho": split.MIN_ABS_RHO, "max_p_value": split.MAX_P_VALUE},
        "signals": [
            {
                "name": s.name,
                "first_decile": round(s.first_decile, 4),
                "last_decile": round(s.last_decile, 4),
                "delta": round(s.delta, 4),
                "spearman_rho": round(s.spearman_rho, 4),
                "p_value": round(s.p_value, 6),
                "trends": s.trends,
            }
            for s in verification.signals
        ],
        "discontinuity_rosiglitazone": split.detect_discontinuities(raw, "rosiglitazone"),
        "discontinuity_pioglitazone": split.detect_discontinuities(raw, "pioglitazone"),
    }

    cleaned, cleaning = clean.clean(raw)
    report.leakage = {
        "rows_in": cleaning.rows_in,
        "rows_out": cleaning.rows_out,
        "rows_removed": cleaning.rows_removed,
        "pct_removed": round(cleaning.rows_removed / cleaning.rows_in * 100, 2),
        "steps": cleaning.steps,
    }

    # Evidence for the censoring buffer is measured on the CLEANED frame,
    # because the effect only appears after first-encounter deduplication —
    # on all encounters the positive rate is flat across the period.
    report.censoring = split.censoring_buffer_evidence(cleaned)

    result = split.chronological_split(cleaned, verification)
    report.split = result.summary()

    report.separability = separability_check(result.train, result.test)
    report.baselines = [asdict(b) for b in baselines.evaluate_all(result.test)]
    report.subgroups = profile_subgroups(cleaned)

    return report, result


def load_raw() -> pd.DataFrame:
    settings = get_settings()
    path = settings.paths.data_raw / "diabetic_data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `uv run mlservice data download` first.")
    return pd.read_csv(path, low_memory=False)


__all__ = [
    "SEPARABILITY_ALARM_AUC",
    "AuditReport",
    "load_raw",
    "profile_imbalance",
    "profile_missingness",
    "profile_raw",
    "profile_subgroups",
    "run_audit",
    "separability_check",
]
