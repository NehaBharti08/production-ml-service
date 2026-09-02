"""Drift detection: data, prediction, and delayed-label.

Three kinds, because they answer different questions and arrive at different
times:

*   **Data drift** — has the input population changed? Available immediately.
*   **Prediction drift** — has the model's output distribution moved? Also
    immediate, and it moves before accuracy does.
*   **Delayed-label drift** — has actual performance degraded? The only one that
    directly answers "is the model still good", and the one that cannot be known
    for 30 days.

The ordering matters operationally. Data and prediction drift are *leading*
indicators — cheap, fast, and suggestive. Label drift is the *lagging* ground
truth. A responsible system watches the leading indicators to decide where to
look, and waits for the lagging one before concluding the model degraded.

Thresholds come from :mod:`mlservice.monitoring.null_calibration`, which derives
a per-feature bar from that feature's own churn between stable training windows.
See docs/DECISIONS/0007-drift-thresholds.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlservice.config import get_settings, get_thresholds
from mlservice.data import schema
from mlservice.logging_ import get_logger
from mlservice.monitoring.null_calibration import population_stability_index

log = get_logger(__name__)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    threshold: float
    breaching: bool
    reference_n: int
    current_n: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["psi"] = round(self.psi, 6)
        d["threshold"] = round(self.threshold, 6)
        return d


@dataclass
class DriftReport:
    window_start: str
    window_end: str
    window_rows: int
    reference_rows: int
    feature_schema_hash: str
    comparable: bool
    features: list[FeatureDrift] = field(default_factory=list)
    prediction: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def breaching(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.breaching]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_rows": self.window_rows,
            "reference_rows": self.reference_rows,
            "feature_schema_hash": self.feature_schema_hash,
            "comparable": self.comparable,
            "n_features": len(self.features),
            "n_breaching": len(self.breaching),
            "breaching_features": [f.feature for f in self.breaching],
            "features": [f.to_dict() for f in self.features],
            "prediction": self.prediction,
            "labels": self.labels,
            "notes": self.notes,
        }


def _feature_thresholds() -> dict[str, float]:
    per_feature = get_thresholds().model_dump()["drift"]["per_feature"]
    if per_feature.get("provenance") == "PLACEHOLDER":
        raise RuntimeError(
            "drift.per_feature is still PLACEHOLDER — run "
            "`uv run mlservice monitor calibrate` before detecting drift. "
            "Alerting on an unmeasured threshold is worse than not alerting."
        )
    thresholds: dict[str, float] = per_feature.get("thresholds", {})
    return thresholds


def detect_data_drift(
    reference: pd.DataFrame, current: pd.DataFrame, thresholds: dict[str, float] | None = None
) -> list[FeatureDrift]:
    """Per-feature PSI against the frozen reference window.

    Each feature is compared against **its own** calibrated threshold, not a
    shared constant. Measured in Phase 6: a uniform 0.10 would have flagged
    ``medical_specialty`` in 11 of 19 windows already accepted as stable,
    because its median churn (0.1196) is above the conventional bar.
    """
    thresholds = thresholds if thresholds is not None else _feature_thresholds()
    features = [
        c
        for c in (*schema.NUMERIC_FEATURES, *schema.CATEGORICAL_FEATURES)
        if c in reference.columns and c in current.columns
    ]

    results: list[FeatureDrift] = []
    for feature in features:
        threshold = thresholds.get(feature)
        if threshold is None:
            # A feature with no calibrated threshold is skipped loudly rather
            # than given a default. A silently-defaulted threshold is exactly
            # the arbitrary number this project avoids.
            log.warning("feature_has_no_calibrated_threshold", feature=feature)
            continue

        psi = population_stability_index(reference[feature], current[feature])
        results.append(
            FeatureDrift(
                feature=feature,
                psi=psi,
                threshold=threshold,
                breaching=psi > threshold,
                reference_n=len(reference),
                current_n=len(current),
            )
        )
    return results


def detect_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    decision_threshold: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Shift in the score distribution and in the alert rate.

    Two signals, because they fail differently:

    *   **Mean score** says the model is seeing a different population.
    *   **Alert rate** says downstream human workload is changing — which is
        what an operator actually feels first, and what a capacity plan depends
        on.

    A model can hold its mean while the alert rate moves sharply, if the
    distribution changes shape around the threshold. Watching only the mean
    would miss that entirely.
    """
    config = get_thresholds().model_dump()["drift"]["alert"]["prediction_drift"]
    relative_limit = threshold if threshold is not None else config["alert_rate_relative_change"]

    # decision_threshold is passed in from the LOADED MODEL, never read from
    # config. This is the second place that mistake appeared: an earlier version
    # read settings.model.decision_threshold, which still holds the 0.5
    # placeholder, so every score fell below it, the alert rate was 0 in every
    # window, and the signal was silently dead. The same bug hit the API in
    # Phase 3 — the threshold belongs to the model, not the environment.

    ref_mean = float(np.mean(reference_scores))
    cur_mean = float(np.mean(current_scores))
    ref_rate = float(np.mean(reference_scores >= decision_threshold))
    cur_rate = float(np.mean(current_scores >= decision_threshold))

    mean_change = (cur_mean - ref_mean) / ref_mean if ref_mean else 0.0
    rate_change = (cur_rate - ref_rate) / ref_rate if ref_rate else 0.0

    # Bootstrap CI on the reference mean, so "moved" is judged against sampling
    # variation rather than an absolute number pulled from nowhere.
    rng = np.random.default_rng(42)
    boots = [
        float(np.mean(rng.choice(reference_scores, len(reference_scores), replace=True)))
        for _ in range(500)
    ]
    lower, upper = float(np.percentile(boots, 0.5)), float(np.percentile(boots, 99.5))

    return {
        "reference_mean": round(ref_mean, 6),
        "current_mean": round(cur_mean, 6),
        "mean_relative_change": round(mean_change, 6),
        "mean_outside_reference_ci": not (lower <= cur_mean <= upper),
        "reference_ci_99": [round(lower, 6), round(upper, 6)],
        "reference_alert_rate": round(ref_rate, 6),
        "current_alert_rate": round(cur_rate, 6),
        "alert_rate_relative_change": round(rate_change, 6),
        "alert_rate_threshold": relative_limit,
        "breaching": abs(rate_change) > relative_limit or not (lower <= cur_mean <= upper),
    }


def detect_label_drift(
    matured: pd.DataFrame, baseline_pr_auc: float, threshold_se: float | None = None
) -> dict[str, Any]:
    """Performance on matured labels, against the test-set baseline.

    Expressed in **bootstrap standard errors, not raw PR-AUC points**. A 0.02
    drop means something entirely different at n=500 than at n=5000, and a fixed
    point threshold would page constantly on small windows while staying silent
    on large ones.
    """
    config = get_thresholds().model_dump()["drift"]["alert"]["label_drift"]
    limit = threshold_se if threshold_se is not None else config["pr_auc_drop_in_standard_errors"]
    minimum = config["min_matured_labels"]

    if len(matured) < minimum:
        return {
            "sufficient_labels": False,
            "n_matured": len(matured),
            "min_required": minimum,
            "note": (
                f"only {len(matured)} matured labels; {minimum} required before "
                "a performance claim is meaningful"
            ),
            "breaching": False,
        }

    from sklearn.metrics import average_precision_score

    y = matured["outcome_label"].to_numpy()
    scores = matured["predicted_proba"].to_numpy()

    if len(np.unique(y)) < 2:
        return {
            "sufficient_labels": False,
            "n_matured": len(matured),
            "note": "all matured labels are one class — PR-AUC undefined",
            "breaching": False,
        }

    observed = float(average_precision_score(y, scores))

    rng = np.random.default_rng(42)
    boots = []
    for _ in range(config["bootstrap_iterations"]):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(average_precision_score(y[idx], scores[idx])))
    standard_error = float(np.std(boots)) if boots else 0.0

    drop = baseline_pr_auc - observed
    drop_in_se = drop / standard_error if standard_error else 0.0

    return {
        "sufficient_labels": True,
        "n_matured": len(matured),
        "baseline_pr_auc": round(baseline_pr_auc, 6),
        "observed_pr_auc": round(observed, 6),
        "drop": round(drop, 6),
        "bootstrap_se": round(standard_error, 6),
        "drop_in_standard_errors": round(drop_in_se, 3),
        "threshold_se": limit,
        "breaching": drop_in_se > limit,
    }


def analyse_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
    decision_threshold: float | None = None,
    matured: pd.DataFrame | None = None,
    baseline_pr_auc: float | None = None,
) -> DriftReport:
    """Run all three drift checks over one monitoring window."""
    config = get_thresholds().model_dump()["drift"]["alert"]["data_drift"]
    notes: list[str] = []

    # Comparability first. Drift analysis across a feature-schema change
    # compares distributions that are not comparable, and the report would
    # silently lie rather than refuse.
    ref_hash = str(reference.attrs.get("feature_schema_hash", "unknown"))
    cur_hash = str(current.attrs.get("feature_schema_hash", ref_hash))
    comparable = ref_hash == cur_hash
    if not comparable:
        notes.append(
            f"SCHEMA MISMATCH: reference {ref_hash} vs current {cur_hash}. "
            "Drift numbers below are not comparable and must not be acted on."
        )

    if len(current) < config["min_window_rows"]:
        notes.append(
            f"window has {len(current)} rows, below the {config['min_window_rows']} "
            "minimum — PSI is unstable at this size and results are indicative only"
        )

    report = DriftReport(
        window_start=str(current[schema.ENCOUNTER_ID].min())
        if schema.ENCOUNTER_ID in current.columns
        else "",
        window_end=str(current[schema.ENCOUNTER_ID].max())
        if schema.ENCOUNTER_ID in current.columns
        else "",
        window_rows=len(current),
        reference_rows=len(reference),
        feature_schema_hash=cur_hash,
        comparable=comparable,
        notes=notes,
    )

    report.features = detect_data_drift(reference, current)

    if reference_scores is not None and current_scores is not None:
        if decision_threshold is None:
            raise ValueError(
                "decision_threshold is required for prediction drift and must come "
                "from the loaded model, not from config"
            )
        report.prediction = detect_prediction_drift(
            reference_scores, current_scores, decision_threshold
        )

    if matured is not None and baseline_pr_auc is not None:
        report.labels = detect_label_drift(matured, baseline_pr_auc)

    log.info(
        "drift_window_analysed",
        window_rows=len(current),
        n_features=len(report.features),
        n_breaching=len(report.breaching),
        breaching=[f.feature for f in report.breaching],
        prediction_breaching=report.prediction.get("breaching"),
        labels_breaching=report.labels.get("breaching"),
        comparable=comparable,
    )
    return report


def alert_state_from_counts(breaching_counts: list[int]) -> dict[str, Any]:
    """Decide whether consecutive windows constitute a confirmed alert.

    **Two-window confirmation is the whole point.** A single window breaching is
    noise: with ~43 features at a 99th-percentile threshold, roughly 0.4 features
    breach per window by chance alone. Requiring the same condition in
    consecutive windows is what stops the pager firing on sampling variation —
    and a pager that fires on noise is one people learn to ignore.

    Takes counts rather than reports so the retraining CLI can evaluate the same
    rule against drift reports it loaded from disk. The confirmation logic is the
    thing that decides whether a retrain fires; it must not exist twice.
    """
    config = get_thresholds().model_dump()["drift"]["alert"]["data_drift"]
    required_features = config["min_features_breaching"]
    required_windows = config["consecutive_windows"]

    recent = breaching_counts[-required_windows:]
    confirmed = len(recent) >= required_windows and all(c >= required_features for c in recent)

    return {
        "confirmed": confirmed,
        "required_features_per_window": required_features,
        "required_consecutive_windows": required_windows,
        "windows_examined": len(recent),
        "breaching_counts": recent,
        "reason": (
            f"{required_features}+ features breached in {required_windows} consecutive windows"
            if confirmed
            else "not confirmed — a single-window breach is treated as noise"
        ),
    }


def alert_state(reports: list[DriftReport]) -> dict[str, Any]:
    """:func:`alert_state_from_counts` over live report objects."""
    return alert_state_from_counts([len(r.breaching) for r in reports])


def load_reports(limit: int | None = None) -> list[dict[str, Any]]:
    """Load every saved drift window oldest-first, from both report shapes.

    Sorted by filename, which is a UTC timestamp — so lexical order *is*
    chronological order. That is a property of the naming scheme in
    :func:`save_report`, not an accident, and the two must stay in step.

    Replay runs are read as well as standalone reports, because they store
    their windows inline. Reading only ``drift_*.json`` meant the retraining
    trigger saw an empty history immediately after a replay that had detected
    drift, and answered "no confirmed drift" having examined nothing at all.
    A monitoring system that cannot tell *no evidence* from *evidence of no
    drift* reports reassuringly in exactly the case that should worry you.
    """
    import json

    reports_dir = get_settings().paths.reports
    out: list[dict[str, Any]] = []

    for path in sorted(reports_dir.glob("drift_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source"] = path.name
        out.append(payload)

    for path in sorted(reports_dir.glob("replay_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for window in payload.get("windows", []):
            report = dict(window.get("drift") or {})
            if not report:
                continue
            # Carried through so a caller can tell induced demo drift from the
            # real 1999->2008 shift. Losing that distinction here is how an
            # honest report turns into a misleading one two layers up.
            report.setdefault("drift_origin", window.get("drift_origin"))
            report["source"] = path.name
            out.append(report)

    if limit is not None:
        out = out[-limit:]
    return out


def breaching_counts(reports: list[dict[str, Any]]) -> list[int]:
    """Breaching-feature count per report, indexed straight rather than ``.get``.

    Deliberately raises ``KeyError`` on a report missing the key instead of
    defaulting to zero. The first version of the retraining trigger read
    ``report.get("breaching", [])`` — but :meth:`DriftReport.to_dict` emits
    ``n_breaching``; ``breaching`` is only a property on the live object. Every
    window therefore scored zero and the trigger reported "no confirmed drift"
    against a replay whose own alert state said ``confirmed: True``.

    A default is the wrong tool for a key that must exist. Silence there buys a
    tidy line of code and pays for it with a monitoring system that says the
    reassuring thing when it is broken.
    """
    return [int(r["n_breaching"]) for r in reports]


def save_report(report: DriftReport, path: Path | None = None) -> Path:
    import json

    settings = get_settings()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = path or (settings.paths.reports / f"drift_{stamp}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    log.info("drift_report_written", path=str(target))
    return target


__all__ = [
    "DriftReport",
    "FeatureDrift",
    "alert_state",
    "alert_state_from_counts",
    "analyse_window",
    "breaching_counts",
    "detect_data_drift",
    "detect_label_drift",
    "detect_prediction_drift",
    "load_reports",
    "save_report",
]
