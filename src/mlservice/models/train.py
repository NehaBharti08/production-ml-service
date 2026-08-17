"""Train candidates, calibrate, evaluate, and register the champion.

Deliberately a *small* number of candidates. The project's thesis is that
operations are the product, so this stage exists to produce a defensible model
and stop — not to chase decimal places that the audit already showed are not
available (published ceiling ROC-AUC 0.65-0.68).

Candidate selection follows one rule: **the simplest model that is not
significantly worse wins.** Non-overlapping bootstrap confidence intervals are
required to prefer a more complex model, because a higher point estimate with
overlapping intervals is noise, and promoting on noise is how portfolios end up
with an unexplainable gradient-boosted ensemble.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from mlservice.config import get_settings
from mlservice.data import features, schema
from mlservice.logging_ import get_logger
from mlservice.models import calibration, evaluate, registry, subgroups

log = get_logger(__name__)


@dataclass
class Candidate:
    name: str
    estimator: Any
    rationale: str
    complexity_rank: int  # lower = simpler; ties broken toward simplicity


def candidates(seed: int) -> list[Candidate]:
    """The models actually tried, each with a reason for being here.

    XGBoost is deliberately absent. The plan admitted it only on audit evidence
    of non-linear structure worth the operational cost, and the audit produced
    the opposite: an unconstrained decision tree — which can represent arbitrary
    interactions — reached 0.519 test ROC-AUC. There is no non-linear signal
    going unexploited, so adding a gradient booster would buy dependency weight
    and opacity for nothing. Recorded in docs/DECISIONS/0005-model-selection.md.
    """
    return [
        Candidate(
            name="baseline_prevalence",
            estimator=DummyClassifier(strategy="prior"),
            rationale="Predicts the base rate for everyone. The Brier score to beat.",
            complexity_rank=0,
        ),
        Candidate(
            name="logistic_l2",
            estimator=LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=2000,
                # Weighting matters far more than the solver here: at a 7.6%
                # positive rate, an unweighted fit optimises almost entirely for
                # the negative class and produces a near-constant score.
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
            rationale="The intended champion. Linear, inspectable, cheap to serve.",
            complexity_rank=1,
        ),
        Candidate(
            name="logistic_l2_strong",
            estimator=LogisticRegression(
                penalty="l2",
                C=0.05,  # heavier regularisation
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
            rationale=(
                "Tests whether the wide one-hot matrix is overfitting. If this "
                "matches C=1.0, the extra capacity was not being used."
            ),
            complexity_rank=2,
        ),
        Candidate(
            name="random_forest_shallow",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=6,  # constrained: the audit showed deep trees do not generalise
                min_samples_leaf=50,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            rationale=(
                "Checks for interaction effects a linear model cannot capture. "
                "Must beat logistic with non-overlapping CIs to justify itself."
            ),
            complexity_rank=3,
        ),
    ]


@dataclass
class CandidateResult:
    name: str
    rationale: str
    complexity_rank: int
    threshold: float
    calibration_method: str
    val_pr_auc: float
    test: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = get_settings().paths.data_processed
    missing = [n for n in ("train", "val", "test") if not (paths / f"{n}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing splits {missing} in {paths}. Run `uv run mlservice data audit` first."
        )
    train, val, test = (pd.read_parquet(paths / f"{n}.parquet") for n in ("train", "val", "test"))
    return train, val, test


def train_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    schema_hash: str,
) -> tuple[CandidateResult, Any]:
    """Fit, calibrate on validation, choose a threshold, evaluate on test."""
    x_train, y_train = features.split_xy(train)
    x_val, y_val = features.split_xy(val)
    x_test, y_test = features.split_xy(test)

    with registry.run(candidate.name, tags={"candidate": candidate.name}) as active:
        pipeline = features.build_pipeline(train, candidate.estimator)
        pipeline.fit(x_train, y_train)

        # Calibrate on VALIDATION. Fitting the calibrator on test would measure
        # the model's fit to its own evaluation set.
        is_dummy = isinstance(candidate.estimator, DummyClassifier)
        cal_reports: dict[str, calibration.CalibrationReport] = {}
        if is_dummy:
            method = "uncalibrated"
            calibrated = pipeline
        else:
            method, cal_reports = calibration.choose_calibration_method(pipeline, x_val, y_val)
            calibrated = calibration.calibrate(pipeline, x_val, y_val, method=method)

        val_score = np.asarray(calibrated.predict_proba(x_val))[:, 1]
        val_pr_auc = float(average_precision_score(np.asarray(y_val), val_score))
        threshold, threshold_info = evaluate.choose_threshold(np.asarray(y_val), val_score)

        test_score = np.asarray(calibrated.predict_proba(x_test))[:, 1]
        report = evaluate.evaluate(np.asarray(y_test), test_score, threshold)
        test_cal = calibration.expected_calibration_error(np.asarray(y_test), test_score)
        test_cal.method = method

        sub = subgroups.evaluate_subgroups(
            test, np.asarray(y_test), test_score, threshold, schema.SUBGROUP_DIMENSIONS
        )

        mlflow.log_params(
            {
                "candidate": candidate.name,
                "complexity_rank": candidate.complexity_rank,
                "calibration_method": method,
                "threshold": round(threshold, 6),
                "target_recall": evaluate.TARGET_RECALL,
                "feature_schema_hash": schema_hash,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            }
        )
        mlflow.log_metrics(
            {
                "test_pr_auc": report.pr_auc.point,
                "test_pr_auc_lower": report.pr_auc.lower,
                "test_pr_auc_upper": report.pr_auc.upper,
                "test_roc_auc": report.roc_auc.point,
                "test_brier": report.brier.point,
                "test_ece": test_cal.ece,
                "test_mce": test_cal.mce,
                "test_recall": report.recall,
                "test_precision": report.precision,
                "test_lift": report.lift_over_prevalence,
                "test_flagged_rate": report.flagged_rate,
                "val_pr_auc": val_pr_auc,
                "worst_subgroup_recall_gap": sub.worst_recall_gap,
            }
        )
        mlflow.log_dict(report.to_dict(), "evaluation.json")
        mlflow.log_dict(sub.to_dict(), "subgroups.json")
        if cal_reports:
            mlflow.log_dict(
                {k: v.to_dict() for k, v in cal_reports.items()}, "calibration_val.json"
            )
        mlflow.log_dict(test_cal.to_dict(), "calibration_test.json")

        result = CandidateResult(
            name=candidate.name,
            rationale=candidate.rationale,
            complexity_rank=candidate.complexity_rank,
            threshold=threshold,
            calibration_method=method,
            val_pr_auc=val_pr_auc,
            test=report.to_dict(),
            calibration=test_cal.to_dict(),
            run_id=active.info.run_id,
        )
        result.test["subgroups"] = sub.to_dict()
        result.test["threshold_info"] = threshold_info

        log.info(
            "candidate_evaluated",
            candidate=candidate.name,
            pr_auc=round(report.pr_auc.point, 4),
            ci=[round(report.pr_auc.lower, 4), round(report.pr_auc.upper, 4)],
            ece=round(test_cal.ece, 4),
            recall=round(report.recall, 4),
            precision=round(report.precision, 4),
        )
        return result, calibrated


def select_champion(results: list[CandidateResult]) -> tuple[CandidateResult, dict[str, Any]]:
    """Simplest model that is not significantly worse than the best.

    Not "highest PR-AUC". A more complex model must clear the simpler one with
    **non-overlapping** confidence intervals to be preferred; otherwise the
    simpler model wins. On a task with this much irreducible error, point
    estimates move by more than the real differences between these models.
    """
    real = [r for r in results if not r.name.startswith("baseline")]
    best = max(real, key=lambda r: r.test["pr_auc"]["point"])

    def interval(r: CandidateResult) -> evaluate.Interval:
        d = r.test["pr_auc"]
        return evaluate.Interval(d["point"], d["lower"], d["upper"])

    best_ci = interval(best)
    # Anything whose interval overlaps the best is statistically indistinguishable.
    indistinguishable = [r for r in real if interval(r).overlaps(best_ci)]
    champion = min(indistinguishable, key=lambda r: r.complexity_rank)

    rationale = {
        "rule": "simplest candidate statistically indistinguishable from the best",
        "highest_point_estimate": best.name,
        "highest_pr_auc": best.test["pr_auc"],
        "chosen": champion.name,
        "chosen_pr_auc": champion.test["pr_auc"],
        "indistinguishable_from_best": [r.name for r in indistinguishable],
        "note": (
            f"{best.name} has the higher point estimate but its confidence "
            f"interval overlaps {champion.name}'s, so the difference is not "
            "evidence of improvement. The simpler model is preferred: it is "
            "cheaper to serve, easier to explain, and easier to debug at 2am."
        )
        if champion.name != best.name
        else "highest point estimate and simplest among indistinguishable candidates",
    }

    log.info("champion_selected", champion=champion.name, **{"rule": rationale["rule"]})
    return champion, rationale


def run_training(register_model: bool = True) -> dict[str, Any]:
    """Full Phase 2 pipeline."""
    settings = get_settings()
    train, val, test = load_splits()
    schema_hash = features.feature_schema_hash(train)

    uri, is_server = registry.setup_tracking()

    results: list[CandidateResult] = []
    fitted: dict[str, Any] = {}

    for candidate in candidates(settings.data.random_seed):
        result, model = train_candidate(candidate, train, val, test, schema_hash)
        results.append(result)
        fitted[candidate.name] = model

    champion, rationale = select_champion(results)
    champion_model = fitted[champion.name]

    # Reliability diagram for the model card, from the champion on test.
    x_test, y_test = features.split_xy(test)
    test_score = np.asarray(champion_model.predict_proba(x_test))[:, 1]
    reports = {"champion": calibration.expected_calibration_error(np.asarray(y_test), test_score)}
    reports["champion"].method = champion.calibration_method
    diagram = calibration.plot_reliability(
        reports,
        settings.paths.reports.parent / "docs" / "images" / "reliability_test.png",
        title=f"Reliability — {champion.name} on the held-out test split",
    )

    fallback = registry.save_local_fallback(champion_model)

    version = None
    if register_model:
        with registry.run(f"champion_{champion.name}", tags={"role": "champion"}) as active:
            mlflow.sklearn.log_model(
                champion_model,
                name="model",
                input_example=x_test.head(3),
                # MLflow 3.x serialises with skops, which refuses to load types
                # not on an allow-list. The calibrator's internal class is one
                # of them. Naming it explicitly is preferable to switching the
                # whole artifact to cloudpickle: this keeps the safe loader and
                # states exactly which type is being trusted and why.
                skops_trusted_types=["sklearn.calibration._CalibratedClassifier"],
            )
            mlflow.log_params(
                {
                    "champion": champion.name,
                    "threshold": round(champion.threshold, 6),
                    "calibration_method": champion.calibration_method,
                    "feature_schema_hash": schema_hash,
                }
            )
            mlflow.log_dict(rationale, "selection_rationale.json")
            try:
                version = registry.register(
                    f"runs:/{active.info.run_id}/model",
                    settings.model.name,
                    alias=settings.model.serving_alias,
                )
            except Exception as exc:
                # The file store supports the registry in recent MLflow, but not
                # every backend does. A registry failure must not lose the run.
                log.warning(
                    "model_registration_failed",
                    error=str(exc)[:200],
                    consequence="model is logged and the local fallback exists; "
                    "re-register once the MLflow server is running",
                )

    summary = {
        "tracking_uri": uri,
        "tracking_is_server": is_server,
        "feature_schema_hash": schema_hash,
        "champion": champion.name,
        "champion_threshold": champion.threshold,
        "calibration_method": champion.calibration_method,
        "selection_rationale": rationale,
        "registered_version": getattr(version, "version", None),
        "local_fallback": str(fallback),
        "reliability_diagram": str(diagram),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "candidates": [
            {
                "name": r.name,
                "rationale": r.rationale,
                "complexity_rank": r.complexity_rank,
                "calibration_method": r.calibration_method,
                "threshold": round(r.threshold, 6),
                "val_pr_auc": round(r.val_pr_auc, 6),
                "calibration": r.calibration,
                **r.test,
            }
            for r in results
        ],
    }

    out = settings.paths.reports / "training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("training_complete", champion=champion.name, summary=str(out))

    return summary


__all__ = [
    "Candidate",
    "CandidateResult",
    "candidates",
    "load_splits",
    "run_training",
    "select_champion",
    "train_candidate",
]
