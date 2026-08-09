"""Trivial baselines — the floor every later model must clear.

Without these, "PR-AUC 0.22" is an uninterpretable number. With them it becomes
"0.22 against a 0.11 prevalence floor", which is a claim someone can check.

The majority-class baseline exists to make one point concrete: it scores ~89%
accuracy while catching **zero** readmissions. Any project reporting accuracy on
this task is reporting that number. Including it in the audit is the clearest
possible argument for why PR-AUC is the headline metric here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)

from mlservice.logging_ import get_logger

log = get_logger(__name__)

BOOTSTRAP_ITERATIONS = 1_000
CI_LEVEL = 0.95


@dataclass
class BaselineResult:
    name: str
    description: str
    accuracy: float
    recall: float
    pr_auc: float
    roc_auc: float | None
    brier: float
    pr_auc_ci: tuple[float, float] | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    level: float = CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for PR-AUC.

    Every later "improvement" is compared against these intervals. Without them
    a 0.01 PR-AUC gain looks like progress when it is noise — and on an 11%
    positive class the sampling variance is not small.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # degenerate resample
            continue
        scores.append(float(average_precision_score(y_true[idx], y_score[idx])))
    lower = float(np.percentile(scores, (1 - level) / 2 * 100))
    upper = float(np.percentile(scores, (1 + level) / 2 * 100))
    return lower, upper


def majority_class(y_true: np.ndarray) -> BaselineResult:
    """Always predict the majority class — the accuracy trap, made explicit."""
    pred = np.zeros_like(y_true)
    prevalence = float(y_true.mean())
    score = np.full_like(y_true, prevalence, dtype=float)

    return BaselineResult(
        name="majority_class",
        description="Always predicts 'not readmitted'",
        accuracy=float((pred == y_true).mean()),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_true, score)),
        roc_auc=0.5,
        brier=float(brier_score_loss(y_true, score)),
        notes=(
            "High accuracy, zero recall. It never identifies a single patient "
            "at risk. This is why accuracy is not reported as a headline metric "
            "for this task."
        ),
    )


def prevalence_constant(y_true: np.ndarray) -> BaselineResult:
    """Predict the base rate for everyone — the calibration reference point.

    Perfectly calibrated *on average* and completely without discrimination.
    Any model whose Brier score fails to beat this is adding nothing, however
    good its ROC curve looks.
    """
    prevalence = float(y_true.mean())
    score = np.full(len(y_true), prevalence)

    return BaselineResult(
        name="prevalence_constant",
        description=f"Predicts P(readmit)={prevalence:.4f} for every patient",
        accuracy=float((np.zeros_like(y_true) == y_true).mean()),
        recall=0.0,
        pr_auc=float(average_precision_score(y_true, score)),
        roc_auc=0.5,
        brier=float(brier_score_loss(y_true, score)),
        notes=(
            "The Brier score to beat. A model that cannot improve on this is "
            "not producing usable probabilities, whatever its ranking metrics."
        ),
    )


def single_feature_heuristic(
    df: pd.DataFrame, y_true: np.ndarray, feature: str = "number_inpatient"
) -> BaselineResult:
    """Rank by one clinically obvious feature — prior inpatient visits.

    This is the baseline that actually matters. Prior utilisation is the
    strongest single predictor of readmission and is available at admission
    with no modelling at all. A trained model that cannot beat *this* has not
    earned its deployment, monitoring and retraining infrastructure.
    """
    score = df[feature].to_numpy(dtype=float)
    threshold = float(np.percentile(score, 89))  # flag ~11%, matching prevalence
    pred = (score > threshold).astype(int)

    normalised = (score - score.min()) / (score.max() - score.min() or 1)
    lower, upper = bootstrap_pr_auc_ci(y_true, normalised)

    return BaselineResult(
        name=f"heuristic_{feature}",
        description=f"Ranks patients by {feature} (prior inpatient admissions)",
        accuracy=float((pred == y_true).mean()),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_true, normalised)),
        roc_auc=float(roc_auc_score(y_true, normalised)),
        brier=float(brier_score_loss(y_true, np.clip(normalised, 0, 1))),
        pr_auc_ci=(lower, upper),
        notes=(
            "The bar that matters. Available at admission with no model. Any "
            "trained model must clear this with non-overlapping confidence "
            "intervals to justify the operational cost of running it."
        ),
        extra={"threshold": threshold, "flagged_pct": round(float(pred.mean()) * 100, 2)},
    )


def evaluate_all(df: pd.DataFrame, target_column: str = "target") -> list[BaselineResult]:
    """Run every baseline against a split."""
    y = df[target_column].to_numpy()
    results = [
        majority_class(y),
        prevalence_constant(y),
        single_feature_heuristic(df, y),
    ]
    for r in results:
        log.info(
            "baseline_evaluated",
            baseline=r.name,
            accuracy=round(r.accuracy, 4),
            recall=round(r.recall, 4),
            pr_auc=round(r.pr_auc, 4),
            brier=round(r.brier, 4),
        )
    return results


__all__ = [
    "BaselineResult",
    "bootstrap_pr_auc_ci",
    "evaluate_all",
    "majority_class",
    "prevalence_constant",
    "single_feature_heuristic",
]
