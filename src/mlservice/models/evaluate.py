"""Evaluation for imbalanced clinical prediction.

Metric choices are not neutral here, so each is justified:

*   **PR-AUC is the headline, not ROC-AUC.** At a ~7.6% positive rate, ROC-AUC
    is dominated by the large negative class: a model can move ROC-AUC
    substantially while barely changing what happens to the patients who
    actually get readmitted. PR-AUC responds to precision and recall on the
    minority class, which is the question being asked.
*   **Bootstrap confidence intervals on everything.** Without them, "PR-AUC
    improved from 0.086 to 0.094" is uninterpretable — on 13,298 rows with
    ~1,000 positives the sampling variance is not small, and later phases
    compare models on exactly this number.
*   **The operating threshold is chosen for a stated recall target**, on
    validation, never 0.5. A 0.5 threshold on an 11%-prevalence problem flags
    almost nobody, which is why the majority-class baseline gets 0% recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlservice.logging_ import get_logger

log = get_logger(__name__)

BOOTSTRAP_ITERATIONS = 1_000
CI_LEVEL = 0.95

#: Recall target the operating threshold is tuned to hit on validation.
#:
#: Chosen, not derived — and the reasoning is a resourcing argument rather than
#: a statistical one. A readmission-risk score exists to direct a limited
#: intervention (a follow-up call, a pharmacy review). Catching half the
#: readmissions is a meaningful clinical yield, and at this prevalence a higher
#: target flags so much of the population that the list stops being actionable.
#: The precision this costs is reported openly next to it.
TARGET_RECALL = 0.50


@dataclass
class Interval:
    point: float
    lower: float
    upper: float

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals overlap — the test for 'genuinely better'."""
        return not (self.upper < other.lower or other.upper < self.lower)

    def to_dict(self) -> dict[str, float]:
        return {
            "point": round(self.point, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
        }


@dataclass
class EvaluationReport:
    n: int
    n_positive: int
    prevalence: float
    threshold: float

    pr_auc: Interval
    roc_auc: Interval
    brier: Interval

    precision: float
    recall: float
    f1: float
    specificity: float

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    lift_over_prevalence: float
    flagged_rate: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("pr_auc", "roc_auc", "brier"):
            out[key] = getattr(self, key).to_dict()
        return out


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Any,
    iterations: int = BOOTSTRAP_ITERATIONS,
    level: float = CI_LEVEL,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap CI for any score-based metric.

    Resamples that end up single-class are skipped rather than counted: they
    make the metric undefined, and substituting a default would bias the
    interval toward that default.
    """
    point = float(metric(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)

    samples: list[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        samples.append(float(metric(y_true[idx], y_score[idx])))

    if not samples:
        return Interval(point, point, point)

    alpha = (1 - level) / 2
    return Interval(
        point=point,
        lower=float(np.percentile(samples, alpha * 100)),
        upper=float(np.percentile(samples, (1 - alpha) * 100)),
    )


def choose_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float = TARGET_RECALL
) -> tuple[float, dict[str, float]]:
    """Lowest-cost threshold that still reaches ``target_recall``.

    Must be called on **validation**. Choosing an operating point on the test
    set is a subtle form of leakage: the reported precision/recall then reflect
    a threshold fitted to the very data used to report it.

    Among all thresholds meeting the recall target, the one with the highest
    precision is selected — meeting the clinical requirement while flagging as
    few patients as possible.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more precision/recall than thresholds
    precisions, recalls = precisions[:-1], recalls[:-1]

    eligible = recalls >= target_recall
    if not eligible.any():
        # Cannot reach the target at any threshold: fall back to best F1 and say so.
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
        best = int(np.argmax(f1s))
        log.warning(
            "target_recall_unreachable",
            target_recall=target_recall,
            max_recall=float(recalls.max()),
            fallback="best_f1",
        )
        return float(thresholds[best]), {
            "achieved_recall": float(recalls[best]),
            "achieved_precision": float(precisions[best]),
            "target_met": False,
        }

    candidates = np.where(eligible)[0]
    best = int(candidates[np.argmax(precisions[candidates])])

    log.info(
        "threshold_chosen",
        threshold=round(float(thresholds[best]), 6),
        target_recall=target_recall,
        achieved_recall=round(float(recalls[best]), 4),
        achieved_precision=round(float(precisions[best]), 4),
    )
    return float(thresholds[best]), {
        "achieved_recall": float(recalls[best]),
        "achieved_precision": float(precisions[best]),
        "target_met": True,
    }


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    bootstrap: bool = True,
) -> EvaluationReport:
    """Full evaluation at a fixed operating threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    if bootstrap:
        pr_auc = bootstrap_metric(y_true, y_score, average_precision_score)
        roc_auc = bootstrap_metric(y_true, y_score, roc_auc_score)
        brier = bootstrap_metric(y_true, y_score, brier_score_loss)
    else:
        pr_auc = Interval(float(average_precision_score(y_true, y_score)), 0.0, 0.0)
        roc_auc = Interval(float(roc_auc_score(y_true, y_score)), 0.0, 0.0)
        brier = Interval(float(brier_score_loss(y_true, y_score)), 0.0, 0.0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prevalence = float(y_true.mean())
    precision = float(precision_score(y_true, y_pred, zero_division=0))

    return EvaluationReport(
        n=len(y_true),
        n_positive=int(y_true.sum()),
        prevalence=prevalence,
        threshold=float(threshold),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier=brier,
        precision=precision,
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        # How much better than flagging patients at random. The single most
        # honest one-number summary of whether the model is worth running.
        lift_over_prevalence=round(precision / prevalence, 3) if prevalence else 0.0,
        flagged_rate=float(y_pred.mean()),
    )


def beats(challenger: Interval, incumbent: Interval) -> dict[str, Any]:
    """Whether ``challenger`` is *genuinely* better, not just numerically ahead.

    Requires non-overlapping confidence intervals. A point estimate that is
    higher while the intervals overlap is not evidence of improvement, and
    treating it as such is how models get promoted on noise.
    """
    overlapping = challenger.overlaps(incumbent)
    return {
        "challenger": challenger.to_dict(),
        "incumbent": incumbent.to_dict(),
        "point_difference": round(challenger.point - incumbent.point, 6),
        "intervals_overlap": overlapping,
        "genuinely_better": challenger.point > incumbent.point and not overlapping,
    }


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "TARGET_RECALL",
    "EvaluationReport",
    "Interval",
    "beats",
    "bootstrap_metric",
    "choose_threshold",
    "evaluate",
]
