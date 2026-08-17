"""Subgroup performance, reported openly — including where it is unflattering.

Two rules this module exists to enforce:

*   **Report every subgroup above the size floor, not the flattering ones.**
    Selective reporting is worse than no fairness analysis, because it implies a
    check that was not really performed.
*   **Never report a subgroup too small to say anything.** A recall gap computed
    on 40 patients is sampling noise presented as a finding, and it discredits
    the gaps that are real.

The disparity measure is deliberately *relative to the overall population*
rather than to a "reference group". Picking a reference group encodes a
judgement about whose performance is the norm; comparing everyone to the
population average does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score

from mlservice.logging_ import get_logger
from mlservice.models.calibration import expected_calibration_error

log = get_logger(__name__)

#: Below this, a subgroup metric is too noisy to act on. Matches
#: promotion.subgroup.min_subgroup_n in configs/thresholds.yaml — the gate and
#: the report must agree, or the gate protects a number nobody looked at.
MIN_SUBGROUP_N = 500


@dataclass
class SubgroupMetrics:
    dimension: str
    group: str
    n: int
    n_positive: int
    prevalence: float
    recall: float
    precision: float
    pr_auc: float | None
    brier: float
    ece: float
    flagged_rate: float
    sufficient: bool
    recall_gap_vs_overall: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class SubgroupReport:
    overall_recall: float
    overall_precision: float
    overall_pr_auc: float
    threshold: float
    min_n: int = MIN_SUBGROUP_N
    groups: list[SubgroupMetrics] = field(default_factory=list)

    @property
    def analysable(self) -> list[SubgroupMetrics]:
        return [g for g in self.groups if g.sufficient]

    @property
    def worst_recall_gap(self) -> float:
        """Largest shortfall below overall recall, among analysable groups.

        This is the number the Phase 7 promotion gate watches: a challenger may
        not widen it by more than 20% relative to the incumbent.
        """
        gaps = [g.recall_gap_vs_overall for g in self.analysable]
        return min(gaps) if gaps else 0.0

    @property
    def worst_group(self) -> SubgroupMetrics | None:
        analysable = self.analysable
        if not analysable:
            return None
        return min(analysable, key=lambda g: g.recall_gap_vs_overall)

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst_group
        return {
            "overall": {
                "recall": round(self.overall_recall, 6),
                "precision": round(self.overall_precision, 6),
                "pr_auc": round(self.overall_pr_auc, 6),
                "threshold": round(self.threshold, 6),
            },
            "min_subgroup_n": self.min_n,
            "n_groups": len(self.groups),
            "n_analysable": len(self.analysable),
            "worst_recall_gap": round(self.worst_recall_gap, 6),
            "worst_group": (f"{worst.dimension}={worst.group}" if worst else None),
            "groups": [g.to_dict() for g in self.groups],
        }


def evaluate_subgroups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    dimensions: tuple[str, ...],
) -> SubgroupReport:
    """Per-subgroup metrics at the production operating threshold.

    The **same** threshold is applied to every group. Per-group thresholds would
    improve the numbers and would be indefensible: it means treating patients
    differently based on demographics, which is the thing fairness analysis
    exists to detect rather than to implement.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    overall_recall = float(recall_score(y_true, y_pred, zero_division=0))
    from sklearn.metrics import precision_score

    report = SubgroupReport(
        overall_recall=overall_recall,
        overall_precision=float(precision_score(y_true, y_pred, zero_division=0)),
        overall_pr_auc=float(average_precision_score(y_true, y_score)),
        threshold=float(threshold),
    )

    for dim in dimensions:
        if dim not in df.columns:
            continue
        for group in sorted(df[dim].astype(str).unique()):
            mask = (df[dim].astype(str) == group).to_numpy()
            n = int(mask.sum())
            n_pos = int(y_true[mask].sum())
            sufficient = n >= MIN_SUBGROUP_N and n_pos > 0

            note = ""
            if n < MIN_SUBGROUP_N:
                note = f"n below {MIN_SUBGROUP_N} — reported for transparency, not analysed"
            elif n_pos == 0:
                note = "no positive cases — recall undefined"

            group_recall = (
                float(recall_score(y_true[mask], y_pred[mask], zero_division=0)) if n_pos else 0.0
            )
            # PR-AUC is undefined for a single-class subgroup.
            pr_auc = (
                float(average_precision_score(y_true[mask], y_score[mask]))
                if len(np.unique(y_true[mask])) > 1
                else None
            )
            ece = (
                expected_calibration_error(y_true[mask], y_score[mask]).ece
                if n >= 50
                else float("nan")
            )

            report.groups.append(
                SubgroupMetrics(
                    dimension=dim,
                    group=group,
                    n=n,
                    n_positive=n_pos,
                    prevalence=float(y_true[mask].mean()) if n else 0.0,
                    recall=group_recall,
                    precision=float(precision_score(y_true[mask], y_pred[mask], zero_division=0)),
                    pr_auc=pr_auc,
                    brier=float(brier_score_loss(y_true[mask], y_score[mask])) if n else 0.0,
                    ece=ece,
                    flagged_rate=float(y_pred[mask].mean()) if n else 0.0,
                    sufficient=sufficient,
                    recall_gap_vs_overall=group_recall - overall_recall if n_pos else 0.0,
                    note=note,
                )
            )

    worst = report.worst_group
    log.info(
        "subgroups_evaluated",
        n_groups=len(report.groups),
        n_analysable=len(report.analysable),
        worst_gap=round(report.worst_recall_gap, 4),
        worst_group=f"{worst.dimension}={worst.group}" if worst else None,
    )
    return report


__all__ = [
    "MIN_SUBGROUP_N",
    "SubgroupMetrics",
    "SubgroupReport",
    "evaluate_subgroups",
]
