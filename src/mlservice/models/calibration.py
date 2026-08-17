"""Calibration: reliability diagrams, Brier score, expected calibration error.

Most portfolio projects never measure this. For a health-adjacent task it
matters more than the point prediction: a model that ranks patients correctly
but outputs 0.4 where the true rate is 0.1 will misallocate any resource
allocated on the magnitude of its score, while looking excellent on every
ranking metric.

That is why calibration is a **promotion gate** here, not a report — see
docs/DECISIONS/0002-calibration-as-deployment-gate.md. A challenger that
discriminates better but calibrates worse is blocked.

One methodological point that is easy to get wrong: the calibrator is fitted on
**validation**, never on test. Fitting it on the data you then evaluate it with
measures a model's fit to its own evaluation set and reports a number that
cannot be reproduced in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from mlservice.logging_ import get_logger

log = get_logger(__name__)

#: 10 equal-width bins is the conventional operating point for ECE and what the
#: 0.05 gate threshold in configs/thresholds.yaml is calibrated against. Changing
#: it changes the meaning of that number, so it lives here as a named constant.
DEFAULT_BINS = 10


@dataclass
class CalibrationReport:
    brier: float
    ece: float
    mce: float
    bins: int
    bin_edges: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)
    bin_confidence: list[float] = field(default_factory=list)
    bin_accuracy: list[float] = field(default_factory=list)
    method: str = "uncalibrated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "brier": round(self.brier, 6),
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "bins": self.bins,
            "bin_counts": self.bin_counts,
            "bin_confidence": [round(v, 4) for v in self.bin_confidence],
            "bin_accuracy": [round(v, 4) for v in self.bin_accuracy],
        }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = DEFAULT_BINS
) -> CalibrationReport:
    """Compute ECE, MCE and the reliability-diagram data in one pass.

    ECE is the **count-weighted** mean gap between predicted confidence and
    observed frequency. Weighting matters: on an imbalanced task most mass sits
    in the low-probability bins, and an unweighted mean would let a nearly empty
    high-probability bin dominate a number meant to summarise typical behaviour.

    MCE (the worst single bin) is reported alongside because ECE can look
    healthy while one region of the score range is badly wrong — and that region
    is often exactly the high-risk one a decision would act on.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # right-closed so p=1.0 lands in the last bin rather than falling outside
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, bins - 1)

    counts, confidence, accuracy = [], [], []
    weighted_gap, worst_gap = 0.0, 0.0
    n = len(y_true)

    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            confidence.append(0.0)
            accuracy.append(0.0)
            continue

        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        confidence.append(conf)
        accuracy.append(acc)

        gap = abs(conf - acc)
        weighted_gap += (count / n) * gap
        worst_gap = max(worst_gap, gap)

    return CalibrationReport(
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=weighted_gap,
        mce=worst_gap,
        bins=bins,
        bin_edges=[float(e) for e in edges],
        bin_counts=counts,
        bin_confidence=confidence,
        bin_accuracy=accuracy,
    )


def calibrate(
    estimator: Any,
    x_val: Any,
    y_val: Any,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Wrap a fitted estimator in a calibrator fitted on validation data.

    ``FrozenEstimator`` is load-bearing, not decoration: the base model is
    already trained on the training split, and this must fit *only* the
    calibration map on validation. Without freezing, ``CalibratedClassifierCV``
    cross-fits the base estimator on the validation data — which would train on
    a later time period and quietly destroy the chronological separation the
    whole split exists to preserve.

    (This replaces the old ``cv="prefit"`` argument, removed in scikit-learn
    1.9. Same intent, stated more explicitly.)
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    calibrated.fit(x_val, y_val)
    log.info("calibrator_fitted", method=method, n_val=len(y_val))
    return calibrated


def choose_calibration_method(
    estimator: Any,
    x_val: Any,
    y_val: Any,
    bins: int = DEFAULT_BINS,
) -> tuple[str, dict[str, CalibrationReport]]:
    """Pick isotonic or Platt on validation, and return both for the record.

    Selected on validation, never on test. Isotonic is more flexible but can
    overfit on small samples; Platt (sigmoid) is more constrained and often
    better below a few thousand rows. Rather than assume, both are fitted and
    the one with the lower validation ECE wins — with both numbers reported so
    the choice is visible rather than asserted.
    """
    y = np.asarray(y_val)
    reports: dict[str, CalibrationReport] = {}

    uncal = np.asarray(estimator.predict_proba(x_val))[:, 1]
    reports["uncalibrated"] = expected_calibration_error(y, uncal, bins)
    reports["uncalibrated"].method = "uncalibrated"

    for method in ("isotonic", "sigmoid"):
        calibrated = calibrate(estimator, x_val, y_val, method=method)
        prob = np.asarray(calibrated.predict_proba(x_val))[:, 1]
        report = expected_calibration_error(y, prob, bins)
        report.method = method
        reports[method] = report

    best = min(("isotonic", "sigmoid"), key=lambda m: reports[m].ece)

    log.info(
        "calibration_method_selected",
        chosen=best,
        ece_uncalibrated=round(reports["uncalibrated"].ece, 5),
        ece_isotonic=round(reports["isotonic"].ece, 5),
        ece_sigmoid=round(reports["sigmoid"].ece, 5),
    )
    return best, reports


def plot_reliability(
    reports: dict[str, CalibrationReport],
    path: Path,
    title: str = "Reliability diagram",
) -> Path:
    """Render reliability curves. Committed as an image for the model card.

    Bin sample counts are shown underneath because a calibration curve without
    them is misleading: a wild deviation in a bin holding 12 patients reads as a
    serious defect when it is sampling noise.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display in CI or a container
    import matplotlib.pyplot as plt

    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 7), height_ratios=[3, 1], sharex=True)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated", zorder=1)

    for name, report in reports.items():
        mask = [c > 0 for c in report.bin_counts]
        xs = [c for c, keep in zip(report.bin_confidence, mask, strict=True) if keep]
        ys = [a for a, keep in zip(report.bin_accuracy, mask, strict=True) if keep]
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=4,
            lw=1.5,
            label=f"{name} (ECE {report.ece:.4f}, Brier {report.brier:.4f})",
            zorder=2,
        )

    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    first = next(iter(reports.values()))
    centres = [(first.bin_edges[i] + first.bin_edges[i + 1]) / 2 for i in range(first.bins)]
    ax_hist.bar(centres, first.bin_counts, width=1 / first.bins * 0.9, color="0.6")
    ax_hist.set_yscale("log")  # the imbalance spans orders of magnitude
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("n (log)")
    ax_hist.grid(alpha=0.25, lw=0.5, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    log.info("reliability_diagram_written", path=str(path))
    return path


__all__ = [
    "DEFAULT_BINS",
    "CalibrationReport",
    "calibrate",
    "choose_calibration_method",
    "expected_calibration_error",
    "plot_reliability",
]
