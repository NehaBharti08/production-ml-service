"""Replay harness: demonstrate drift detection, honestly labelled.

This module can produce two very different things, and conflating them would be
the most dishonest thing in the repository:

**REAL drift** — replaying the held-out test split in chronological order. The
2007–2015 period contains genuine policy change: Lending Club repriced (mean
`int_rate` moves 11.97% to 12.85% across one window boundary in late 2011),
introduced whole-loan listing to institutional buyers (`initial_list_status`
goes 0.7% to 22.2% in late 2012), and tightened income verification (`Not
Verified` falls 58.4% to 32.8% in late 2010). Any drift detected here is real
drift in real data.

**INDUCED drift** — deliberately resampling a window to shift a distribution,
so the detector has something unambiguous to catch. This is a demonstration, not
a finding.

Every report this module writes carries ``drift_origin: "real"`` or
``"induced"``, every induced run records exactly what was manipulated, and the
documentation says which is which. A monitoring demo that shows a detector
firing without saying the drift was manufactured is claiming something it has
not earned.

The clock is compressed for the demo. In credit a label does not mature in
days: a loan's outcome is not final until its term ends, which is 1,096 days
for the 92.4% of loans on a 36-month term and 1,826 for the rest. Waiting three
years to show a delayed-label join is not a demo. The compression is cosmetic —
ordering and the *structure* of the delay are preserved, which is what the
monitoring logic actually depends on.

That maturation lag is not merely inconvenient, and the honest consequence is
recorded in configs/thresholds.yaml: a label-pipeline watchdog keyed to it
cannot distinguish "the join broke" from "no loan has matured yet".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from mlservice.config import get_settings, get_thresholds
from mlservice.data import clean
from mlservice.logging_ import get_logger
from mlservice.monitoring.null_calibration import population_stability_index

log = get_logger(__name__)

DriftOrigin = Literal["real", "induced"]


@dataclass
class Manipulation:
    """A single deliberate change, recorded so it cannot be forgotten."""

    feature: str
    kind: str
    detail: str
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)


@dataclass
class ReplayWindow:
    index: int
    rows: int
    drift_origin: DriftOrigin
    manipulations: list[Manipulation] = field(default_factory=list)
    frame: pd.DataFrame | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "rows": self.rows,
            "drift_origin": self.drift_origin,
            "manipulations": [
                {
                    "feature": m.feature,
                    "kind": m.kind,
                    "detail": m.detail,
                    "before": m.before,
                    "after": m.after,
                }
                for m in self.manipulations
            ],
        }


def chronological_windows(
    frame: pd.DataFrame, window_rows: int | None = None
) -> list[ReplayWindow]:
    """Split a frame into consecutive windows in ``issue_d`` order.

    No manipulation. Any drift detected across these is **real** — it is the
    dataset's own change over 2007–2015, across which Lending Club's book grew
    by three orders of magnitude and its credit policy changed repeatedly.
    """
    config = get_thresholds().model_dump()["drift"]["alert"]["data_drift"]
    size = window_rows or config["window_size_rows"]

    # Chronological, via the parsed date. Sorting the raw string column ordered
    # the replay alphabetically by month name, so "real drift over 2007-2015"
    # was drift between Apr and Aug of unrelated years.
    ordered = clean.order_by_time(frame).reset_index(drop=True)
    windows: list[ReplayWindow] = []
    for i in range(0, len(ordered) - size + 1, size):
        chunk = ordered.iloc[i : i + size].copy()
        windows.append(
            ReplayWindow(index=len(windows), rows=len(chunk), drift_origin="real", frame=chunk)
        )

    log.info(
        "chronological_windows_built",
        n_windows=len(windows),
        window_rows=size,
        drift_origin="real",
        note="no manipulation applied; any drift detected is real",
    )
    return windows


def induce_grade_shift(
    window: pd.DataFrame,
    target_grades: tuple[str, ...] = ("D", "E", "F", "G"),
    weight: float = 4.0,
) -> tuple[pd.DataFrame, Manipulation]:
    """Over-sample low grades — a lender loosening its credit policy.

    The most realistic drift in this domain, and it genuinely happened: Lending
    Club's grade mix moved repeatedly across 2007-2015 as they chased volume.

    Chosen because `grade` carries the largest coefficient in the champion
    (grade_A at -0.78), so shifting it moves the input distribution **and** the
    score distribution together. That exercises data drift and prediction
    drift in one manipulation, which a synthetic column could not.

    **The target set is D-G, not E-G.** At E-G this inducer tripled grade E and
    moved PSI to 0.0388 — under the 0.10 threshold, so the deliberately drifted
    windows came back *quiet* and the demonstration demonstrated nothing. E-G
    is only 3.4% of the book, and no reweighting of a 3% tail moves the
    population. D-G is the sub-prime shoulder and reaches PSI 0.2746 at the
    same weight of 4.0 — the correction is to which grades count as
    below-prime, not to the weight, which would be tuning until it passed.
    """
    before = window["grade"].value_counts(normalize=True).to_dict()

    weights = np.where(window["grade"].isin(target_grades), weight, 1.0)
    weights = weights / weights.sum()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(window), size=len(window), replace=True, p=weights)
    shifted = window.iloc[idx].reset_index(drop=True)

    after = shifted["grade"].value_counts(normalize=True).to_dict()
    return shifted, Manipulation(
        feature="grade",
        kind="resample",
        detail=(
            f"over-sampled grades {list(target_grades)} with weight {weight} — a "
            "lender loosening credit policy to chase volume"
        ),
        before={k: round(float(v), 4) for k, v in sorted(before.items())},
        after={k: round(float(v), 4) for k, v in sorted(after.items())},
    )


def induce_leverage_shift(
    window: pd.DataFrame, feature: str = "dti", shift: float = 8.0
) -> tuple[pd.DataFrame, Manipulation]:
    """Raise debt-to-income across the board — a macro deterioration.

    Models borrowers arriving more leveraged than the training population, the
    way they would in a downturn. Deliberately blunt: the point of an induced
    demo is that the detector's response is unambiguous, not subtle.
    """
    before = {
        "mean": round(float(window[feature].mean()), 4),
        "p90": round(float(window[feature].quantile(0.9)), 4),
    }
    shifted = window.copy()
    shifted[feature] = (shifted[feature] + shift).clip(upper=100)
    after = {
        "mean": round(float(shifted[feature].mean()), 4),
        "p90": round(float(shifted[feature].quantile(0.9)), 4),
    }
    return shifted, Manipulation(
        feature=feature,
        kind="additive_shift",
        detail=(
            f"added {shift} points of debt-to-income to every borrower — a "
            "macro deterioration in household leverage"
        ),
        before=before,
        after=after,
    )


def induce_purpose_collapse(
    window: pd.DataFrame, keep: str = "debt_consolidation"
) -> tuple[pd.DataFrame, Manipulation]:
    """Collapse ``purpose`` to one value — an upstream recording change.

    Models a very common real failure: a source system stops populating a
    field properly, or a new application form defaults it. Included because it
    is the kind of drift that is **not a population change at all**, and the
    correct response differs completely — you fix the pipeline, you do not
    retrain. Retraining on it would bake the break into the model.
    """
    before = window["purpose"].value_counts(normalize=True).head(5).to_dict()
    shifted = window.copy()
    shifted["purpose"] = keep
    return shifted, Manipulation(
        feature="purpose",
        kind="collapse",
        detail=(
            f"forced every record to '{keep}' — models an upstream system that "
            "stopped populating the field. Requires a pipeline fix, NOT a retrain."
        ),
        before={k: round(float(v), 4) for k, v in before.items()},
        after={keep: 1.0},
    )


INDUCERS: dict[str, Callable[..., tuple[pd.DataFrame, Manipulation]]] = {
    "grade": induce_grade_shift,
    "leverage": induce_leverage_shift,
    "purpose": induce_purpose_collapse,
}


def _assert_actually_induced(
    before: pd.DataFrame, after: pd.DataFrame, manipulation: Manipulation
) -> None:
    """An inducer that does not induce must fail, not pass quietly.

    The grade inducer spent a domain change targeting a 3.4% tail: it ran, it
    recorded a manipulation, the report said ``drift_origin: induced`` — and
    the detector stayed under threshold, so the demo showed clean windows and
    "drifted" windows behaving identically. Nothing in the pipeline objected,
    because nothing checked that the manipulation had an effect.

    Measured against the feature's own calibrated threshold, so this asks the
    only question that matters: would the monitor we actually ship see it?
    """
    from mlservice.monitoring import drift as drift_mod

    feature = manipulation.feature
    threshold = drift_mod._feature_thresholds().get(feature)
    if threshold is None:
        return

    psi = population_stability_index(before[feature], after[feature])
    if psi <= threshold:
        raise ValueError(
            f"inducer '{manipulation.kind}' on {feature!r} moved PSI to only "
            f"{psi:.4f}, at or under its {threshold:.4f} threshold — the "
            "window would be reported as induced drift that the detector "
            "cannot see. Strengthen the manipulation rather than shipping a "
            "demonstration that demonstrates nothing."
        )


def induced_windows(
    frame: pd.DataFrame,
    inducer: str = "grade",
    window_rows: int | None = None,
    clean_windows: int = 2,
    drifted_windows: int = 3,
) -> list[ReplayWindow]:
    """Clean windows, then deliberately drifted ones.

    The clean prefix matters: it establishes that the detector is *quiet* on
    undisturbed data before it fires. A demo that only shows the alarm going off
    has not shown the alarm works — only that it is loud.
    """
    if inducer not in INDUCERS:
        raise ValueError(f"unknown inducer {inducer!r}; choose from {sorted(INDUCERS)}")

    config = get_thresholds().model_dump()["drift"]["alert"]["data_drift"]
    size = window_rows or config["window_size_rows"]
    ordered = clean.order_by_time(frame).reset_index(drop=True)

    windows: list[ReplayWindow] = []
    total = clean_windows + drifted_windows

    for i in range(total):
        start = i * size
        chunk = ordered.iloc[start : start + size].copy()
        if len(chunk) < size // 2:
            break

        if i < clean_windows:
            windows.append(ReplayWindow(index=i, rows=len(chunk), drift_origin="real", frame=chunk))
        else:
            shifted, manipulation = INDUCERS[inducer](chunk)
            _assert_actually_induced(chunk, shifted, manipulation)
            windows.append(
                ReplayWindow(
                    index=i,
                    rows=len(shifted),
                    drift_origin="induced",
                    manipulations=[manipulation],
                    frame=shifted,
                )
            )

    log.info(
        "induced_windows_built",
        n_windows=len(windows),
        clean=clean_windows,
        drifted=len(windows) - clean_windows,
        inducer=inducer,
        note="DRIFT IS ARTIFICIAL from the clean prefix onward",
    )
    return windows


def simulate_maturation(
    frame: pd.DataFrame, scores: np.ndarray, fraction: float = 0.6
) -> pd.DataFrame:
    """Return the subset whose labels would have matured.

    Real maturation is the loan's term, so at any moment only the oldest part of
    a window has a label. Taking the **earliest** rows rather than a
    random sample preserves that structure — a random sample would quietly
    pretend labels arrive uniformly, which is the one property delayed-label
    monitoring exists to handle.
    """
    n = int(len(frame) * fraction)
    matured = frame.iloc[:n].copy()
    matured["predicted_proba"] = scores[:n]
    matured["outcome_label"] = matured["target"].to_numpy()
    return matured


@dataclass
class ReplayResult:
    drift_origin: DriftOrigin
    inducer: str | None
    windows: list[dict[str, Any]] = field(default_factory=list)
    alert: dict[str, Any] = field(default_factory=dict)
    first_detection_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_utc": datetime.now(UTC).isoformat(),
            # First key in the file, deliberately: nobody should be able to read
            # this report without seeing whether the drift was manufactured.
            "drift_origin": self.drift_origin,
            "honesty_note": (
                "Drift in these windows is ARTIFICIAL — deliberately introduced to "
                "demonstrate detection. It is not a finding about the data."
                if self.drift_origin == "induced"
                else "No manipulation applied. Any drift detected is real change in "
                "the 2007-2015 dataset."
            ),
            "inducer": self.inducer,
            "first_detection_window": self.first_detection_window,
            "alert": self.alert,
            "windows": self.windows,
        }


def save_result(result: ReplayResult, path: Path | None = None) -> Path:
    settings = get_settings()
    name = f"replay_{result.drift_origin}"
    if result.inducer:
        name += f"_{result.inducer}"
    target = path or (settings.paths.reports / f"{name}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log.info("replay_result_written", path=str(target), drift_origin=result.drift_origin)
    return target


__all__ = [
    "INDUCERS",
    "Manipulation",
    "ReplayResult",
    "ReplayWindow",
    "chronological_windows",
    "induce_grade_shift",
    "induce_leverage_shift",
    "induce_purpose_collapse",
    "induced_windows",
    "save_result",
    "simulate_maturation",
]
