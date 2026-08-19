"""Replay harness: demonstrate drift detection, honestly labelled.

This module can produce two very different things, and conflating them would be
the most dishonest thing in the repository:

**REAL drift** — replaying the held-out test split in chronological order. The
1999–2008 period contains genuine practice change: `medical_specialty` recording
rates shift, prescribing patterns move, and the 2007 rosiglitazone withdrawal
appears as a sharp discontinuity at the 80th percentile of the ordering (Phase 1
verified this against the dated event). Any drift detected here is real drift in
real data.

**INDUCED drift** — deliberately resampling a window to shift a distribution,
so the detector has something unambiguous to catch. This is a demonstration, not
a finding.

Every report this module writes carries ``drift_origin: "real"`` or
``"induced"``, every induced run records exactly what was manipulated, and the
documentation says which is which. A monitoring demo that shows a detector
firing without saying the drift was manufactured is claiming something it has
not earned.

The clock is compressed for the demo: label maturation is 30 days by definition,
and waiting 30 real days to show a delayed-label join is not a demo. The
compression is cosmetic — ordering and the *structure* of the delay are
preserved, which is what the monitoring logic actually depends on.
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
from mlservice.data import schema
from mlservice.logging_ import get_logger

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
    """Split a frame into consecutive windows in ``encounter_id`` order.

    No manipulation. Any drift detected across these is **real** — it is the
    dataset's own change over 1999–2008.
    """
    config = get_thresholds().model_dump()["drift"]["alert"]["data_drift"]
    size = window_rows or config["window_size_rows"]

    ordered = frame.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)
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


def induce_age_shift(
    window: pd.DataFrame,
    target_bands: tuple[str, ...] = ("[70-80)", "[80-90)"),
    weight: float = 4.0,
) -> tuple[pd.DataFrame, Manipulation]:
    """Over-sample older patients — a plausible demographic shift.

    Chosen because age is a real risk factor here (Phase 2 measured recall from
    0.233 at [40-50) to 0.692 at [80-90)), so shifting it moves both the input
    distribution *and* the score distribution. That exercises data drift and
    prediction drift together, which a synthetic column could not.
    """
    before = window["age"].value_counts(normalize=True).to_dict()

    weights = np.where(window["age"].isin(target_bands), weight, 1.0)
    weights = weights / weights.sum()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(window), size=len(window), replace=True, p=weights)
    shifted = window.iloc[idx].reset_index(drop=True)

    after = shifted["age"].value_counts(normalize=True).to_dict()
    return shifted, Manipulation(
        feature="age",
        kind="resample",
        detail=(
            f"over-sampled {list(target_bands)} with weight {weight} — a plausible "
            "demographic shift toward an older population"
        ),
        before={k: round(float(v), 4) for k, v in sorted(before.items())},
        after={k: round(float(v), 4) for k, v in sorted(after.items())},
    )


def induce_utilisation_shift(
    window: pd.DataFrame, feature: str = "number_inpatient", shift: int = 2
) -> tuple[pd.DataFrame, Manipulation]:
    """Add prior admissions — simulates a sicker referred population.

    ``number_inpatient`` is the strongest single predictor, so this produces a
    large, obvious prediction shift. Deliberately blunt: the point of an induced
    demo is that the detector's response is unambiguous.
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
        detail=f"added {shift} prior admissions to every record — a sicker referred population",
        before=before,
        after=after,
    )


def induce_specialty_collapse(
    window: pd.DataFrame, keep: str = "InternalMedicine"
) -> tuple[pd.DataFrame, Manipulation]:
    """Collapse medical_specialty to one value — an upstream recording change.

    Models a very common real failure: a source system stops populating a field
    properly. Included because it is the kind of drift that is *not* a
    population change at all, and the response differs — you fix the pipeline,
    you do not retrain.
    """
    before = window["medical_specialty"].value_counts(normalize=True).head(5).to_dict()
    shifted = window.copy()
    shifted["medical_specialty"] = keep
    return shifted, Manipulation(
        feature="medical_specialty",
        kind="collapse",
        detail=(
            f"forced every record to '{keep}' — models an upstream system that "
            "stopped populating the field. Requires a pipeline fix, NOT a retrain."
        ),
        before={k: round(float(v), 4) for k, v in before.items()},
        after={keep: 1.0},
    )


INDUCERS: dict[str, Callable[..., tuple[pd.DataFrame, Manipulation]]] = {
    "age": induce_age_shift,
    "utilisation": induce_utilisation_shift,
    "specialty": induce_specialty_collapse,
}


def induced_windows(
    frame: pd.DataFrame,
    inducer: str = "age",
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
    ordered = frame.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)

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

    Real maturation is 30 days by definition, so at any moment only the oldest
    part of a window has a label. Taking the **earliest** rows rather than a
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
                "the 1999-2008 dataset."
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
    "induce_age_shift",
    "induce_specialty_collapse",
    "induce_utilisation_shift",
    "induced_windows",
    "save_result",
    "simulate_maturation",
]
