"""Per-feature drift thresholds, calibrated against an empirical null.

This module is the answer to "why that number?" — the question that separates
monitoring somebody reasoned about from monitoring somebody copied.

**The problem with the conventional approach.** The credit-risk convention says
PSI < 0.1 is stable, 0.1–0.2 moderate, > 0.2 significant. Those numbers are
defensible and widely used, but they are *generic*: they say nothing about how
much a particular feature in a particular dataset naturally churns. Applied
uniformly, a feature that is genuinely volatile alarms constantly while a frozen
one could shift materially and never breach.

**The problem with p-values.** A Kolmogorov–Smirnov test on a 5,000-row window
returns p < 0.05 for shifts far too small to act on. Statistical significance
scales with n; practical significance does not. Alerting on p-values at
monitoring window sizes guarantees a permanently red dashboard, and a
permanently red dashboard is one nobody reads.

**What this does instead.** Measure the null. Split the training period — which
we already accepted as stable, by training on it — into consecutive windows,
compute PSI between adjacent pairs, and let that distribution define what normal
churn looks like *for each feature individually*:

    threshold[feature] = clamp(percentile_99(null_psi[feature]), floor, ceiling)

Every threshold then answers the question with: *"because this feature moved
that much between stable training windows only 1% of the time."*

The clamp is not arbitrary either. Its floor and ceiling are the credit-risk
convention, and they stop the calibration producing something absurd: a
pathologically noisy feature cannot set a permissive bar that hides real drift,
and a frozen feature cannot set a hair-trigger that fires on rounding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlservice.config import PROJECT_ROOT, get_settings, get_thresholds
from mlservice.data import schema
from mlservice.logging_ import get_logger

log = get_logger(__name__)

#: Added to every bin proportion before the log. Without it a category present
#: in one window and absent in the other makes PSI infinite, which would turn a
#: single rare value into a permanent alarm.
EPSILON = 1e-6


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    bin_edges: np.ndarray | None = None,
) -> float:
    """PSI between two samples of one feature.

    ``sum((cur% - ref%) * ln(cur% / ref%))`` over bins.

    Numeric features use **quantile** edges derived from the reference, not
    equal-width ones. Equal-width bins on a skewed feature — and most features
    here are skewed, e.g. ``number_inpatient`` is zero for most patients — put
    nearly all mass in one bin, so the statistic loses the resolution to detect
    anything.

    Categorical features compare category proportions directly, with the union
    of both category sets so a *new* category registers as drift rather than
    being silently dropped.
    """
    if pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current):
        if bin_edges is None:
            bin_edges = _quantile_edges(reference, bins)
        ref_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
        cur_counts, _ = np.histogram(current.dropna(), bins=bin_edges)
    else:
        categories = sorted(set(reference.astype(str).unique()) | set(current.astype(str).unique()))
        ref_counts = np.array([(reference.astype(str) == c).sum() for c in categories], dtype=float)
        cur_counts = np.array([(current.astype(str) == c).sum() for c in categories], dtype=float)

    ref_total = max(ref_counts.sum(), 1)
    cur_total = max(cur_counts.sum(), 1)
    ref_pct = np.clip(ref_counts / ref_total, EPSILON, None)
    cur_pct = np.clip(cur_counts / cur_total, EPSILON, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _quantile_edges(series: pd.Series, bins: int) -> np.ndarray:
    """Quantile bin edges, deduplicated.

    A feature where one value dominates (``number_emergency`` is 0 for most
    patients) produces duplicate quantiles. Deduplicating collapses the bin
    count rather than creating zero-width bins, which would divide by zero.
    """
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(series.dropna().to_numpy(dtype=float), quantiles))
    # Widen the outer edges so values beyond the reference range still land in a
    # bin instead of being dropped — an out-of-range value is drift, not noise.
    if len(edges) < 2:
        return np.array([series.min() - 1, series.max() + 1], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


@dataclass
class FeatureNull:
    """The observed null distribution of PSI for one feature."""

    feature: str
    n_comparisons: int
    psi_values: list[float]
    median: float
    p90: float
    p99: float
    maximum: float
    threshold: float
    clamped: str | None = None  # "floor" | "ceiling" | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["psi_values"] = [round(v, 6) for v in self.psi_values]
        for k in ("median", "p90", "p99", "maximum", "threshold"):
            d[k] = round(d[k], 6)
        return d


@dataclass
class CalibrationResult:
    n_windows: int
    window_rows: int
    percentile: int
    floor: float
    ceiling: float
    features: dict[str, FeatureNull] = field(default_factory=dict)

    def thresholds(self) -> dict[str, float]:
        return {name: round(f.threshold, 6) for name, f in self.features.items()}

    def summary(self) -> dict[str, Any]:
        clamped_floor = [n for n, f in self.features.items() if f.clamped == "floor"]
        clamped_ceiling = [n for n, f in self.features.items() if f.clamped == "ceiling"]
        return {
            "n_windows": self.n_windows,
            "n_comparisons_per_feature": self.n_windows - 1,
            "window_rows": self.window_rows,
            "percentile": self.percentile,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "n_features": len(self.features),
            "clamped_to_floor": sorted(clamped_floor),
            "clamped_to_ceiling": sorted(clamped_ceiling),
            "n_unclamped": len(self.features) - len(clamped_floor) - len(clamped_ceiling),
        }


def calibrate(
    reference: pd.DataFrame,
    n_windows: int | None = None,
    percentile: int | None = None,
    floor: float | None = None,
    ceiling: float | None = None,
) -> CalibrationResult:
    """Derive a per-feature PSI threshold from the training period's own churn.

    ``reference`` must be the **training** split, in chronological order. Using
    anything later would calibrate the null against data the model has not been
    accepted on, which defeats the purpose: the null is supposed to describe
    variation we have already decided is tolerable.
    """
    config = get_thresholds().model_dump()["drift"]["per_feature"]["calibration"]
    n_windows = n_windows or config["n_null_windows"]
    percentile = percentile or config["percentile"]
    floor = floor if floor is not None else config["floor"]
    ceiling = ceiling if ceiling is not None else config["ceiling"]

    ordered = reference.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)
    window_rows = len(ordered) // n_windows
    if window_rows < 100:
        raise ValueError(
            f"{n_windows} windows over {len(ordered)} rows gives {window_rows} rows each — "
            "too few for PSI to be stable. Use fewer windows or more data."
        )

    windows = [ordered.iloc[i * window_rows : (i + 1) * window_rows] for i in range(n_windows)]

    features = [
        c for c in (*schema.NUMERIC_FEATURES, *schema.CATEGORICAL_FEATURES) if c in ordered.columns
    ]

    result = CalibrationResult(
        n_windows=n_windows,
        window_rows=window_rows,
        percentile=percentile,
        floor=floor,
        ceiling=ceiling,
    )

    for feature in features:
        # Adjacent pairs only. Comparing window 1 against window 20 would fold
        # genuine long-run drift into the "null", inflating the threshold and
        # making the detector blind to exactly what it is meant to catch.
        psi_values = [
            population_stability_index(windows[i][feature], windows[i + 1][feature])
            for i in range(n_windows - 1)
        ]

        p99 = float(np.percentile(psi_values, percentile))
        threshold = float(np.clip(p99, floor, ceiling))
        clamped = "floor" if p99 < floor else "ceiling" if p99 > ceiling else None

        result.features[feature] = FeatureNull(
            feature=feature,
            n_comparisons=len(psi_values),
            psi_values=psi_values,
            median=float(np.median(psi_values)),
            p90=float(np.percentile(psi_values, 90)),
            p99=p99,
            maximum=float(np.max(psi_values)),
            threshold=threshold,
            clamped=clamped,
        )

    log.info("null_calibration_complete", **result.summary())
    return result


def write_thresholds(result: CalibrationResult, path: Path | None = None) -> Path:
    """Write the calibrated thresholds back into configs/thresholds.yaml.

    Edited surgically rather than rewritten, so the provenance comments and
    every unrelated block survive. A regenerated file would lose the reasoning
    that makes the rest of this project's thresholds defensible.
    """
    target = path or (PROJECT_ROOT / "configs" / "thresholds.yaml")
    text = target.read_text(encoding="utf-8")

    block_start = text.index("  per_feature:")
    block_end = text.index("  # ---", block_start)

    summary = result.summary()
    lines = [
        "  per_feature:",
        "    provenance: MEASURED",
        f'    source: "empirical null over {result.n_windows} training windows; '
        f'see docs/MONITORING.md"',
        "    measured_in_phase: 6",
        '    generated_by: "mlservice.monitoring.null_calibration"',
        "    calibration:",
        f"      n_null_windows: {result.n_windows}",
        f"      window_rows: {result.window_rows}",
        f"      percentile: {result.percentile}",
        f"      floor: {result.floor}",
        f"      ceiling: {result.ceiling}",
        f"      n_clamped_to_floor: {len(summary['clamped_to_floor'])}",
        f"      n_clamped_to_ceiling: {len(summary['clamped_to_ceiling'])}",
        f"      n_unclamped: {summary['n_unclamped']}",
        "",
        "    # Each value is that feature's OWN 99th-percentile PSI between",
        "    # adjacent stable training windows, clamped to the credit-risk",
        "    # convention. Regenerate with: uv run mlservice monitor calibrate",
        "    thresholds:",
    ]
    for name, threshold in sorted(result.thresholds().items()):
        lines.append(f"      {name}: {threshold}")
    lines.append("")

    target.write_text(
        text[:block_start] + "\n".join(lines) + "\n" + text[block_end:], encoding="utf-8"
    )
    log.info("thresholds_written", path=str(target), n_features=len(result.features))
    return target


def save_report(result: CalibrationResult, path: Path | None = None) -> Path:
    """Persist the full null distribution, not just the chosen thresholds.

    The distribution is the evidence. Keeping only the threshold would leave the
    number unfalsifiable — nobody could later check whether it was derived or
    invented.
    """
    settings = get_settings()
    target = path or (settings.paths.reports / "null_calibration.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": result.summary(),
        "features": {name: f.to_dict() for name, f in result.features.items()},
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("null_calibration_report_written", path=str(target))
    return target


__all__ = [
    "EPSILON",
    "CalibrationResult",
    "FeatureNull",
    "calibrate",
    "population_stability_index",
    "save_report",
    "write_thresholds",
]
