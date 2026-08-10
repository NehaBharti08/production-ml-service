"""Chronological split, and the verification that earns the right to call it one.

This dataset has **no timestamp column**. The only time signal is the ordering
of ``encounter_id``. Splitting on that ordering is standard practice, but it is
a *proxy*, and asserting "temporal validation" without evidence would be exactly
the kind of unearned claim this project exists to avoid.

So the proxy is tested before it is trusted. :func:`verify_time_proxy` looks for
**monotonic** shifts in recording and prescribing practice across encounter_id
deciles. The reasoning: clinical practice change is directional — a drug is
adopted, a field starts being captured — whereas a meaningless row ordering
produces non-monotonic noise. Spearman rank correlation against decile index
distinguishes the two.

If verification fails, callers should downgrade the claim to "ordered holdout"
and say so in the README. Both outcomes are publishable; only the untested
assertion is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mlservice.config import get_settings
from mlservice.data import schema
from mlservice.logging_ import get_logger

log = get_logger(__name__)

#: A signal counts as evidence only if it is both strongly monotonic and
#: unlikely by chance. |rho| > 0.8 across 10 deciles is a strong rank trend;
#: p < 0.01 with n=10 is a demanding bar precisely because the sample is small.
MIN_ABS_RHO = 0.8
MAX_P_VALUE = 0.01

#: Verification passes when at least this many independent signals trend. One
#: trending signal could be a single recording change; several, across
#: unrelated columns, is a time axis.
MIN_TRENDING_SIGNALS = 3


@dataclass
class ProxySignal:
    name: str
    column: str
    first_decile: float
    last_decile: float
    spearman_rho: float
    p_value: float
    values: list[float] = field(default_factory=list)

    @property
    def trends(self) -> bool:
        return abs(self.spearman_rho) > MIN_ABS_RHO and self.p_value < MAX_P_VALUE

    @property
    def delta(self) -> float:
        return self.last_decile - self.first_decile


@dataclass
class ProxyVerification:
    passed: bool
    signals: list[ProxySignal]
    n_trending: int
    n_deciles: int

    @property
    def claim(self) -> str:
        """The strongest claim the evidence supports. Used verbatim in docs."""
        return (
            "chronological split (encounter_id proxy, empirically verified)"
            if self.passed
            else "ordered holdout (encounter_id ordering NOT verified as temporal)"
        )


def _rate(fn: Callable[[pd.Series], float]) -> Callable[[pd.Series], float]:
    return fn


#: Signals chosen to be independent of each other and of the target: two
#: data-capture practices, two lab-ordering practices, four prescribing
#: patterns. If they trend together, the common cause is time.
_SIGNALS: tuple[tuple[str, str, Callable[[pd.Series], float]], ...] = (
    ("medical_specialty missing", "medical_specialty", lambda s: float((s == "?").mean())),
    ("payer_code missing", "payer_code", lambda s: float((s == "?").mean())),
    ("A1Cresult measured", "A1Cresult", lambda s: float(s.notna().mean())),
    ("max_glu_serum measured", "max_glu_serum", lambda s: float(s.notna().mean())),
    ("insulin prescribed", "insulin", lambda s: float((s != "No").mean())),
    ("metformin prescribed", "metformin", lambda s: float((s != "No").mean())),
    ("pioglitazone prescribed", "pioglitazone", lambda s: float((s != "No").mean())),
    ("rosiglitazone prescribed", "rosiglitazone", lambda s: float((s != "No").mean())),
)


def verify_time_proxy(df: pd.DataFrame, n_deciles: int = 10) -> ProxyVerification:
    """Test whether ordering by ``encounter_id`` carries real time signal.

    Must be run on the **raw** frame, before cleaning: it inspects the ``"?"``
    sentinels and native NaNs that cleaning deliberately removes.
    """
    ordered = df.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)
    decile = pd.qcut(ordered[schema.ENCOUNTER_ID].rank(method="first"), n_deciles, labels=False)

    signals: list[ProxySignal] = []
    for name, column, fn in _SIGNALS:
        if column not in ordered.columns:
            continue
        per_decile = ordered.groupby(decile)[column].apply(fn)
        rho, p = stats.spearmanr(np.arange(len(per_decile)), per_decile.to_numpy())
        signals.append(
            ProxySignal(
                name=name,
                column=column,
                first_decile=float(per_decile.iloc[0]),
                last_decile=float(per_decile.iloc[-1]),
                spearman_rho=float(rho),
                p_value=float(p),
                values=[float(v) for v in per_decile.to_numpy()],
            )
        )

    n_trending = sum(s.trends for s in signals)
    passed = n_trending >= MIN_TRENDING_SIGNALS

    log.info(
        "time_proxy_verified" if passed else "time_proxy_FAILED",
        n_trending=n_trending,
        n_signals=len(signals),
        threshold=MIN_TRENDING_SIGNALS,
        trending=[s.name for s in signals if s.trends],
    )
    return ProxyVerification(
        passed=passed, signals=signals, n_trending=n_trending, n_deciles=n_deciles
    )


def detect_discontinuities(df: pd.DataFrame, column: str, n_bins: int = 20) -> dict[str, Any]:
    """Find the sharpest bin-to-bin change in a column's prescribing rate.

    Trend tests establish that ordering carries time signal. A *discontinuity*
    can do something stronger: if an abrupt, drug-specific break appears at a
    known point, it dates the ordering against real-world events, which is a
    prediction that could have failed rather than a pattern found after the
    fact.
    """
    ordered = df.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)
    bins = pd.qcut(ordered[schema.ENCOUNTER_ID].rank(method="first"), n_bins, labels=False)
    rate = ordered.groupby(bins)[column].apply(lambda s: float((s != "No").mean()))

    deltas = np.diff(rate.to_numpy())
    idx = int(np.argmin(deltas))
    others = np.delete(deltas, idx)
    typical = float(np.mean(np.abs(others))) or 1e-9

    return {
        "column": column,
        "n_bins": n_bins,
        "rates": [round(float(v), 4) for v in rate.to_numpy()],
        "largest_drop_at_bin": idx,
        "largest_drop_pct_points": round(float(deltas[idx]) * 100, 2),
        "percentile_of_ordering": round((idx + 1) / n_bins * 100, 1),
        "ratio_to_typical_change": round(abs(float(deltas[idx])) / typical, 1),
    }


#: Fraction of the ordering discarded from the end before splitting.
#:
#: Measured, not chosen. A first encounter can only carry a positive label if a
#: *subsequent* encounter exists in the data, so patients admitted near the end
#: of the collection window have their readmissions systematically unobserved —
#: classic right-censoring. The first-encounter positive rate sits at ~9.4%
#: through the first 80% of the ordering and then falls away: -15%, -20%, -27%
#: relative in the next three 5% bins, and -58% (to 3.89%) in the final one.
#:
#: The last bin is where censoring dominates rather than merely contributes, so
#: that is what gets cut. Keeping it would depress every test-set metric for a
#: reason that has nothing to do with the model, and would make Phase 6 drift
#: monitoring chase an artifact of data collection.
CENSORING_BUFFER_FRACTION = 0.05


def censoring_buffer_evidence(df: pd.DataFrame, n_bins: int = 20) -> dict[str, Any]:
    """Quantify label censoring at the end of the observation window."""
    ordered = df.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)
    target = (
        ordered["target"]
        if "target" in ordered.columns
        else (ordered[schema.TARGET] == schema.POSITIVE_LABEL).astype(int)
    )
    bins = pd.qcut(ordered[schema.ENCOUNTER_ID].rank(method="first"), n_bins, labels=False)
    rate = target.groupby(bins).mean()

    n_stable = int(n_bins * (1 - CENSORING_BUFFER_FRACTION * 4))
    stable = float(rate.iloc[:n_stable].mean())

    return {
        "n_bins": n_bins,
        "positive_rate_per_bin": [round(float(v), 4) for v in rate.to_numpy()],
        "stable_rate_first_80pct": round(stable, 4),
        "final_bin_rate": round(float(rate.iloc[-1]), 4),
        "final_bin_relative_drop": round(float(rate.iloc[-1]) / stable - 1, 4),
        "buffer_fraction": CENSORING_BUFFER_FRACTION,
        "interpretation": (
            "A first encounter is labelled positive only if a later encounter "
            "exists in the data. Near the end of collection those later "
            "encounters are unobserved, so the label is missing rather than "
            "negative. The final bin is discarded as a censoring buffer."
        ),
    }


def apply_censoring_buffer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop the final ``CENSORING_BUFFER_FRACTION`` of the ordering."""
    ordered = df.sort_values(schema.ENCOUNTER_ID).reset_index(drop=True)
    keep = int(len(ordered) * (1 - CENSORING_BUFFER_FRACTION))
    kept, dropped = ordered.iloc[:keep].copy(), ordered.iloc[keep:]

    detail = {
        "rows_dropped": len(dropped),
        "pct_dropped": round(len(dropped) / len(ordered) * 100, 2),
        "positive_rate_dropped_region": round(float(dropped["target"].mean()), 4),
        "positive_rate_retained": round(float(kept["target"].mean()), 4),
        "reason": "right-censoring — readmissions after the collection window are unobservable",
    }
    log.info("censoring_buffer_applied", **detail)
    return kept, detail


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    verification: ProxyVerification
    boundaries: dict[str, int]
    censoring: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "claim": self.verification.claim,
            "censoring_buffer": self.censoring,
            "sizes": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
            },
            "positive_rate": {
                "train": round(float(self.train["target"].mean()), 6),
                "val": round(float(self.val["target"].mean()), 6),
                "test": round(float(self.test["target"].mean()), 6),
            },
            "encounter_id_boundaries": self.boundaries,
        }


def chronological_split(df: pd.DataFrame, verification: ProxyVerification) -> SplitResult:
    """Split by position in ``encounter_id`` order — never at random.

    A random split on time-structured data leaks the future into training and
    inflates every number that follows. Because practice genuinely shifts across
    this period, a random split would let the model see post-2007 prescribing
    patterns while being evaluated on them.
    """
    settings = get_settings().data

    # Censoring buffer first: the discarded tail would otherwise land entirely
    # in the test set, depressing every held-out metric for a reason that has
    # nothing to do with the model.
    ordered, censoring = apply_censoring_buffer(df)

    n = len(ordered)
    train_end = int(n * settings.train_fraction)
    val_end = train_end + int(n * settings.val_fraction)

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    # Cheap, absolute guarantee — the failure this protects against is silent.
    assert train[schema.ENCOUNTER_ID].max() < val[schema.ENCOUNTER_ID].min()
    assert val[schema.ENCOUNTER_ID].max() < test[schema.ENCOUNTER_ID].min()
    overlap = set(train[schema.PATIENT_ID]) & set(test[schema.PATIENT_ID])
    assert not overlap, f"{len(overlap)} patients straddle train/test"

    boundaries = {
        "train_min": int(train[schema.ENCOUNTER_ID].min()),
        "train_max": int(train[schema.ENCOUNTER_ID].max()),
        "val_max": int(val[schema.ENCOUNTER_ID].max()),
        "test_max": int(test[schema.ENCOUNTER_ID].max()),
    }

    result = SplitResult(train, val, test, verification, boundaries, censoring)
    log.info("split_complete", **result.summary())
    return result


__all__ = [
    "MIN_TRENDING_SIGNALS",
    "ProxySignal",
    "ProxyVerification",
    "SplitResult",
    "chronological_split",
    "detect_discontinuities",
    "verify_time_proxy",
]
