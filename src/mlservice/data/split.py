"""Chronological splitting on a real date.

**This module is a fraction of the size it was.** The medical version of this
project had to *prove* that an ID sequence carried time signal before anything
could split on it — a decile-shift test across eight independent signals,
Spearman correlations, a discontinuity hunt, and an ADR explaining why the
claim was "ordered" rather than "temporal" if the test failed.

Lending Club records ``issue_d``. The split is chronological by construction,
so all of that apparatus is gone. That is the upgrade the domain switch bought,
and it is worth noticing how much complexity a single trustworthy column
removes.

What remains is the part that still matters: splitting at **month boundaries**
rather than row positions, and stating plainly the one guarantee this dataset
cannot give.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mlservice.config import get_settings
from mlservice.data import clean
from mlservice.logging_ import get_logger

log = get_logger(__name__)

#: Borrower identity is unavailable: Lending Club scrubbed ``member_id`` before
#: release, and it is null for every row. The medical version could keep one
#: encounter per patient and assert that nobody straddled the split; here that
#: is simply not possible.
#:
#: A borrower with two loans can therefore appear in both train and test, and
#: the held-out numbers are inflated to whatever extent repeat borrowing
#: occurs. Stated in the model card as a limitation rather than left for a
#: reviewer to infer from a missing check.
BORROWER_IDENTITY_AVAILABLE = False


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict[str, str]
    limitations: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "claim": "chronological split on issue_d, a real date column",
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
            "date_boundaries": self.boundaries,
            "limitations": self.limitations,
        }


def month_boundaries(
    dates: pd.Series, train_fraction: float, val_fraction: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the two cut dates that land closest to the requested fractions.

    Cutting on a **date**, not a row index. Loans issued in the same month
    share an origination cohort — the same credit policy, the same marketing,
    the same macro conditions — so slicing through a month puts near-siblings
    on both sides of the boundary. The leak is small but it is real, and it is
    free to avoid.

    The cost is that the realised fractions are approximate, because months
    have wildly different volumes here: 2007 has hundreds of loans and 2015 has
    tens of thousands. The achieved split is reported rather than assumed.
    """
    counts = dates.dt.to_period("M").value_counts().sort_index()
    cumulative = counts.cumsum() / counts.sum()

    train_cut = cumulative[cumulative >= train_fraction].index[0]
    val_cut = cumulative[cumulative >= train_fraction + val_fraction].index[0]
    return train_cut.to_timestamp(), val_cut.to_timestamp()


def chronological_split(df: pd.DataFrame) -> SplitResult:
    """Train on the past, test on the future. Never at random.

    A random split on time-structured data leaks the future into training and
    inflates every number that follows. Lending Club's book changed enormously
    across this period — volume, grade mix, and the credit policy itself — so a
    random split would let the model see 2015 underwriting while being scored
    on it.
    """
    settings = get_settings().data

    dates = clean.parse_issue_date(df)
    ordered = df.assign(_issued=dates).sort_values("_issued", kind="stable")

    train_cut, val_cut = month_boundaries(
        ordered["_issued"], settings.train_fraction, settings.val_fraction
    )

    train = ordered[ordered["_issued"] < train_cut].drop(columns="_issued").copy()
    val = ordered[(ordered["_issued"] >= train_cut) & (ordered["_issued"] < val_cut)]
    val = val.drop(columns="_issued").copy()
    test = ordered[ordered["_issued"] >= val_cut].drop(columns="_issued").copy()

    # Cheap, absolute guarantees. The failure they protect against is silent:
    # a mis-ordered split still trains, still scores, and simply reports
    # numbers that are too good.
    train_dates = clean.parse_issue_date(train)
    val_dates = clean.parse_issue_date(val)
    test_dates = clean.parse_issue_date(test)
    assert train_dates.max() < val_dates.min(), "train overlaps val in time"
    assert val_dates.max() < test_dates.min(), "val overlaps test in time"
    assert len(train) + len(val) + len(test) == len(df), "rows lost in splitting"

    limitations = []
    if not BORROWER_IDENTITY_AVAILABLE:
        limitations.append(
            "member_id is null for every row, so repeat borrowers cannot be "
            "detected and may appear in both train and test. Held-out metrics "
            "are inflated to whatever extent repeat borrowing occurs."
        )

    boundaries = {
        "train_start": str(train_dates.min().date()),
        "train_end": str(train_dates.max().date()),
        "val_end": str(val_dates.max().date()),
        "test_end": str(test_dates.max().date()),
    }

    result = SplitResult(train, val, test, boundaries, limitations)
    log.info("chronological_split", **result.summary())
    return result


def reference_window(split: SplitResult) -> pd.DataFrame:
    """The frozen window drift is measured against.

    The training split, deliberately: drift means "different from what the
    model learned", so the reference has to be exactly what it learned from.
    Using a later window would measure drift against data the model never saw.
    """
    return split.train.copy()


__all__ = [
    "BORROWER_IDENTITY_AVAILABLE",
    "SplitResult",
    "chronological_split",
    "month_boundaries",
    "reference_window",
]
