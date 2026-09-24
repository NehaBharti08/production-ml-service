"""The feature pipeline: one fitted object that travels with the model.

Two properties of credit-bureau data force choices the medical version never
had to make.

**Missingness is informative.** ``mths_since_recent_inq`` is null for 17.4% of
loans, and it is null precisely when there has been no recent credit inquiry —
which is a *good* sign, not an absence of information. Median-imputing it
silently converts "never" into "typical". So every numeric column is imputed
**and** accompanied by a missingness indicator, letting the model learn from
the fact of absence rather than having it papered over.

**The tails are extreme.** ``tot_coll_amt`` has a skew of 747; ``annual_inc``
of 44. Under ``StandardScaler`` a handful of millionaires and one enormous
collection balance would dominate the L2 penalty, and the fitted coefficients
would describe those outliers rather than the population. A quantile transform
maps each feature onto a normal distribution by rank, which fixes skew and
outliers in one step — and because it is **monotone**, it preserves the
directional relationships the behaviour tests assert.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy import special
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.preprocessing._data import BOUNDS_THRESHOLD

from mlservice.data import schema

#: Enough quantiles to describe the distribution without memorising it. With
#: 389k training rows, 1000 is a fine grid. The grid size was never the cost —
#: see FastNormalQuantileTransformer for what was.
N_QUANTILES = 1000


class FastNormalQuantileTransformer(QuantileTransformer):  # type: ignore[misc]  # sklearn is untyped
    """QuantileTransformer with a normal output, minus scipy's per-call overhead.

    **Bit-identical output, roughly a fifth of the latency.** Profiling one
    prediction on the credit champion:

        numeric preprocessing   109 ms
          QuantileTransformer   104 ms
        categorical encoding      8 ms
        the model itself          0.31 ms

    99.7% of inference was preprocessing, and nearly all of that was this one
    step. The interpolation is cheap. The expense is that for
    ``output_distribution="normal"`` sklearn calls ``scipy.stats.norm.ppf``
    three times *per column* — the mapping itself plus two clip bounds — and
    each call goes through scipy's generic distribution machinery, argument
    checking included. At 54 numeric columns that is 162 trips through it for
    a single row, which is also why a 200-row batch cost the same as one row.

    ``scipy.special.ndtri`` is the same function as ``norm.ppf``, evaluated
    directly in C. Swapping it in changes no score by any amount: the
    equivalence test compares this class against its parent on real training
    columns and requires exact equality, which is also what protects against
    this mirror of a private sklearn method drifting from a future version.

    Only the forward, normal-output path is replaced. Everything else,
    including the inverse transform, is the parent's code.
    """

    def _transform_col(self, X_col, quantiles, inverse):  # type: ignore[no-untyped-def]  # noqa: N803
        if inverse or self.output_distribution != "normal":
            return super()._transform_col(X_col, quantiles, inverse)

        lower_bound_x = quantiles[0]
        upper_bound_x = quantiles[-1]

        with np.errstate(invalid="ignore"):
            lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
            upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        finite = X_col[isfinite_mask]
        X_col[isfinite_mask] = 0.5 * (
            np.interp(finite, quantiles, self.references_)
            - np.interp(-finite, -quantiles[::-1], -self.references_[::-1])
        )
        X_col[upper_bounds_idx] = 1
        X_col[lower_bounds_idx] = 0

        with np.errstate(invalid="ignore"):
            normal = special.ndtri(X_col)
            return np.clip(normal, _NORMAL_CLIP_MIN, _NORMAL_CLIP_MAX)


# The clip bounds are constants; sklearn recomputes them on every column of
# every call. Computed once here, exactly as sklearn defines them.
_NORMAL_CLIP_MIN = special.ndtri(BOUNDS_THRESHOLD - np.spacing(1))
_NORMAL_CLIP_MAX = special.ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1)))


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric and categorical feature names actually present in ``df``.

    Intersected with the frame rather than taken from the schema wholesale, so
    a column dropped during cleaning does not cause a KeyError here.
    """
    numeric = [c for c in schema.NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in schema.CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build the (unfitted) feature transformer."""
    numeric, categorical = feature_columns(df)

    numeric_pipeline = Pipeline(
        [
            # add_indicator is the load-bearing argument. Missing here means
            # "this never happened", which is signal; without the indicator the
            # median silently stands in for it and the distinction is lost.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "scale",
                FastNormalQuantileTransformer(
                    n_quantiles=N_QUANTILES,
                    output_distribution="normal",
                    subsample=200_000,
                    random_state=42,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    # addr_state has 51 levels and purpose has 14; a level seen
                    # fewer than 30 times in 389k rows cannot support a stable
                    # coefficient, so it folds into an infrequent bucket rather
                    # than adding a column that fits noise.
                    min_frequency=30,
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        # Identifiers, the target and issue_d must never reach the model.
        # issue_d in particular would be catastrophic under a chronological
        # split: it is perfectly correlated with the split boundary.
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(df: pd.DataFrame, estimator: object) -> Pipeline:
    """Preprocessor + estimator as one artifact.

    One object to log, register, load and serve. The API calls
    ``predict_proba`` on raw records and the fitted transforms travel with it —
    which is what stops training-time and serving-time preprocessing drifting
    apart.
    """
    return Pipeline([("preprocess", build_preprocessor(df)), ("model", estimator)])


def feature_schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of the feature contract: column names and category levels.

    Written into every prediction log record and checked by the promotion
    gates. Its purpose is to answer one question definitively: *are these two
    windows even comparable?* Drift analysis across a schema change is
    meaningless, and without a hash the change is invisible.
    """
    numeric, categorical = feature_columns(df)
    contract = {
        "numeric": sorted(numeric),
        "categorical": {
            col: sorted(str(v) for v in df[col].dropna().unique()) for col in sorted(categorical)
        },
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def split_xy(df: pd.DataFrame, target: str = "target") -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from label, dropping anything that is not a feature.

    ``issue_d`` is dropped here rather than relied upon being ignored
    downstream. Under a chronological split it encodes the split boundary
    exactly, so a model given it would "predict" the future by reading the
    date — and the held-out metrics would look excellent.
    """
    drop = [c for c in (schema.TIME_COLUMN, schema.TARGET_SOURCE, target) if c in df.columns]
    return df.drop(columns=drop), df[target]


__all__ = [
    "N_QUANTILES",
    "build_pipeline",
    "build_preprocessor",
    "feature_columns",
    "feature_schema_hash",
    "split_xy",
]
