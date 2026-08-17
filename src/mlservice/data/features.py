"""Feature transformation, fitted on train and carried with the model.

The transformer is part of the model artifact, never a separate preprocessing
step the API has to reproduce. Reimplementing encoding at serving time is one of
the most common ways a service silently serves garbage: the training pipeline
and the serving pipeline drift apart, both remain individually valid, and
nothing raises.

Two choices worth defending:

*   **``handle_unknown="infrequent_if_exist"``** on the categorical encoder. At
    serving time a category the model never saw must not raise — a single
    unfamiliar `medical_specialty` should degrade one prediction, not return a
    500. Phase 4's robustness tests assert exactly this.
*   **No imputation.** Cleaning already turned every absence into an explicit
    ``Unknown`` or ``NotMeasured`` category, because whether a clinician ordered
    a test is itself informative. There is nothing left to impute.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlservice.data import schema
from mlservice.logging_ import get_logger

log = get_logger(__name__)


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

    return ColumnTransformer(
        transformers=[
            # Scaling matters for regularised logistic regression: without it,
            # the L2 penalty is applied unevenly across features whose natural
            # scales differ by orders of magnitude (num_lab_procedures reaches
            # 100+, number_emergency is usually 0).
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=30,  # rarer levels fold into an infrequent bucket
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",  # identifiers and target must never reach the model
        verbose_feature_names_out=False,
    )


def build_pipeline(df: pd.DataFrame, estimator: object) -> Pipeline:
    """Preprocessor + estimator as one artifact.

    One object to log, register, load and serve. The API calls ``predict_proba``
    on raw records and the fitted transforms travel with it.
    """
    return Pipeline([("preprocess", build_preprocessor(df)), ("model", estimator)])


def feature_schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of the feature contract: column names, dtypes, categories.

    Written into every prediction log record in Phase 3 and checked by the
    Phase 7 promotion gates. Its purpose is to answer one question definitively:
    *are these two windows even comparable?* Drift analysis across a schema
    change is meaningless, and without a hash the change is invisible.
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
    """Separate features from label, dropping identifiers.

    Identifiers are removed here rather than relied upon being ignored
    downstream. ``patient_nbr`` in particular is a high-cardinality integer that
    a tree would happily split on, memorising individuals.
    """
    drop = [c for c in (*schema.IDENTIFIER_COLUMNS, target) if c in df.columns]
    return df.drop(columns=drop), df[target]


__all__ = [
    "build_pipeline",
    "build_preprocessor",
    "feature_columns",
    "feature_schema_hash",
    "split_xy",
]
