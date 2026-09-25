"""The canonical column contract, shared by training and the API.

One definition, imported by both sides. The alternative — a feature list in the
training code and a Pydantic model written separately for the API — is how a
model ends up scoring columns the caller never sent, in an order nobody
checked. ``feature_schema_hash`` is computed from this module, travels with the
artifact, and is compared at load time by the operational gate.

Everything here is derived from the cleaned frame rather than transcribed from
the data dictionary. That distinction is not pedantry: the dictionary describes
``loan_status`` as a loan attribute, which is true right up until it becomes the
label, and writing the schema from it leaked the target back in as a feature.
"""

from __future__ import annotations

import hashlib
from typing import Final

# --------------------------------------------------------------------------- #
# Identity and target
# --------------------------------------------------------------------------- #

#: The raw status column. NOT a feature — the label is derived from it, which
#: makes it a perfect predictor. Dropped in cleaning; named here so the reason
#: is recorded somewhere a reader will look.
TARGET_SOURCE: Final = "loan_status"

#: The binary label written by cleaning: 1 = charged off / defaulted.
TARGET: Final = "target"

#: A REAL date, unlike the medical version's ID proxy. Used to order the
#: chronological split and to bucket drift windows. Never a feature: it is the
#: one column guaranteed to differ between train and serve.
TIME_COLUMN: Final = "issue_d"

UNKNOWN_CATEGORY: Final = "Unknown"

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

#: Borrower finances and credit-bureau history, as known at origination.
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "credit_history_months",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "delinq_2yrs",
    "delinq_amnt",
    "acc_now_delinq",
    "acc_open_past_24mths",
    "inq_last_6mths",
    "pub_rec",
    "pub_rec_bankruptcies",
    "tax_liens",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "total_rev_hi_lim",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "percent_bc_gt_75",
    "pct_tl_nvr_dlq",
    "mort_acc",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
)

#: Loan terms and borrower attributes that arrive as labels rather than numbers.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
)

ALL_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Lending Club's OWN risk assessment, produced by their model at origination.
#:
#: Keeping these is a deliberate, arguable choice. They are legitimately
#: available at scoring time, so they are not leakage — but they make this
#: model partly a function of theirs, and a drift in their grading policy would
#: propagate here as a silent distribution shift. Recorded so the choice is
#: visible, and monitored per-feature for exactly that reason.
#: See docs/DECISIONS/0009-lender-grade-as-a-feature.md.
LENDER_ASSESSMENT_FEATURES: Final[tuple[str, ...]] = ("grade", "sub_grade", "int_rate")

# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #

#: Subgroups for disparate-impact analysis.
#:
#: US credit data legally EXCLUDES race, gender and marital status, so there
#: are no protected attributes to slice on directly. Fair-lending practice uses
#: geography and socioeconomic position as proxies instead — which is precisely
#: what redlining analysis has always done — and that is what these are.
#:
#: Reported openly, including where unflattering. A disparity found here is not
#: proof of discrimination; it is the signal that prompts someone to look.
SUBGROUP_DIMENSIONS: Final[tuple[str, ...]] = (
    "addr_state",
    "emp_length",
    "home_ownership",
    "income_band",
)

#: Derived at evaluation time from ``annual_inc``. Quantile-based rather than
#: fixed-dollar so the bands stay populated across an eleven-year span in which
#: incomes drifted.
INCOME_BAND_QUANTILES: Final[tuple[float, ...]] = (0.0, 0.25, 0.50, 0.75, 1.0)
INCOME_BAND_LABELS: Final[tuple[str, ...]] = ("Q1_lowest", "Q2", "Q3", "Q4_highest")

# --------------------------------------------------------------------------- #
# Expectations — asserted against the real file in the data-quality suite
# --------------------------------------------------------------------------- #

RAW_ROW_COUNT: Final = 2_260_701
RAW_COLUMN_COUNT: Final = 151
CLEANED_ROW_COUNT: Final = 672_379
CLEANED_COLUMN_COUNT: Final = 66

#: Observed on the cleaned frame. A drift of more than a point or so means the
#: cleaning rules changed, and every downstream number changed with them.
CLEANED_POSITIVE_RATE: Final = 0.1481


def expected_feature_columns() -> tuple[str, ...]:
    """Every column the model consumes, in a stable order.

    Order is part of the contract. A ColumnTransformer fitted on one order and
    fed another does not raise — it silently scores the wrong columns.
    """
    return ALL_FEATURES


def feature_schema_hash() -> str:
    """A short, stable hash of the feature contract.

    Recorded beside the artifact and compared at load time. If the training
    contract and the serving contract diverge, the operational gate fails
    instead of the service quietly scoring mismatched columns.
    """
    payload = "|".join(ALL_FEATURES).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "CLEANED_COLUMN_COUNT",
    "CLEANED_POSITIVE_RATE",
    "CLEANED_ROW_COUNT",
    "INCOME_BAND_LABELS",
    "INCOME_BAND_QUANTILES",
    "LENDER_ASSESSMENT_FEATURES",
    "NUMERIC_FEATURES",
    "RAW_COLUMN_COUNT",
    "RAW_ROW_COUNT",
    "SUBGROUP_DIMENSIONS",
    "TARGET",
    "TARGET_SOURCE",
    "TIME_COLUMN",
    "UNKNOWN_CATEGORY",
    "expected_feature_columns",
    "feature_schema_hash",
]
