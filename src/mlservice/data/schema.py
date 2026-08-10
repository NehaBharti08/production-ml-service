"""Canonical column specification, shared by training and the API.

One definition of "what a valid record looks like", imported by the training
pipeline and by the Pydantic request models in Phase 3. When these two drift
apart, the service starts serving predictions on features the model was never
trained on — and nothing raises, because both halves are individually valid.

Every exclusion below carries its reason. Dropping a column is a modelling
decision, and an undocumented one is indistinguishable from an oversight.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Identifiers — never features
# --------------------------------------------------------------------------- #

ENCOUNTER_ID: Final = "encounter_id"
PATIENT_ID: Final = "patient_nbr"
TARGET: Final = "readmitted"
POSITIVE_LABEL: Final = "<30"

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (ENCOUNTER_ID, PATIENT_ID)

# --------------------------------------------------------------------------- #
# Missingness sentinels
# --------------------------------------------------------------------------- #

#: This dataset encodes missing values as a literal "?" rather than an empty
#: field, so pandas reads them as ordinary strings. Anything that counts nulls
#: without normalising this first will report zero missingness and be wrong.
MISSING_SENTINEL: Final = "?"

#: Columns where a native NaN means "the test was not ordered" — a clinical
#: decision, not an absent measurement. These become an explicit category
#: rather than being imputed: whether a clinician ordered an HbA1c is itself
#: informative, and imputing it destroys that signal.
NOT_MEASURED_COLUMNS: Final[tuple[str, ...]] = ("max_glu_serum", "A1Cresult")
NOT_MEASURED_CATEGORY: Final = "NotMeasured"
UNKNOWN_CATEGORY: Final = "Unknown"

# --------------------------------------------------------------------------- #
# Exclusions, each with its reason
# --------------------------------------------------------------------------- #

#: measured in the audit -> reason for dropping
DROPPED_COLUMNS: Final[dict[str, str]] = {
    "weight": (
        "96.86% missing. Imputing a value present in 3% of records would "
        "manufacture a variable rather than measure one. The honest move is to "
        "drop it and say so."
    ),
    "examide": "Zero variance — constant 'No' across all 101,766 rows.",
    "citoglipton": "Zero variance — constant 'No' across all 101,766 rows.",
    "payer_code": (
        "TIME-CONFOUNDED, not merely missing. Capture rises from 0% in the "
        "first encounter_id decile to 86% in the last, because the field was "
        "rolled out mid-period. Under a chronological split the model would "
        "learn 'payer_code present' as a proxy for 'later era' — an artifact of "
        "data capture with no clinical meaning, which cannot generalise. See "
        "docs/DECISIONS/0004-temporal-split-proxy.md."
    ),
}

#: discharge_disposition_id values where the patient died. Verified against
#: IDS_mapping.csv AND against the data: 0 positives in 1,652 such rows.
#: A dead patient cannot be readmitted, so the label is deterministic and a
#: model would learn "died -> not readmitted" from the discharge code alone.
EXPIRED_DISCHARGE_IDS: Final[tuple[int, ...]] = (11, 19, 20, 21)

#: Hospice discharges. Excluded too, but for a DIFFERENT reason, and the
#: distinction matters: these are NOT deterministic — observed positive rates
#: are 4.76% (code 13) and 6.45% (code 14), so hospice patients genuinely are
#: readmitted. They are excluded as a clinical-relevance judgement (readmission
#: is not a meaningful quality signal in end-of-life care), not as leakage.
#: Calling this "leakage" would overstate the finding.
HOSPICE_DISCHARGE_IDS: Final[tuple[int, ...]] = (13, 14)

EXCLUDED_DISCHARGE_IDS: Final[tuple[int, ...]] = EXPIRED_DISCHARGE_IDS + HOSPICE_DISCHARGE_IDS

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)

DEMOGRAPHIC_FEATURES: Final[tuple[str, ...]] = ("race", "gender", "age")

#: Kept as categorical, not numeric. They are unordered code sets — an integer
#: encoding would imply admission_type_id 8 is "more" than 2, which is meaningless.
ADMINISTRATIVE_FEATURES: Final[tuple[str, ...]] = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
)

DIAGNOSIS_FEATURES: Final[tuple[str, ...]] = ("diag_1", "diag_2", "diag_3")

LAB_FEATURES: Final[tuple[str, ...]] = ("max_glu_serum", "A1Cresult")

#: Medication columns retained. The near-zero-variance ones are kept for now
#: (a 0.03%-prevalence drug still carries a little signal and costs one column);
#: Phase 2 revisits whether they earn their place.
MEDICATION_FEATURES: Final[tuple[str, ...]] = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)

TREATMENT_FEATURES: Final[tuple[str, ...]] = ("change", "diabetesMed")

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    DEMOGRAPHIC_FEATURES
    + ADMINISTRATIVE_FEATURES
    + DIAGNOSIS_FEATURES
    + LAB_FEATURES
    + MEDICATION_FEATURES
    + TREATMENT_FEATURES
)

ALL_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Dimensions for the subgroup analysis in Phase 2. Reported openly, including
#: where the results are unflattering.
SUBGROUP_DIMENSIONS: Final[tuple[str, ...]] = ("race", "gender", "age")

# --------------------------------------------------------------------------- #
# Expected raw shape — asserted before anything downstream runs
# --------------------------------------------------------------------------- #

RAW_ROW_COUNT: Final = 101_766
RAW_COLUMN_COUNT: Final = 50
RAW_PATIENT_COUNT: Final = 71_518
# NOTE: the archive checksum deliberately lives ONLY in data/checksums.txt,
# which download.py reads and enforces. Duplicating it here would create two
# sources of truth that can silently disagree.


def expected_raw_columns() -> tuple[str, ...]:
    """Every column expected in the raw file, in no particular order."""
    return (*IDENTIFIER_COLUMNS, TARGET, *ALL_FEATURES, *DROPPED_COLUMNS)


__all__ = [
    "ADMINISTRATIVE_FEATURES",
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "DEMOGRAPHIC_FEATURES",
    "DIAGNOSIS_FEATURES",
    "DROPPED_COLUMNS",
    "ENCOUNTER_ID",
    "EXCLUDED_DISCHARGE_IDS",
    "EXPIRED_DISCHARGE_IDS",
    "HOSPICE_DISCHARGE_IDS",
    "IDENTIFIER_COLUMNS",
    "LAB_FEATURES",
    "MEDICATION_FEATURES",
    "MISSING_SENTINEL",
    "NOT_MEASURED_COLUMNS",
    "NUMERIC_FEATURES",
    "PATIENT_ID",
    "POSITIVE_LABEL",
    "RAW_COLUMN_COUNT",
    "RAW_PATIENT_COUNT",
    "RAW_ROW_COUNT",
    "SUBGROUP_DIMENSIONS",
    "TARGET",
    "TREATMENT_FEATURES",
    "expected_raw_columns",
]
