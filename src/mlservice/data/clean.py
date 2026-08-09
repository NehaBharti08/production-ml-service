"""Cleaning: leakage removal, missingness handling, diagnosis grouping.

Every step returns a record of what it did, so ``docs/DATA_AUDIT.md`` is
generated from the actual run rather than transcribed by hand. A hand-written
audit drifts from the code the first time anyone changes a threshold.

The ordering of steps matters and is deliberate:

1. Exclude expired/hospice discharges  — before anything measures a rate
2. Deduplicate to first encounter       — before the split, so it cannot straddle
3. Normalise missingness                — before any encoder sees a "?"
4. Group diagnoses                      — 717 ICD-9 codes into clinical bands
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mlservice.data import schema
from mlservice.logging_ import get_logger

log = get_logger(__name__)


@dataclass
class CleaningReport:
    """What cleaning actually did — the raw material for the audit document."""

    rows_in: int = 0
    rows_out: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, **detail: Any) -> None:
        self.steps.append({"step": step, **detail})
        log.info(f"clean_{step}", **detail)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out


def binarise_target(df: pd.DataFrame) -> pd.Series:
    """``readmitted`` -> 1 if readmitted within 30 days, else 0.

    ``>30`` collapses into the negative class with ``NO``. The clinical and
    operational question is specifically *early* readmission — that is what
    quality programmes penalise and what an intervention could plausibly
    prevent. A three-class model would answer a question nobody asked.
    """
    return (df[schema.TARGET] == schema.POSITIVE_LABEL).astype("int8")


def drop_excluded_discharges(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove expired and hospice discharges.

    Two distinct reasons, kept distinct because conflating them overstates the
    leakage finding:

    *   **Expired** (11, 19, 20, 21) is genuine label leakage. A dead patient
        cannot be readmitted; the label is deterministically negative and a
        model learns "died -> not readmitted" from the discharge code.
    *   **Hospice** (13, 14) is not leakage — these patients *are* readmitted,
        at 4.76% and 6.45%. They are excluded on clinical-relevance grounds:
        readmission is not a meaningful quality signal in end-of-life care.
    """
    y = binarise_target(df)

    expired = df[schema.ADMINISTRATIVE_FEATURES[1]].isin(schema.EXPIRED_DISCHARGE_IDS)
    hospice = df[schema.ADMINISTRATIVE_FEATURES[1]].isin(schema.HOSPICE_DISCHARGE_IDS)

    report.record(
        "exclude_expired",
        reason="deterministic label — a dead patient cannot be readmitted",
        n_rows=int(expired.sum()),
        n_positive=int(y[expired].sum()),
        positive_rate=round(float(y[expired].mean()), 6) if expired.any() else 0.0,
        codes=list(schema.EXPIRED_DISCHARGE_IDS),
        classification="LEAKAGE",
    )
    report.record(
        "exclude_hospice",
        reason="end-of-life care — readmission is not a quality signal here",
        n_rows=int(hospice.sum()),
        n_positive=int(y[hospice].sum()),
        positive_rate=round(float(y[hospice].mean()), 6) if hospice.any() else 0.0,
        codes=list(schema.HOSPICE_DISCHARGE_IDS),
        classification="CLINICAL_JUDGEMENT",  # deliberately NOT called leakage
    )

    return df.loc[~(expired | hospice)].copy()


def keep_first_encounter(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Keep only each patient's earliest encounter.

    101,766 encounters span ~71,518 patients. Without this, the same patient
    appears on both sides of any split and the held-out estimate is inflated by
    memorised patient-level idiosyncrasy rather than learned signal.

    Costs the prior-utilisation richness of repeat visits, which is a real loss
    — the audit reports an all-encounter sensitivity variant so the size of the
    inflation is visible rather than assumed.
    """
    before = len(df)
    out = (
        df.sort_values(schema.ENCOUNTER_ID)
        .drop_duplicates(subset=schema.PATIENT_ID, keep="first")
        .copy()
    )
    report.record(
        "first_encounter_only",
        reason="prevent the same patient straddling the split",
        rows_before=before,
        rows_after=len(out),
        rows_removed=before - len(out),
        unique_patients=int(df[schema.PATIENT_ID].nunique()),
    )
    return out


def normalise_missing(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Turn sentinels into explicit categories. Never impute.

    Two different kinds of absence, handled differently because they mean
    different things:

    *   ``"?"`` -> ``"Unknown"``. Not recorded.
    *   ``NaN`` in a lab column -> ``"NotMeasured"``. The test was not ordered,
        which is a *clinical decision* and therefore informative. Imputing it
        with a modal value would destroy real signal and invent a measurement
        that was never taken.
    """
    out = df.copy()
    detail: dict[str, dict[str, Any]] = {}

    for col in out.columns:
        if out[col].dtype == object:
            n = int((out[col] == schema.MISSING_SENTINEL).sum())
            if n:
                out[col] = out[col].replace(schema.MISSING_SENTINEL, schema.UNKNOWN_CATEGORY)
                detail[col] = {
                    "sentinel": n,
                    "pct": round(n / len(out) * 100, 2),
                    "action": "-> Unknown category",
                }

    for col in schema.NOT_MEASURED_COLUMNS:
        if col in out.columns:
            n = int(out[col].isna().sum())
            if n:
                out[col] = out[col].fillna(schema.NOT_MEASURED_CATEGORY)
                detail[col] = {
                    "native_nan": n,
                    "pct": round(n / len(out) * 100, 2),
                    "action": "-> NotMeasured category (test not ordered)",
                }

    report.record("normalise_missing", columns_affected=len(detail), detail=detail)
    return out


def group_diagnoses(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Collapse ICD-9 codes into clinical bands.

    ``diag_1`` alone has 717 distinct values, many with a handful of rows.
    One-hot encoding raw codes would produce a very wide, very sparse matrix
    whose rare columns are pure noise under any split.

    Bands follow the grouping used in the original Strack et al. (2014) paper,
    so the results stay comparable to the published literature.
    """
    out = df.copy()

    def band(code: object) -> str:
        text = str(code)
        if text in ("Unknown", "?", "nan", "None"):
            return "Unknown"
        # V codes (supplementary) and E codes (external cause) are not numeric.
        if text.startswith(("V", "E")):
            return "Other"
        try:
            value = float(text)
        except ValueError:
            return "Other"
        if 390 <= value <= 459 or value == 785:
            return "Circulatory"
        if 460 <= value <= 519 or value == 786:
            return "Respiratory"
        if 520 <= value <= 579 or value == 787:
            return "Digestive"
        if int(value) == 250:
            return "Diabetes"
        if 800 <= value <= 999:
            return "Injury"
        if 710 <= value <= 739:
            return "Musculoskeletal"
        if 580 <= value <= 629 or value == 788:
            return "Genitourinary"
        if 140 <= value <= 239:
            return "Neoplasms"
        return "Other"

    cardinality_before = {c: int(out[c].nunique()) for c in schema.DIAGNOSIS_FEATURES}
    for col in schema.DIAGNOSIS_FEATURES:
        out[col] = out[col].map(band)

    report.record(
        "group_diagnoses",
        reason="717 raw ICD-9 codes would one-hot into mostly-noise columns",
        cardinality_before=cardinality_before,
        cardinality_after={c: int(out[c].nunique()) for c in schema.DIAGNOSIS_FEATURES},
        bands=sorted(out[schema.DIAGNOSIS_FEATURES[0]].unique().tolist()),
    )
    return out


def drop_unusable_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop columns that cannot legitimately contribute, each with its reason."""
    present = [c for c in schema.DROPPED_COLUMNS if c in df.columns]
    report.record(
        "drop_columns",
        columns={c: schema.DROPPED_COLUMNS[c] for c in present},
        n_dropped=len(present),
    )
    return df.drop(columns=present)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return the data plus its report."""
    report = CleaningReport(rows_in=len(df))

    out = drop_excluded_discharges(df, report)
    out = keep_first_encounter(out, report)
    out = normalise_missing(out, report)
    out = group_diagnoses(out, report)
    out = drop_unusable_columns(out, report)

    out["target"] = binarise_target(out)
    out = out.drop(columns=[schema.TARGET])

    report.rows_out = len(out)
    report.record(
        "complete",
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        rows_removed=report.rows_removed,
        pct_removed=round(report.rows_removed / report.rows_in * 100, 2),
        positive_rate=round(float(out["target"].mean()), 6),
    )
    return out, report


__all__ = [
    "CleaningReport",
    "binarise_target",
    "clean",
    "drop_excluded_discharges",
    "drop_unusable_columns",
    "group_diagnoses",
    "keep_first_encounter",
    "normalise_missing",
]
