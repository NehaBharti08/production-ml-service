"""Assemble the evidence the promotion gates judge.

The gates were the easy part. This module is the part that is easy to skip and
then quietly never notice: gates that are unit-tested but never wired to the
pipeline's real output are gates that have never blocked anything.

The training summary carries most of what is needed — PR-AUC and Brier with
bootstrap intervals, calibration, subgroup gaps, the feature schema hash. Three
things it does not carry, because they are not properties of a training run:

*   whether the artifact **loads and scores** (an operational question),
*   whether the **behavioural suite** passes (a separate test run),
*   whether the **data-quality suite** passes (a separate test run).

Those are collected here, and where they cannot be collected they are left
**absent rather than assumed**. The gates fail closed on missing evidence — a
model is not promoted because nobody checked. Defaulting them to "passing"
would turn every gate into a formality, which is the failure mode this whole
phase exists to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlservice.config import get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)


def _champion_record(summary: dict[str, Any]) -> dict[str, Any]:
    name = summary["champion"]
    for candidate in summary["candidates"]:
        if candidate["name"] == name:
            return dict(candidate)
    raise KeyError(f"champion {name!r} is not among the candidates in the summary")


def canary_inference() -> tuple[bool, str]:
    """Load the serving artifact and score one row.

    "The file exists" is not the question. A pickle can deserialise and then
    raise on the first ``predict_proba`` because a transformer inside it was
    fitted against a different column set — which is discovered in production
    as a 500 on every request. So this scores a real row.
    """
    try:
        from mlservice.data import features as feature_mod
        from mlservice.monitoring import reports as reports_mod

        model = reports_mod.load_model()
        reference = reports_mod.load_reference()
        if reference.empty:
            return False, "reference window is empty — cannot run a canary inference"

        # Exactly the path the API takes: split the target off, hand the model a
        # record dict. Not `pipeline.predict_proba(frame)`.
        #
        # Scoring the pipeline directly is the mistake this canary exists to
        # catch, and it caught it here first: `LoadedModel.predict_proba` builds
        # its own frame from a record, so passing a DataFrame double-wraps it
        # into a 3-d array and raises. A canary that talks to the model by a
        # different route than production does is not a canary — it is a second
        # code path that can pass while the real one is broken.
        x, _ = feature_mod.split_xy(reference.head(1))
        record = x.to_dict("records")[0]

        proba = model.predict_proba(record)
        if not (0.0 <= proba <= 1.0):
            return False, f"canary produced {proba!r}, which is not a probability"
        return True, f"artifact loaded and scored one row via the serving path -> {proba:.4f}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def build_evidence(
    summary: dict[str, Any],
    behavioral: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    run_canary: bool = True,
) -> dict[str, Any]:
    """Project a training summary into the shape the gates read.

    ``behavioral`` and ``data_quality`` are passed in rather than computed here
    because they come from pytest runs, and a module that shells out to its own
    test suite is a module that cannot be tested.
    """
    champion = _champion_record(summary)
    calibration = champion.get("calibration", {})
    subgroups = champion.get("subgroups", {})

    loads, canary_reason = (True, "not run") if not run_canary else canary_inference()

    evidence: dict[str, Any] = {
        "model": summary["champion"],
        "registered_version": summary.get("registered_version"),
        "pr_auc": champion["pr_auc"],
        "brier": champion["brier"],
        "calibration": {"ece": calibration.get("ece"), "mce": calibration.get("mce")},
        "subgroups": {
            "worst_recall_gap": subgroups.get("worst_recall_gap"),
            "worst_group": subgroups.get("worst_group"),
            "n_analysable": subgroups.get("n_analysable"),
        },
        "artifact_loads": loads,
        "canary": canary_reason,
        "feature_schema_hash": summary["feature_schema_hash"],
    }

    # Absent, not assumed. `behavioral: {}` makes the gate fail with "the suite
    # did not run", which is the correct verdict — and a materially different
    # message from "a behavioural test failed".
    if behavioral is not None:
        evidence["behavioral"] = behavioral
    if data_quality is not None:
        evidence["data_quality"] = data_quality

    missing = [k for k in ("behavioral", "data_quality") if k not in evidence]
    if missing:
        evidence["evidence_gaps"] = missing
        log.warning(
            "gate_evidence_incomplete",
            missing=missing,
            consequence="those gates will block; collect the evidence rather than defaulting it",
        )

    return evidence


def parse_pytest_report(path: Path) -> dict[str, Any]:
    """Read a ``pytest --json-report`` summary into the behavioural gate's shape.

    Counts errors and failures alike as not-passed. A suite that errored did not
    demonstrate the property it exists to demonstrate, and treating an error as
    anything other than a failure is how a broken test file becomes a silent
    pass.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    total = int(summary.get("total", 0))
    passed_count = int(summary.get("passed", 0))

    failures = [
        test.get("nodeid", "?")
        for test in payload.get("tests", [])
        if test.get("outcome") not in {"passed", "skipped"}
    ]
    return {"total": total, "passed": passed_count, "failures": failures}


def load_summary(path: Path | None = None) -> dict[str, Any]:
    target = path or (get_settings().paths.reports / "training_summary.json")
    return dict(json.loads(target.read_text(encoding="utf-8")))


def save_evidence(evidence: dict[str, Any], path: Path | None = None) -> Path:
    target = path or (get_settings().paths.reports / "gate_evidence.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    log.info("gate_evidence_written", path=str(target), gaps=evidence.get("evidence_gaps", []))
    return target


__all__ = [
    "build_evidence",
    "canary_inference",
    "load_summary",
    "parse_pytest_report",
    "save_evidence",
]
