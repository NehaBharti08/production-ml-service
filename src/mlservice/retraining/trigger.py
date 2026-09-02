"""When to retrain.

Four triggers, and — just as important — an explicit list of things that are
**not** triggers. Retraining on noise burns compute and, worse, ships a model
nobody asked for on evidence nobody checked.

The ordering of trust matters. Drift is a *leading* signal: it says the input
population moved, which is a reason to look, not proof the model got worse.
Performance on matured labels is the *lagging* ground truth. A system that
retrains on drift alone will retrain on population changes the model handles
perfectly well.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from mlservice.config import get_thresholds
from mlservice.logging_ import get_logger

log = get_logger(__name__)


@dataclass
class TriggerResult:
    name: str
    fired: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerDecision:
    retrain: bool
    triggers: list[TriggerResult] = field(default_factory=list)

    @property
    def fired(self) -> list[TriggerResult]:
        return [t for t in self.triggers if t.fired]

    @property
    def reason(self) -> str:
        if not self.retrain:
            return "no trigger fired"
        return "fired: " + ", ".join(t.name for t in self.fired)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrain": self.retrain,
            "reason": self.reason,
            "fired": [t.name for t in self.fired],
            "triggers": [t.to_dict() for t in self.triggers],
        }


def scheduled_trigger(last_trained: datetime | None, now: datetime | None = None) -> TriggerResult:
    """Monthly refresh, regardless of drift.

    Baseline freshness matters even when nothing alarms: a model can decay
    slowly enough that no single window breaches while the cumulative shift is
    material. A schedule bounds that.
    """
    config = get_thresholds().model_dump()["retraining"]["triggers"]["scheduled"]
    if not config["enabled"]:
        return TriggerResult("scheduled", False, "scheduled retraining is disabled")

    now = now or datetime.now(UTC)
    if last_trained is None:
        return TriggerResult(
            "scheduled", True, "no previous training recorded", {"last_trained": None}
        )

    age = now - last_trained
    due = age >= timedelta(days=30)
    return TriggerResult(
        name="scheduled",
        fired=due,
        reason=(
            f"model is {age.days} days old, past the 30-day refresh"
            if due
            else f"model is {age.days} days old, refresh not yet due"
        ),
        detail={"last_trained": last_trained.isoformat(), "age_days": age.days},
    )


def drift_trigger(drift_alert: dict[str, Any]) -> TriggerResult:
    """Confirmed data drift — sustained, not a single window.

    Requires the same confirmation the alert does: a single-window breach is
    noise, and retraining on noise produces a model justified by nothing.
    """
    config = get_thresholds().model_dump()["retraining"]["triggers"]["drift"]
    if not config["enabled"]:
        return TriggerResult("drift", False, "drift-triggered retraining is disabled")

    confirmed = bool(drift_alert.get("confirmed"))
    return TriggerResult(
        name="drift",
        fired=confirmed,
        reason=(
            f"confirmed data drift — {drift_alert.get('reason', '')}"
            if confirmed
            else "no confirmed drift"
        ),
        detail={
            "confirmed": confirmed,
            "breaching_counts": drift_alert.get("breaching_counts", []),
            "required_windows": drift_alert.get("required_consecutive_windows"),
        },
    )


def performance_trigger(label_drift: dict[str, Any]) -> TriggerResult:
    """Measured degradation on matured labels — the signal that matters.

    Only this trigger observes actual performance. The others are proxies for
    it, which is why a performance trigger firing is stronger evidence than a
    drift trigger firing.
    """
    config = get_thresholds().model_dump()["retraining"]["triggers"]["performance"]
    if not config["enabled"]:
        return TriggerResult("performance", False, "performance-triggered retraining is disabled")

    if not label_drift.get("sufficient_labels"):
        return TriggerResult(
            name="performance",
            fired=False,
            reason=(
                "insufficient matured labels to judge performance — "
                f"{label_drift.get('n_matured', 0)} available"
            ),
            detail={"n_matured": label_drift.get("n_matured", 0)},
        )

    breaching = bool(label_drift.get("breaching"))
    return TriggerResult(
        name="performance",
        fired=breaching,
        reason=(
            f"PR-AUC dropped {label_drift.get('drop_in_standard_errors')} SE below baseline"
            if breaching
            else "performance within tolerance on matured labels"
        ),
        detail={
            "observed_pr_auc": label_drift.get("observed_pr_auc"),
            "baseline_pr_auc": label_drift.get("baseline_pr_auc"),
            "drop_in_standard_errors": label_drift.get("drop_in_standard_errors"),
            "threshold_se": label_drift.get("threshold_se"),
            "n_matured": label_drift.get("n_matured"),
        },
    )


def manual_trigger(requested: bool, requested_by: str | None = None) -> TriggerResult:
    """Human override. Always available, always recorded.

    An automated system that cannot be driven by hand is one people work around.
    Recording who asked keeps the audit trail intact.
    """
    return TriggerResult(
        name="manual",
        fired=requested,
        reason=f"manual retrain requested by {requested_by or 'unknown'}"
        if requested
        else "no manual request",
        detail={"requested_by": requested_by},
    )


def evaluate_triggers(
    last_trained: datetime | None = None,
    drift_alert: dict[str, Any] | None = None,
    label_drift: dict[str, Any] | None = None,
    manual: bool = False,
    requested_by: str | None = None,
    now: datetime | None = None,
) -> TriggerDecision:
    """Evaluate every trigger. Any one firing is sufficient.

    All are evaluated rather than short-circuited, so the record shows the full
    state at the moment of the decision — "we retrained on drift, and note that
    performance was also degrading" is a materially different story from
    "we retrained on drift".
    """
    triggers = [
        scheduled_trigger(last_trained, now),
        drift_trigger(drift_alert or {}),
        performance_trigger(label_drift or {}),
        manual_trigger(manual, requested_by),
    ]

    decision = TriggerDecision(retrain=any(t.fired for t in triggers), triggers=triggers)

    log.info(
        "retraining_triggers_evaluated",
        retrain=decision.retrain,
        reason=decision.reason,
        **{f"trigger_{t.name}": t.fired for t in triggers},
    )
    return decision


#: Conditions deliberately excluded, mirrored from configs/thresholds.yaml.
#: Written down because "what does NOT trigger a retrain" is the half people
#: forget, and an over-eager trigger is how a pipeline ends up shipping models
#: continuously on evidence nobody examined.
NOT_TRIGGERS = (
    "a single feature breaching in a single window",
    "prediction drift without a corresponding data-drift signal",
    "a latency or error-rate alert — those are serving problems, not model problems",
    "an upstream schema break — that needs a pipeline fix, and retraining on it "
    "would bake the break into the model",
)


__all__ = [
    "NOT_TRIGGERS",
    "TriggerDecision",
    "TriggerResult",
    "drift_trigger",
    "evaluate_triggers",
    "manual_trigger",
    "performance_trigger",
    "scheduled_trigger",
]
