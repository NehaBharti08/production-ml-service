"""Promotion and rollback.

Both operations are a **single registry alias flip**. That is the whole design:

*   Promotion points ``champion`` at the challenger's version.
*   Rollback points ``champion`` back at the previous version.

One atomic operation each, no multi-step state machine that can be interrupted
halfway leaving the system in a state nobody designed. The API loads by alias,
so a flip takes effect on the next model load without a deploy.

**Rollback is only real if it is tested.** A rollback path that has never been
exercised is a plan, not a capability — and the moment you need it is the worst
possible time to discover it does not work. :func:`verify_rollback_path` runs the
whole cycle against the registry and is invoked by the CLI, so the claim in the
README is backed by something that actually ran.

The previous version is recorded *before* promotion rather than looked up
afterwards. After the flip, "what was it before" is no longer answerable from
the alias, and reconstructing it from run history is exactly the archaeology you
do not want to be doing during an incident.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlservice.config import get_settings
from mlservice.logging_ import get_logger
from mlservice.models import registry

log = get_logger(__name__)


@dataclass
class PromotionRecord:
    """The audit trail. Written on every promotion and every rollback."""

    timestamp_utc: str
    action: str  # "promote" | "rollback"
    model_name: str
    alias: str
    from_version: str | None
    to_version: str
    trigger: str
    gates_passed: bool
    approver: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _history_path() -> Path:
    return get_settings().paths.reports / "promotion_history.ndjson"


def record(entry: PromotionRecord) -> None:
    """Append to the promotion log.

    Append-only NDJSON, for the same reason the prediction log is: the history
    of what was deployed when is evidence, and evidence that can be rewritten is
    not evidence.
    """
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
    log.info("promotion_recorded", **entry.to_dict())


def history() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def current_champion() -> str | None:
    settings = get_settings()
    return registry.current_version(settings.model.name, settings.model.serving_alias)


def promote(
    version: str,
    trigger: str,
    gates_passed: bool,
    approver: str = "automated",
    detail: dict[str, Any] | None = None,
) -> PromotionRecord:
    """Point the serving alias at ``version``.

    Refuses when the gates did not pass. The gates are the whole control; a
    promote function that would bypass them on request is a promote function
    with no gates.
    """
    if not gates_passed:
        raise ValueError(
            "refusing to promote: promotion gates did not pass. "
            "Fix the failing gate or make the override explicit and recorded."
        )

    settings = get_settings()
    from mlflow.tracking import MlflowClient

    # Captured BEFORE the flip. Afterwards the alias no longer knows.
    previous = current_champion()

    MlflowClient().set_registered_model_alias(
        settings.model.name, settings.model.serving_alias, version
    )

    entry = PromotionRecord(
        timestamp_utc=datetime.now(UTC).isoformat(),
        action="promote",
        model_name=settings.model.name,
        alias=settings.model.serving_alias,
        from_version=previous,
        to_version=version,
        trigger=trigger,
        gates_passed=gates_passed,
        approver=approver,
        detail=detail or {},
    )
    record(entry)
    return entry


def rollback(reason: str, approver: str = "automated") -> PromotionRecord:
    """Point the serving alias back at the previous version.

    Reads the previous version from the promotion history rather than from the
    registry, because the registry only knows where the alias points *now*. This
    is precisely why :func:`promote` records ``from_version`` before flipping.
    """
    settings = get_settings()
    entries = [e for e in history() if e["action"] == "promote"]
    if not entries:
        raise ValueError(
            "no promotion history — nothing to roll back to. The previous "
            "version is recorded at promotion time, not reconstructed."
        )

    last = entries[-1]
    target = last["from_version"]
    if target is None:
        raise ValueError(
            f"the last promotion (to version {last['to_version']}) had no previous "
            "version — this is the first model, so there is nothing to roll back to."
        )

    from mlflow.tracking import MlflowClient

    current = current_champion()
    MlflowClient().set_registered_model_alias(
        settings.model.name, settings.model.serving_alias, target
    )

    entry = PromotionRecord(
        timestamp_utc=datetime.now(UTC).isoformat(),
        action="rollback",
        model_name=settings.model.name,
        alias=settings.model.serving_alias,
        from_version=current,
        to_version=target,
        trigger="rollback",
        gates_passed=False,  # a rollback is not gated; that is the point
        approver=approver,
        detail={"reason": reason, "rolled_back_from_promotion": last["timestamp_utc"]},
    )
    record(entry)
    log.warning("rollback_executed", from_version=current, to_version=target, reason=reason)
    return entry


def verify_rollback_path() -> dict[str, Any]:
    """Exercise a REAL promote -> rollback transition and report.

    A rollback path that has never run is a plan, not a capability.

    The first version of this function passed while proving nothing: it promoted
    whichever version the registry returned first, which happened to be the one
    already serving, so it recorded "promote 2 -> 2, rollback 2 -> 2, verified"
    — a no-op reporting success. A verification that cannot fail is decoration.

    So the cycle is now explicitly constructed to move:

        seed the alias to the OLDEST version
        promote the LATEST                     (a real transition)
        roll back                              (must return to OLDEST)

    and it FAILS if any step is degenerate. The original alias target is
    restored on the way out, so running this never changes what is serving.
    """
    settings = get_settings()
    registry.setup_tracking()

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.model.name}'")
    if len(versions) < 2:
        return {
            "verified": False,
            "reason": (
                f"only {len(versions)} registered version(s) of "
                f"{settings.model.name!r} — a rollback needs a previous version "
                "to return to. Train again to create one."
            ),
            "steps": [],
        }

    # str() is load-bearing, not decoration. MLflow's search_model_versions
    # returns `.version` as an int here, while get_model_version_by_alias
    # returns it as a str — so the obvious `after_promote == latest` compared
    # "2" against 2, reported "alias is 2, expected 2", and failed a cycle that
    # had in fact worked perfectly. Normalise at the boundary, once.
    oldest = str(min(versions, key=lambda v: int(v.version)).version)
    latest = str(max(versions, key=lambda v: int(v.version)).version)
    original = current_champion()
    steps: list[dict[str, Any]] = []

    try:
        # Start from a known state so the transition is real rather than
        # whatever the alias happened to be pointing at.
        client.set_registered_model_alias(settings.model.name, settings.model.serving_alias, oldest)
        steps.append({"step": "seed", "to": oldest, "alias_now": current_champion()})

        promoted = promote(
            version=latest,
            trigger="rollback_verification",
            gates_passed=True,
            approver="verify_rollback_path",
            detail={"note": "synthetic promotion to exercise the rollback path"},
        )
        after_promote = current_champion()
        steps.append(
            {
                "step": "promote",
                "from": promoted.from_version,
                "to": promoted.to_version,
                "alias_now": after_promote,
            }
        )

        if promoted.from_version == promoted.to_version:
            return {
                "verified": False,
                "reason": (
                    "degenerate promotion: from_version == to_version, so nothing was exercised"
                ),
                "steps": steps,
            }

        rolled = rollback(reason="rollback path verification", approver="verify_rollback_path")
        after_rollback = current_champion()
        steps.append(
            {
                "step": "rollback",
                "from": rolled.from_version,
                "to": rolled.to_version,
                "alias_now": after_rollback,
            }
        )

        # The actual assertion: the alias must have MOVED to the promoted
        # version and then MOVED BACK to where it started.
        moved = after_promote == latest
        returned = after_rollback == oldest
        verified = moved and returned

        if verified:
            reason = f"alias moved {oldest} -> {latest} and back to {oldest}"
        elif not moved:
            reason = f"promotion did not take effect: alias is {after_promote}, expected {latest}"
        else:
            reason = f"rollback did not restore: alias is {after_rollback}, expected {oldest}"

        return {
            "verified": verified,
            "reason": reason,
            "oldest_version": oldest,
            "latest_version": latest,
            "steps": steps,
        }
    finally:
        # Never leave the registry pointing wherever the verification chose.
        if original and current_champion() != original:
            client.set_registered_model_alias(
                settings.model.name, settings.model.serving_alias, original
            )
            log.info("rollback_verification_restored_alias", version=original)


__all__ = [
    "PromotionRecord",
    "current_champion",
    "history",
    "promote",
    "record",
    "rollback",
    "verify_rollback_path",
]
