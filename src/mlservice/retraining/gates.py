"""Promotion gates — the code that blocks a bad model from production.

Six independent gates. A challenger is promoted **only if all six pass**;
otherwise it stays in Staging, an alert fires, and the failing gate is named in
the log so the reason is never ambiguous.

Each gate is a pure function of two evaluations, which matters for two reasons:
they are trivially testable against deliberately-bad models, and the decision is
reproducible — the same inputs always yield the same verdict, with the same
stated reason.

The gate that carries this project's thesis is **calibration**. A challenger
that ranks better but calibrates worse is a *regression* for a health-adjacent
use case: a probability nobody can trust cannot support a decision, whatever its
ranking quality. Most projects never measure calibration at all; this one blocks
deployment on it, because a guideline gets waived the first time it is
inconvenient and a gate does not.

Thresholds come from ``configs/thresholds.yaml`` (``promotion.*``), so a change
to the policy is a config diff rather than a code change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mlservice.config import get_thresholds
from mlservice.logging_ import get_logger

log = get_logger(__name__)


@dataclass
class GateResult:
    """One gate's verdict, with the numbers that produced it.

    ``detail`` is not decoration. When a promotion is blocked at 3am, "the
    calibration gate failed" is not enough — the operator needs the challenger's
    ECE, the incumbent's, and the limit, without re-running anything.
    """

    name: str
    passed: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    promote: bool
    gates: list[GateResult] = field(default_factory=list)

    @property
    def failed(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed and g.blocking]

    @property
    def reason(self) -> str:
        if self.promote:
            return f"all {len(self.gates)} gates passed"
        names = ", ".join(g.name for g in self.failed)
        return f"blocked by: {names}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "n_gates": len(self.gates),
            "n_failed": len(self.failed),
            "failed_gates": [g.name for g in self.failed],
            "gates": [g.to_dict() for g in self.gates],
        }


# --------------------------------------------------------------------------- #
# Individual gates
# --------------------------------------------------------------------------- #


def performance_gate(challenger: dict[str, Any], incumbent: dict[str, Any]) -> GateResult:
    """PR-AUC non-inferiority, not strict improvement.

    Demanding an improvement on every scheduled refresh would block legitimate
    freshness updates — a model retrained on newer data that performs the *same*
    is usually the right thing to ship. Worse, a gate that blocks routine
    refreshes gets disabled, and a disabled gate protects nothing.

    The margin is one quarter of the expected bootstrap SE at this test size, so
    it tolerates sampling noise without tolerating real regression.
    """
    config = get_thresholds().model_dump()["promotion"]["performance"]
    margin = config["non_inferiority_margin"]

    challenger_value = challenger["pr_auc"]["point"]
    incumbent_value = incumbent["pr_auc"]["point"]
    floor = incumbent_value - margin
    passed = challenger_value >= floor

    return GateResult(
        name="performance",
        passed=passed,
        reason=(
            f"PR-AUC {challenger_value:.4f} >= {floor:.4f} "
            f"(incumbent {incumbent_value:.4f} - {margin} margin)"
            if passed
            else f"PR-AUC {challenger_value:.4f} below the non-inferiority floor {floor:.4f}"
        ),
        detail={
            "metric": config["metric"],
            "challenger": round(challenger_value, 6),
            "incumbent": round(incumbent_value, 6),
            "margin": margin,
            "floor": round(floor, 6),
            "difference": round(challenger_value - incumbent_value, 6),
        },
    )


def calibration_gate(challenger: dict[str, Any], incumbent: dict[str, Any]) -> GateResult:
    """THE gate this project exists to demonstrate.

    Two conditions, both binding:

    *   Brier no more than 2% worse than the incumbent — a tolerance for noise
        between retrains, not a licence to degrade.
    *   Absolute ECE at or below 0.05 — a challenger cannot be badly calibrated
        just because the incumbent was.

    A challenger with *better* PR-AUC and worse calibration is refused, and that
    is the intended behaviour. See ADR 0002.
    """
    config = get_thresholds().model_dump()["promotion"]["calibration"]
    max_ratio = config["max_brier_ratio_vs_incumbent"]
    max_ece = config["max_ece"]

    challenger_brier = challenger["brier"]["point"]
    incumbent_brier = incumbent["brier"]["point"]
    challenger_ece = challenger["calibration"]["ece"]

    # Lower Brier is better, so the ratio is challenger/incumbent.
    ratio = challenger_brier / incumbent_brier if incumbent_brier else float("inf")
    brier_ok = ratio <= max_ratio
    ece_ok = challenger_ece <= max_ece
    passed = brier_ok and ece_ok

    if passed:
        reason = f"Brier ratio {ratio:.4f} <= {max_ratio} and ECE {challenger_ece:.4f} <= {max_ece}"
    elif not brier_ok and not ece_ok:
        reason = f"Brier ratio {ratio:.4f} > {max_ratio} AND ECE {challenger_ece:.4f} > {max_ece}"
    elif not brier_ok:
        reason = (
            f"Brier ratio {ratio:.4f} > {max_ratio} — calibration degraded versus the incumbent"
        )
    else:
        reason = f"ECE {challenger_ece:.4f} > {max_ece} — probabilities are not trustworthy"

    return GateResult(
        name="calibration",
        passed=passed,
        reason=reason,
        detail={
            "challenger_brier": round(challenger_brier, 6),
            "incumbent_brier": round(incumbent_brier, 6),
            "brier_ratio": round(ratio, 6),
            "max_brier_ratio": max_ratio,
            "challenger_ece": round(challenger_ece, 6),
            "max_ece": max_ece,
            "brier_ok": brier_ok,
            "ece_ok": ece_ok,
        },
    )


def subgroup_gate(challenger: dict[str, Any], incumbent: dict[str, Any]) -> GateResult:
    """Do not widen the worst subgroup gap.

    Measured against the **incumbent's existing disparity**, not an absolute
    fairness constant. No absolute number would be defensible on this data, and
    inventing one implies a guarantee the evaluation cannot support. What *is*
    defensible is "do not make it worse", which is what this encodes.

    Phase 2 measured the incumbent's worst analysable gap at -0.232
    (age=[40-50)). A challenger may not widen that by more than 20% relative.
    """
    config = get_thresholds().model_dump()["promotion"]["subgroup"]
    max_widening = config["max_relative_gap_widening"]

    challenger_gap = challenger["subgroups"]["worst_recall_gap"]
    incumbent_gap = incumbent["subgroups"]["worst_recall_gap"]

    # Gaps are negative (a shortfall below overall recall); more negative is
    # worse. Compare magnitudes so the arithmetic reads the same way round as
    # the intent.
    challenger_magnitude = abs(challenger_gap)
    incumbent_magnitude = abs(incumbent_gap)

    if incumbent_magnitude == 0:
        # No existing disparity to widen: fall back to an absolute check so a
        # challenger cannot introduce one from nothing.
        passed = challenger_magnitude <= max_widening
        widening = challenger_magnitude
    else:
        widening = (challenger_magnitude - incumbent_magnitude) / incumbent_magnitude
        passed = widening <= max_widening

    return GateResult(
        name="subgroup",
        passed=passed,
        reason=(
            f"worst gap {challenger_gap:+.4f} vs incumbent {incumbent_gap:+.4f} "
            f"({widening:+.1%} relative, limit {max_widening:+.0%})"
            if passed
            else (
                f"worst subgroup gap widened {widening:+.1%}, above the "
                f"{max_widening:+.0%} limit — {challenger_gap:+.4f} vs "
                f"incumbent {incumbent_gap:+.4f}"
            )
        ),
        detail={
            "challenger_worst_gap": round(challenger_gap, 6),
            "incumbent_worst_gap": round(incumbent_gap, 6),
            "challenger_worst_group": challenger["subgroups"].get("worst_group"),
            "incumbent_worst_group": incumbent["subgroups"].get("worst_group"),
            "relative_widening": round(widening, 6),
            "max_relative_widening": max_widening,
            "min_subgroup_n": config["min_subgroup_n"],
        },
    )


def behavioral_gate(challenger: dict[str, Any]) -> GateResult:
    """Every invariance and directional test must pass — 100%, not most.

    This catches a corrupted feature pipeline that aggregate metrics sail past.
    A transform that silently drops ``number_inpatient`` leaves PR-AUC nearly
    intact, because the remaining features carry correlated signal, while making
    the model blind to its single strongest predictor. Only a directional test
    notices.

    The required pass rate is 1.0 deliberately. "Most behavioural tests pass" is
    not a state a model should be promoted in — each one encodes a property that
    is either true or broken.
    """
    config = get_thresholds().model_dump()["promotion"]["behavioral"]
    required = config["required_pass_rate"]

    results = challenger.get("behavioral", {})
    total = results.get("total", 0)
    passed_count = results.get("passed", 0)

    if total == 0:
        return GateResult(
            name="behavioral",
            passed=False,
            reason=(
                "behavioural suite did not run — a model cannot be promoted on absent evidence"
            ),
            detail={"total": 0, "passed": 0, "required_pass_rate": required},
        )

    rate = passed_count / total
    passed = rate >= required

    return GateResult(
        name="behavioral",
        passed=passed,
        reason=(
            f"{passed_count}/{total} behavioural tests passed"
            if passed
            else f"{total - passed_count} of {total} behavioural tests FAILED — "
            f"{', '.join(results.get('failures', [])[:3]) or 'see the suite output'}"
        ),
        detail={
            "total": total,
            "passed": passed_count,
            "pass_rate": round(rate, 4),
            "required_pass_rate": required,
            "failures": results.get("failures", []),
            "suites": config["suites"],
        },
    )


def operational_gate(challenger: dict[str, Any], incumbent: dict[str, Any]) -> GateResult:
    """The model must actually load, score, and match the serving contract.

    A feature-schema mismatch is blocking rather than advisory: the API sends
    raw records shaped by one contract, and a model expecting another would
    either raise on every request or — worse — silently score the wrong columns.
    """
    config = get_thresholds().model_dump()["promotion"]["operational"]

    loads = challenger.get("artifact_loads", False)
    challenger_hash = challenger.get("feature_schema_hash")
    incumbent_hash = incumbent.get("feature_schema_hash")
    hash_matches = challenger_hash == incumbent_hash

    failures = []
    if config["require_artifact_loads"] and not loads:
        failures.append("artifact does not load or failed its canary inference")
    if config["require_feature_schema_hash_match"] and not hash_matches:
        failures.append(
            f"feature schema hash {challenger_hash} != serving contract {incumbent_hash}"
        )

    return GateResult(
        name="operational",
        passed=not failures,
        reason="artifact loads and matches the serving contract"
        if not failures
        else "; ".join(failures),
        detail={
            "artifact_loads": loads,
            "challenger_schema_hash": challenger_hash,
            "incumbent_schema_hash": incumbent_hash,
            "schema_hash_matches": hash_matches,
        },
    )


def data_quality_gate(challenger: dict[str, Any]) -> GateResult:
    """The training data must pass the same checks the original data did.

    A feature whose missingness jumps by 10 percentage points is usually an
    upstream break, not a population change. Retraining on it bakes the break
    into the model — and the resulting model would pass every performance gate,
    because it learned the broken data faithfully.
    """
    config = get_thresholds().model_dump()["promotion"]["data_quality"]
    max_increase = config["max_missingness_increase_pct_points"]

    suite_passed = challenger.get("data_quality", {}).get("suite_passed", False)
    missingness = challenger.get("data_quality", {}).get("missingness_increase_pct_points", {})
    regressed = {f: v for f, v in missingness.items() if v > max_increase}

    failures = []
    if config["require_suite_pass"] and not suite_passed:
        failures.append("data-quality suite failed")
    if regressed:
        detail = ", ".join(f"{f} +{v:.1f}pp" for f, v in sorted(regressed.items())[:3])
        failures.append(f"missingness rose above {max_increase}pp: {detail}")

    return GateResult(
        name="data_quality",
        passed=not failures,
        reason="training data passed its quality checks" if not failures else "; ".join(failures),
        detail={
            "suite_passed": suite_passed,
            "max_missingness_increase_pct_points": max_increase,
            "features_regressed": regressed,
        },
    )


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


def evaluate_promotion(challenger: dict[str, Any], incumbent: dict[str, Any]) -> PromotionDecision:
    """Run every gate and decide.

    **All gates run even after one fails.** Short-circuiting would hide the
    other problems, and an operator fixing a blocked promotion needs the whole
    list rather than discovering the next failure after each fix.
    """
    gates = [
        performance_gate(challenger, incumbent),
        calibration_gate(challenger, incumbent),
        subgroup_gate(challenger, incumbent),
        behavioral_gate(challenger),
        operational_gate(challenger, incumbent),
        data_quality_gate(challenger),
    ]

    decision = PromotionDecision(promote=all(g.passed for g in gates), gates=gates)

    log.info(
        "promotion_evaluated",
        promote=decision.promote,
        reason=decision.reason,
        failed_gates=[g.name for g in decision.failed],
        **{f"gate_{g.name}": g.passed for g in gates},
    )
    for gate in decision.failed:
        log.warning("promotion_gate_failed", gate=gate.name, reason=gate.reason, **gate.detail)

    return decision


__all__ = [
    "GateResult",
    "PromotionDecision",
    "behavioral_gate",
    "calibration_gate",
    "data_quality_gate",
    "evaluate_promotion",
    "operational_gate",
    "performance_gate",
    "subgroup_gate",
]
