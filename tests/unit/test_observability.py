"""Alert rules and dashboards.

These assert the properties that make monitoring *usable*, which is a different
question from whether it is syntactically valid. promtool checks that the PromQL
parses; nothing but a test checks that every alert has somewhere to send the
person it wakes up.

The failure these guard against is quiet and expensive: monitoring that exists,
looks configured, and does nothing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from mlservice.config import PROJECT_ROOT

pytestmark = pytest.mark.unit

RULES = PROJECT_ROOT / "deploy" / "prometheus" / "rules" / "alerts.yml"
DASHBOARDS = PROJECT_ROOT / "deploy" / "grafana" / "dashboards"


@pytest.fixture(scope="module")
def rules() -> list[dict[str, Any]]:
    groups = yaml.safe_load(RULES.read_text(encoding="utf-8"))["groups"]
    return [rule for group in groups for rule in group["rules"]]


@pytest.fixture(scope="module")
def thresholds() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "thresholds.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dashboards() -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(DASHBOARDS.glob("*.json"))
    }


class TestAlertsAreActionable:
    def test_every_alert_links_to_a_runbook(self, rules: list[dict[str, Any]]) -> None:
        """An alert with no documented response is a notification.

        Notifications train people to ignore pages, which is worse than not
        alerting at all.
        """
        missing = [r["alert"] for r in rules if "runbook_url" not in r.get("annotations", {})]
        assert not missing, f"alerts with no runbook link: {missing}"

    def test_every_alert_has_a_severity(self, rules: list[dict[str, Any]]) -> None:
        missing = [r["alert"] for r in rules if "severity" not in r.get("labels", {})]
        assert not missing, f"alerts with no severity: {missing}"

    def test_severity_has_exactly_two_levels(self, rules: list[dict[str, Any]]) -> None:
        """page or ticket. A third level always becomes one nobody reads."""
        levels = {r["labels"]["severity"] for r in rules}
        assert levels <= {"page", "ticket"}, f"unexpected severity levels: {levels}"

    def test_pages_are_rare(self, rules: list[dict[str, Any]]) -> None:
        """Most alerts should not wake anyone.

        A pager that fires constantly gets silenced, and then the one that
        mattered is silenced too.
        """
        pages = [r for r in rules if r["labels"]["severity"] == "page"]
        assert len(pages) <= len(rules) / 2, (
            f"{len(pages)} of {len(rules)} alerts page — too many to stay credible"
        )

    def test_every_alert_has_a_for_clause(self, rules: list[dict[str, Any]]) -> None:
        """Without `for`, a single scrape fires the alert.

        Momentary spikes are normal; sustained ones are incidents. The `for`
        duration is what distinguishes them.
        """
        missing = [r["alert"] for r in rules if "for" not in r]
        assert not missing, f"alerts that fire on a single scrape: {missing}"

    def test_every_alert_explains_itself(self, rules: list[dict[str, Any]]) -> None:
        """summary and description, so the page is readable at 3am."""
        for rule in rules:
            annotations = rule.get("annotations", {})
            assert annotations.get("summary"), f"{rule['alert']} has no summary"
            assert annotations.get("description"), f"{rule['alert']} has no description"


class TestAlertThresholdsMatchTheSLO:
    """The alerts must fire at the documented thresholds, not near them."""

    def _expr(self, rules: list[dict[str, Any]], name: str) -> str:
        return next(r["expr"] for r in rules if r["alert"] == name)

    def test_latency_page_uses_the_slo(
        self, rules: list[dict[str, Any]], thresholds: dict[str, Any]
    ) -> None:
        slo_seconds = thresholds["slo"]["latency"]["slo_p99_ms"] / 1000
        assert f"> {slo_seconds:.3f}" in self._expr(rules, "LatencyP99AboveSLO")

    def test_latency_ticket_uses_the_ticket_threshold(
        self, rules: list[dict[str, Any]], thresholds: dict[str, Any]
    ) -> None:
        ticket_seconds = thresholds["slo"]["latency"]["ticket_p99_ms"] / 1000
        assert f"> {ticket_seconds:.3f}" in self._expr(rules, "LatencyP99Degrading")

    def test_burn_rate_factors_match_the_standard(
        self, rules: list[dict[str, Any]], thresholds: dict[str, Any]
    ) -> None:
        """14.4x and 6x are the Google SRE Workbook values, not invented."""
        burn = thresholds["slo"]["burn_rate"]
        assert str(burn["page"]["factor"]) in self._expr(rules, "ErrorBudgetBurningFast")
        assert str(int(burn["ticket"]["factor"])) in self._expr(rules, "ErrorBudgetBurningSlowly")

    def test_saturation_alert_sits_below_the_measured_knee(
        self, rules: list[dict[str, Any]], thresholds: dict[str, Any]
    ) -> None:
        """Must warn BEFORE the knee, not at it.

        Past the knee throughput inverts, so an alert that fires there arrives
        after the damage rather than before it.
        """
        knee = thresholds["slo"]["latency"]["measured_knee_rps"]
        expr = self._expr(rules, "ApproachingSaturation")
        fires_at = float(expr.rsplit(">", 1)[1].strip())
        assert fires_at < knee, f"alert fires at {fires_at} rps, at or past the {knee} knee"


class TestAlertsCoverTheFailureModes:
    """The alerts that must exist, named by the failure they catch."""

    def _names(self, rules: list[dict[str, Any]]) -> set[str]:
        return {r["alert"] for r in rules}

    def test_covers_the_four_golden_signals(self, rules: list[dict[str, Any]]) -> None:
        names = self._names(rules)
        assert "LatencyP99AboveSLO" in names  # latency
        assert "ErrorBudgetBurningFast" in names  # errors
        assert "ApproachingSaturation" in names  # saturation
        assert "NoTrafficReceived" in names  # traffic

    def test_covers_silent_monitoring_failure(self, rules: list[dict[str, Any]]) -> None:
        """The way monitoring usually dies: nothing alerts, because nothing is
        being measured, and the dashboards look healthy precisely because they
        have stopped updating."""
        names = self._names(rules)
        assert "NoOutcomesRecorded" in names, "no watchdog for a dead label pipeline"
        assert "PredictionLogWriteFailures" in names, "no watchdog for a starved monitoring log"
        assert "ServiceDown" in names, "nothing fires when the app cannot expose metrics"

    def test_covers_serving_the_fallback(self, rules: list[dict[str, Any]]) -> None:
        """Predictions are fine on the fallback; promotion and rollback are not."""
        assert "ServingFromLocalFallback" in self._names(rules)


class TestDashboards:
    def test_three_dashboards_exist(self, dashboards: dict[str, dict[str, Any]]) -> None:
        assert set(dashboards) == {"golden-signals", "model-health", "drift"}

    def test_every_panel_declares_a_unit(self, dashboards: dict[str, dict[str, Any]]) -> None:
        """A number without a unit is a number nobody can act on.

        "0.7" is meaningless; "0.7 s" is a decision.
        """
        missing = [
            f"{name}/{p['title']}"
            for name, d in dashboards.items()
            for p in d["panels"]
            if p["type"] != "row" and not p["fieldConfig"]["defaults"].get("unit")
        ]
        assert not missing, f"panels with no unit: {missing}"

    def test_stat_panels_show_the_value_as_text(
        self, dashboards: dict[str, dict[str, Any]]
    ) -> None:
        """Status colour must never carry meaning alone.

        A red tile that does not say what is red, or by how much, communicates
        nothing to someone who has not memorised the layout — and is unreadable
        to a colourblind viewer.
        """
        colour_only = [
            f"{name}/{p['title']}"
            for name, d in dashboards.items()
            for p in d["panels"]
            if p["type"] == "stat" and p["options"].get("textMode") == "none"
        ]
        assert not colour_only, f"panels conveying state by colour alone: {colour_only}"

    def test_every_panel_explains_itself(self, dashboards: dict[str, dict[str, Any]]) -> None:
        """The description is where the *reasoning* lives.

        A stranger reading the dashboard should be able to learn why a threshold
        sits where it does without opening the repo.
        """
        missing = [
            f"{name}/{p['title']}"
            for name, d in dashboards.items()
            for p in d["panels"]
            if p["type"] != "row" and not p.get("description")
        ]
        assert not missing, f"panels with no description: {missing}"

    def test_golden_signals_leads_with_four_tiles(
        self, dashboards: dict[str, dict[str, Any]]
    ) -> None:
        """Readable in ten seconds: is it up, fast, failing, near capacity.

        All four must sit on the top row, because anything below the fold is not
        part of a ten-second read.
        """
        panels = dashboards["golden-signals"]["panels"]
        top = [p for p in panels if p["type"] == "stat" and p["gridPos"]["y"] == 0]
        assert len(top) == 4, f"expected 4 tiles on the top row, found {len(top)}"
        assert {p["title"] for p in top} == {
            "Traffic",
            "Latency p99",
            "Error rate (5xx)",
            "Saturation",
        }

    def test_timezone_is_utc(self, dashboards: dict[str, dict[str, Any]]) -> None:
        """Never local time. An incident spans time zones and log correlation
        against UTC timestamps has to be possible without arithmetic."""
        for name, d in dashboards.items():
            assert d["timezone"] == "utc", f"{name} is not in UTC"

    def test_uids_are_stable(self, dashboards: dict[str, dict[str, Any]]) -> None:
        """A changed uid orphans every existing link and bookmark."""
        expected = {
            "golden-signals": "mlservice-golden",
            "model-health": "mlservice-model",
            "drift": "mlservice-drift",
        }
        for name, d in dashboards.items():
            assert d["uid"] == expected[name]

    def test_no_dual_axis_panels(self, dashboards: dict[str, dict[str, Any]]) -> None:
        """Two y-scales on one panel is the most common charting mistake.

        It lets any two series be made to look correlated by choosing scales.
        Two measures of different magnitude get two panels.
        """
        for name, d in dashboards.items():
            for panel in d["panels"]:
                overrides = panel.get("fieldConfig", {}).get("overrides", [])
                for override in overrides:
                    props = [p.get("id") for p in override.get("properties", [])]
                    assert "custom.axisPlacement" not in props, (
                        f"{name}/{panel.get('title')} appears to use a second axis"
                    )
