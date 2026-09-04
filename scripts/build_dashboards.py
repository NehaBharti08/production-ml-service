"""Generate the Grafana dashboards from configs/thresholds.yaml.

Dashboards are code, and this generates that code rather than hand-maintaining
2,000 lines of JSON. Two reasons that matters here:

*   **Thresholds cannot drift from the SLO.** Every red/amber band is read from
    ``configs/thresholds.yaml``, so when Phase 4's measured latency changed, the
    dashboards changed with it. A hand-written dashboard silently keeps showing
    last quarter's threshold.
*   **The reasoning survives.** Panel choices get a comment here; they get
    nothing in raw JSON.

Run: ``uv run python scripts/build_dashboards.py``

Design constraints applied:

*   **Status colour never carries meaning alone.** Every panel sets a unit and
    shows its value as text, so a red tile also says *why* it is red. Colour is
    the fast path, not the only path.
*   **Never a dual-axis panel.** Two measures of different scale get two panels.
*   **The golden-signals dashboard must be readable in ten seconds** — four stat
    tiles across the top answering "is it up, is it fast, is it failing, is it
    near capacity", with the detail below for anyone who wants it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "deploy" / "grafana" / "dashboards"

# Validated status palette. Deliberately not Grafana's defaults: these four
# steps are checked for CVD separation and contrast against both surfaces.
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
TEXT = "text"

DATASOURCE = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}


def thresholds() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "thresholds.yaml").read_text(encoding="utf-8"))


def _steps(*pairs: tuple[float | None, str]) -> dict[str, Any]:
    """Threshold steps. The first step must have a null value (the base band)."""
    return {
        "mode": "absolute",
        "steps": [{"color": color, "value": value} for value, color in pairs],
    }


def stat(
    title: str,
    expr: str,
    unit: str,
    steps: dict[str, Any],
    grid: dict[str, int],
    panel_id: int,
    description: str,
    decimals: int = 1,
) -> dict[str, Any]:
    """A headline number with a traffic-light band.

    ``textMode: value_and_name`` is deliberate: the tile shows the number *and*
    what it measures, so the colour is never the only signal. A bare coloured
    square communicates nothing to someone who has not memorised the layout.
    """
    return {
        "type": "stat",
        "id": panel_id,
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "gridPos": grid,
        "targets": [{"expr": expr, "refId": "A", "datasource": DATASOURCE}],
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "auto",
            "textMode": "value_and_name",
            "colorMode": "value",  # colour the number, not the whole tile
            "graphMode": "area",  # sparkline gives trend without a second panel
            "justifyMode": "auto",
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": decimals,
                "thresholds": steps,
                "mappings": [],
            },
            "overrides": [],
        },
    }


def timeseries(
    title: str,
    targets: list[dict[str, str]],
    unit: str,
    grid: dict[str, int],
    panel_id: int,
    description: str,
    thresholds_config: dict[str, Any] | None = None,
    threshold_style: str | None = None,
    decimals: int | None = None,
    legend_placement: str = "bottom",
) -> dict[str, Any]:
    """A trend panel.

    A legend is always present when there is more than one series, so identity
    is never carried by colour alone. Lines are 2px and the fill is light —
    the data should be the most prominent thing on the panel, not its styling.
    """
    field_config: dict[str, Any] = {
        "defaults": {
            "unit": unit,
            "custom": {
                "lineWidth": 2,
                "fillOpacity": 8,
                "showPoints": "never",
                "spanNulls": False,
            },
            "mappings": [],
        },
        "overrides": [],
    }
    if decimals is not None:
        field_config["defaults"]["decimals"] = decimals
    if thresholds_config:
        field_config["defaults"]["thresholds"] = thresholds_config
        if threshold_style:
            # A dashed line at the SLO, so "how close are we" is legible without
            # reading the axis.
            field_config["defaults"]["custom"]["thresholdsStyle"] = {"mode": threshold_style}

    return {
        "type": "timeseries",
        "id": panel_id,
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "gridPos": grid,
        "targets": [{**t, "datasource": DATASOURCE} for t in targets],
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": legend_placement,
                "showLegend": len(targets) > 1,
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "fieldConfig": field_config,
    }


def row(title: str, panel_id: int, y: int) -> dict[str, Any]:
    return {
        "type": "row",
        "id": panel_id,
        "title": title,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "collapsed": False,
        "panels": [],
    }


def dashboard(
    uid: str, title: str, description: str, panels: list[dict[str, Any]], tags: list[str]
):
    return {
        "uid": uid,
        "title": title,
        "description": description,
        "tags": tags,
        "timezone": "utc",  # never local: an incident spans time zones
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "refresh": "30s",  # matches the 15s scrape without hammering
        "time": {"from": "now-6h", "to": "now"},
        "templating": {
            "list": [
                {
                    "name": "DS_PROMETHEUS",
                    "type": "datasource",
                    "query": "prometheus",
                    "current": {"text": "Prometheus", "value": "Prometheus"},
                    "hide": 0,
                }
            ]
        },
        "annotations": {"list": []},
        "panels": panels,
    }


# --------------------------------------------------------------------------- #
# 1. Golden signals — the ten-second dashboard
# --------------------------------------------------------------------------- #


def golden_signals(t: dict[str, Any]) -> dict[str, Any]:
    """Traffic, latency, errors, saturation. Four tiles, then the detail.

    The four tiles answer, in order: is anyone using it, is it fast, is it
    failing, is it near capacity. Someone who has never seen this service should
    be able to say "fine" or "not fine" without scrolling or reading a legend.
    """
    slo = t["slo"]
    lat = slo["latency"]
    err = slo["error_rate"]

    slo_p99_s = lat["slo_p99_ms"] / 1000
    ticket_p99_s = lat["ticket_p99_ms"] / 1000
    knee = lat["measured_knee_rps"]

    panels: list[dict[str, Any]] = []

    # --- the four tiles -----------------------------------------------------
    panels.append(
        stat(
            "Traffic",
            'sum(rate(http_requests_total{endpoint!~"/health.*|/metrics"}[5m]))',
            "reqps",
            # Amber below 0.01 rps: silence is a signal on a service that should
            # be receiving traffic, and nothing else alerts in that state.
            _steps((None, WARNING), (0.01, GOOD)),
            {"h": 5, "w": 6, "x": 0, "y": 0},
            1,
            "Prediction requests per second. Health probes and metric scrapes "
            "are excluded — they would mask a drop in real traffic.",
            decimals=2,
        )
    )
    panels.append(
        stat(
            "Latency p99",
            "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
            "s",
            _steps((None, GOOD), (ticket_p99_s, WARNING), (slo_p99_s, CRITICAL)),
            {"h": 5, "w": 6, "x": 6, "y": 0},
            2,
            f"Amber at the {lat['ticket_p99_ms']}ms ticket threshold, red at the "
            f"{lat['slo_p99_ms']}ms SLO. Both derived as multiples of the measured "
            "p99 by a rule fixed before measurement.",
            decimals=3,
        )
    )
    panels.append(
        stat(
            "Error rate (5xx)",
            # `or vector(0)` is load-bearing, not defensive padding.
            #
            # With no 5xx in the window the numerator matches no series at all,
            # so the whole expression returns nothing — and this panel reduces
            # with lastNotNull, which then keeps displaying the last value it
            # ever saw. Observed: a red 10.21% held on screen through a
            # completely healthy period, hours after the errors that produced
            # it had stopped. The graph beside it correctly showed zero.
            #
            # A golden-signals panel that shows red while the service is fine
            # is worse than no panel: it is the mechanism by which people learn
            # to ignore the dashboard. Forcing the series to exist at 0 makes
            # "no errors" render as 0 instead of as stale alarm.
            '(sum(rate(http_requests_total{status="5xx"}[5m])) or vector(0)) '
            "/ clamp_min(sum(rate(http_requests_total[5m])), 0.001)",
            "percentunit",
            _steps(
                (None, GOOD),
                (err["ticket_5xx_ratio"], WARNING),
                (err["page_5xx_ratio"], CRITICAL),
            ),
            {"h": 5, "w": 6, "x": 12, "y": 0},
            3,
            "5xx only. A 4xx is a caller sending bad input, not the service "
            "failing — counting it here would make the SLO unactionable.",
            decimals=2,
        )
    )
    panels.append(
        stat(
            "Saturation",
            f"sum(rate(http_requests_total[5m])) / {knee}",
            "percentunit",
            # Amber at 80% of the measured knee, red at the knee itself, where
            # throughput inverts.
            _steps((None, GOOD), (0.8, WARNING), (1.0, CRITICAL)),
            {"h": 5, "w": 6, "x": 18, "y": 0},
            4,
            f"Request rate as a fraction of the measured saturation knee "
            f"(~{knee} req/s). Past 100% throughput INVERTS: the service gets "
            "slower and less productive at once.",
            decimals=2,
        )
    )

    # --- detail -------------------------------------------------------------
    panels.append(row("Detail", 10, 5))

    panels.append(
        timeseries(
            "Request rate by endpoint",
            [
                {
                    "expr": "sum by (endpoint) (rate(http_requests_total[5m]))",
                    "legendFormat": "{{endpoint}}",
                    "refId": "A",
                }
            ],
            "reqps",
            {"h": 8, "w": 12, "x": 0, "y": 6},
            11,
            "Per-endpoint, so a drop in predictions is not hidden by probe traffic.",
        )
    )

    panels.append(
        timeseries(
            "Latency percentiles",
            [
                {
                    "expr": "histogram_quantile(0.50, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
                    "legendFormat": "p50",
                    "refId": "A",
                },
                {
                    "expr": "histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
                    "legendFormat": "p95",
                    "refId": "B",
                },
                {
                    "expr": "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
                    "legendFormat": "p99",
                    "refId": "C",
                },
            ],
            "s",
            {"h": 8, "w": 12, "x": 12, "y": 6},
            12,
            "p50/p95/p99 on ONE axis. The dashed line is the SLO — distance to "
            "it is the thing worth seeing at a glance.",
            thresholds_config=_steps((None, TEXT), (slo_p99_s, CRITICAL)),
            threshold_style="dashed",
            decimals=3,
        )
    )

    panels.append(
        timeseries(
            "Errors by status class",
            [
                {
                    "expr": "sum by (status) (rate(http_requests_total[5m]))",
                    "legendFormat": "{{status}}",
                    "refId": "A",
                }
            ],
            "reqps",
            {"h": 8, "w": 12, "x": 0, "y": 14},
            13,
            "4xx and 5xx separated: they mean different things and need different responses.",
        )
    )

    panels.append(
        timeseries(
            "Model inference time (isolated)",
            [
                {
                    "expr": "histogram_quantile(0.99, sum by (le) (rate(model_inference_duration_seconds_bucket[5m])))",
                    "legendFormat": "p99 inference",
                    "refId": "A",
                }
            ],
            "s",
            {"h": 8, "w": 12, "x": 12, "y": 14},
            14,
            "The model alone, excluding validation and logging. If total latency "
            "rises while this stays flat, the cause is the API layer — measured "
            "at ~1ms for the model against ~64ms for the feature transform.",
            decimals=4,
        )
    )

    return dashboard(
        "mlservice-golden",
        "ML Service — Golden Signals",
        "Traffic, latency, errors, saturation. Thresholds come from "
        "configs/thresholds.yaml; see docs/MONITORING.md for how each was derived.",
        panels,
        ["mlservice", "golden-signals"],
    )


# --------------------------------------------------------------------------- #
# 2. Model health
# --------------------------------------------------------------------------- #


def model_health(_t: dict[str, Any]) -> dict[str, Any]:
    # Signature matches the other builders so main() can call them uniformly;
    # this one needs no thresholds because its panels are informational.
    panels: list[dict[str, Any]] = []

    panels.append(
        stat(
            "Model loaded",
            "model_load_status",
            "short",
            _steps((None, CRITICAL), (1, GOOD)),
            {"h": 5, "w": 6, "x": 0, "y": 0},
            1,
            "The same gauge readiness reports on. 0 means every prediction is returning 503.",
            decimals=0,
        )
    )
    panels.append(
        stat(
            "Predictions / sec",
            "sum(rate(model_predictions_total[5m]))",
            "reqps",
            _steps((None, TEXT)),
            {"h": 5, "w": 6, "x": 6, "y": 0},
            2,
            "Throughput of scored records, including batch items.",
            decimals=2,
        )
    )
    panels.append(
        stat(
            "Flagged rate",
            'sum(rate(model_predictions_total{predicted_label="1"}[1h])) '
            "/ clamp_min(sum(rate(model_predictions_total[1h])), 0.001)",
            "percentunit",
            # This drives downstream human workload, so a large move matters
            # even when the score distribution looks stable.
            _steps((None, TEXT)),
            {"h": 5, "w": 6, "x": 12, "y": 0},
            3,
            "Share of patients above the decision threshold. This is what drives "
            "downstream workload — measured at ~32% on the test split.",
            decimals=3,
        )
    )
    panels.append(
        stat(
            "Outcomes recorded",
            "sum(increase(outcomes_recorded_total[24h]))",
            "short",
            # Amber at zero: no labels means every performance metric is frozen
            # and will look healthy precisely because nothing updates it.
            _steps((None, WARNING), (1, GOOD)),
            {"h": 5, "w": 6, "x": 18, "y": 0},
            4,
            "Matured labels in 24h. Zero means delayed-label monitoring is blind "
            "— and blind looks identical to healthy on every other panel.",
            decimals=0,
        )
    )

    panels.append(row("Prediction distribution", 10, 5))

    panels.append(
        timeseries(
            "Mean predicted probability",
            [
                {
                    "expr": "sum(rate(model_predicted_proba_sum[10m])) "
                    "/ clamp_min(sum(rate(model_predicted_proba_count[10m])), 0.001)",
                    "legendFormat": "mean score",
                    "refId": "A",
                }
            ],
            "short",
            {"h": 8, "w": 12, "x": 0, "y": 6},
            11,
            "Moves before accuracy does, and long before labels mature at 30 "
            "days. The earliest available signal that the input population "
            "changed.",
            decimals=4,
        )
    )

    panels.append(
        timeseries(
            "Predictions by label",
            [
                {
                    "expr": "sum by (predicted_label) (rate(model_predictions_total[5m]))",
                    "legendFormat": "label {{predicted_label}}",
                    "refId": "A",
                }
            ],
            "reqps",
            {"h": 8, "w": 12, "x": 12, "y": 6},
            12,
            "A shift in the flagged/not-flagged mix is visible here before any "
            "aggregate metric moves.",
        )
    )

    panels.append(
        timeseries(
            "Validation failures by field",
            [
                {
                    "expr": "sum by (field) (rate(validation_errors_total[10m]))",
                    "legendFormat": "{{field}}",
                    "refId": "A",
                }
            ],
            "reqps",
            {"h": 8, "w": 12, "x": 0, "y": 14},
            13,
            "Per FIELD deliberately: a spike on one field is an upstream schema "
            "change; the same volume spread across many is a caller sending junk. "
            "Different problems, different responses.",
        )
    )

    panels.append(
        timeseries(
            "Prediction log writes",
            [
                {
                    "expr": "sum by (result) (rate(prediction_log_writes_total[10m]))",
                    "legendFormat": "{{result}}",
                    "refId": "A",
                }
            ],
            "reqps",
            {"h": 8, "w": 12, "x": 12, "y": 14},
            14,
            "A failed write does not fail the request — by design. But it starves "
            "drift detection and retraining, so the gap must be visible.",
        )
    )

    return dashboard(
        "mlservice-model",
        "ML Service — Model Health",
        "What the model is doing: load state, score distribution, flagged rate, "
        "and whether the label pipeline is alive.",
        panels,
        ["mlservice", "model"],
    )


# --------------------------------------------------------------------------- #
# 3. Drift — populated by the Phase 6 monitoring job
# --------------------------------------------------------------------------- #


def drift(t: dict[str, Any]) -> dict[str, Any]:
    """Drift on the same pane of glass as latency — one place to look.

    The metrics this reads are exported by the Phase 6 monitoring job. The
    dashboard is built now so the metric contract is fixed before the job is
    written, rather than the dashboard being shaped around whatever the job
    happened to emit.
    """
    alert = t["drift"]["alert"]["data_drift"]
    panels: list[dict[str, Any]] = []

    panels.append(
        stat(
            "Features breaching",
            "drift_features_breaching",
            "short",
            # Red at the confirmed-alert count from thresholds.yaml, amber one
            # below it — the point where it is worth looking, not yet acting.
            _steps(
                (None, GOOD),
                (max(alert["min_features_breaching"] - 1, 1), WARNING),
                (alert["min_features_breaching"], CRITICAL),
            ),
            {"h": 5, "w": 8, "x": 0, "y": 0},
            1,
            f"Alert fires at {alert['min_features_breaching']} features breaching "
            f"for {alert['consecutive_windows']} consecutive windows. The "
            "two-window confirmation is what stops single-window noise paging.",
            decimals=0,
        )
    )
    panels.append(
        stat(
            "Rolling PR-AUC",
            "rolling_pr_auc",
            "short",
            # Bands are set by the Phase 6 job against the test-set baseline;
            # a fixed constant here would be exactly the arbitrary threshold
            # this project avoids.
            _steps((None, TEXT)),
            {"h": 5, "w": 8, "x": 8, "y": 0},
            2,
            "Measured on matured labels only. Compared against the test-set "
            "baseline in bootstrap standard errors, not raw points — a 0.02 drop "
            "means different things at n=500 and n=5000.",
            decimals=4,
        )
    )
    panels.append(
        stat(
            "Hours since last label",
            "(time() - drift_last_matured_label_timestamp) / 3600",
            "h",
            _steps((None, GOOD), (24, WARNING), (120, CRITICAL)),
            {"h": 5, "w": 8, "x": 16, "y": 0},
            3,
            "The watchdog. Silent label-pipeline failure is how monitoring "
            "usually dies: nothing alerts, because nothing is being measured.",
            decimals=1,
        )
    )

    panels.append(row("Per-feature drift", 10, 5))

    panels.append(
        timeseries(
            "PSI by feature",
            [
                {
                    "expr": 'drift_score{method="psi"}',
                    "legendFormat": "{{feature}}",
                    "refId": "A",
                }
            ],
            "short",
            {"h": 10, "w": 24, "x": 0, "y": 6},
            11,
            "Population Stability Index per feature, against the frozen "
            "reference window. Each feature's alert threshold is its OWN "
            "empirical-null 99th percentile — what that feature's natural churn "
            "looks like between stable training windows — not a shared constant.",
            decimals=4,
            legend_placement="right",
        )
    )

    return dashboard(
        "mlservice-drift",
        "ML Service — Drift",
        "Data, prediction and delayed-label drift. Thresholds are calibrated "
        "per feature against an empirical null; see docs/MONITORING.md.",
        panels,
        ["mlservice", "drift"],
    )


def main() -> None:
    t = thresholds()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    for name, build in (
        ("golden-signals", golden_signals),
        ("model-health", model_health),
        ("drift", drift),
    ):
        target = DASHBOARD_DIR / f"{name}.json"
        # Newline is pinned explicitly. write_text() otherwise translates to
        # the platform line ending, so Windows emitted CRLF while CI (Linux)
        # emits LF - the CI "dashboards are in sync" check would then fail on
        # line endings alone, every time, for a file whose content was fine.
        payload = json.dumps(build(t), indent=2) + chr(10)
        with target.open("w", encoding="utf-8", newline=chr(10)) as fh:
            fh.write(payload)
        panels = [p for p in build(t)["panels"] if p["type"] != "row"]
        print(f"{target.relative_to(ROOT)}  ({len(panels)} panels)")


if __name__ == "__main__":
    main()
