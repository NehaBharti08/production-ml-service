"""Run the monitoring loop and emit its findings.

Ties together the reference window, the model, the drift detectors and the
replay harness. This is what the scheduled job calls, and what the Phase 7
retraining trigger reads.

Drift summary metrics are exported to Prometheus so drift appears on the same
dashboards as latency. One pane of glass matters: an operator who has to open a
separate tool to find out whether the model is still seeing the population it
was trained on will not do it during an incident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlservice.config import PROJECT_ROOT, get_settings
from mlservice.logging_ import get_logger
from mlservice.monitoring import drift as drift_mod
from mlservice.monitoring import replay as replay_mod

log = get_logger(__name__)


def load_reference() -> pd.DataFrame:
    """The frozen training window every comparison is made against.

    Written once in Phase 1 and then left alone. A reference that silently
    tracks recent data cannot detect drift, because it drifts along with it —
    the single most common way a drift detector is rendered useless.
    """
    settings = get_settings()
    path = settings.paths.data_reference / "reference_window.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run mlservice data audit` to write the "
            "frozen reference window."
        )
    frame = pd.read_parquet(path)

    summary = PROJECT_ROOT / "reports" / "training_summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        frame.attrs["feature_schema_hash"] = payload.get("feature_schema_hash", "unknown")
    return frame


def load_model() -> Any:
    from mlservice.api.model_loader import ModelStore

    return ModelStore().load()


def baseline_pr_auc() -> float:
    """The champion's test-set PR-AUC, which label drift is measured against."""
    summary = PROJECT_ROOT / "reports" / "training_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    champion = next(c for c in payload["candidates"] if c["name"] == payload["champion"])
    return float(champion["pr_auc"]["point"])


def _score(model: Any, frame: pd.DataFrame) -> np.ndarray:
    from mlservice.data import features as feature_mod

    x, _ = feature_mod.split_xy(frame)
    return np.asarray(model.predict_proba_batch(x.to_dict("records")))


def run_replay(
    mode: str = "real",
    inducer: str = "age",
    window_rows: int | None = None,
    export_metrics: bool = True,
) -> replay_mod.ReplayResult:
    """Replay windows through the detectors and report what fired.

    ``mode="real"`` replays the held-out test split untouched — any drift found
    is genuine. ``mode="induced"`` deliberately manipulates later windows, and
    every artefact says so.
    """
    settings = get_settings()
    reference = load_reference()
    model = load_model()
    baseline = baseline_pr_auc()

    test_path = settings.paths.data_processed / "test.parquet"
    if not test_path.is_file():
        raise FileNotFoundError(f"{test_path} not found. Run `mlservice data audit` first.")
    current_data = pd.read_parquet(test_path)

    reference_scores = _score(model, reference.sample(min(len(reference), 5000), random_state=42))

    if mode == "real":
        windows = replay_mod.chronological_windows(current_data, window_rows)
        origin: replay_mod.DriftOrigin = "real"
        used_inducer = None
    else:
        windows = replay_mod.induced_windows(current_data, inducer, window_rows)
        origin = "induced"
        used_inducer = inducer

    result = replay_mod.ReplayResult(drift_origin=origin, inducer=used_inducer)
    reports: list[drift_mod.DriftReport] = []

    for window in windows:
        frame = window.frame
        assert frame is not None
        frame.attrs["feature_schema_hash"] = reference.attrs.get("feature_schema_hash", "unknown")

        scores = _score(model, frame)
        matured = replay_mod.simulate_maturation(frame, scores)

        report = drift_mod.analyse_window(
            reference=reference,
            current=frame,
            reference_scores=reference_scores,
            current_scores=scores,
            decision_threshold=model.decision_threshold,
            matured=matured,
            baseline_pr_auc=baseline,
        )
        reports.append(report)

        state = drift_mod.alert_state(reports)
        if state["confirmed"] and result.first_detection_window is None:
            result.first_detection_window = window.index

        entry = window.describe()
        entry["drift"] = report.to_dict()
        entry["alert_state"] = state
        result.windows.append(entry)

    result.alert = drift_mod.alert_state(reports)

    if export_metrics:
        export_to_prometheus(reports[-1] if reports else None)

    log.info(
        "replay_complete",
        drift_origin=origin,
        inducer=used_inducer,
        n_windows=len(result.windows),
        confirmed=result.alert["confirmed"],
        first_detection_window=result.first_detection_window,
    )
    return result


def export_to_prometheus(report: drift_mod.DriftReport | None) -> Path | None:
    """Write drift metrics for the Prometheus textfile collector.

    A file rather than a push gateway: the monitoring job is a periodic batch,
    and a push gateway keeps serving the last value forever after the job dies —
    turning a dead job into a permanently healthy-looking dashboard. A stale
    file at least has a visible mtime.
    """
    if report is None:
        return None

    settings = get_settings()
    target = settings.paths.data_monitoring / "drift_metrics.prom"
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HELP drift_score Population Stability Index per feature against the frozen reference.",
        "# TYPE drift_score gauge",
    ]
    for feature in report.features:
        lines.append(f'drift_score{{feature="{feature.feature}",method="psi"}} {feature.psi:.6f}')

    lines += [
        "# HELP drift_threshold Calibrated per-feature PSI threshold.",
        "# TYPE drift_threshold gauge",
    ]
    for feature in report.features:
        lines.append(f'drift_threshold{{feature="{feature.feature}"}} {feature.threshold:.6f}')

    lines += [
        "# HELP drift_features_breaching Features above their calibrated threshold.",
        "# TYPE drift_features_breaching gauge",
        f"drift_features_breaching {len(report.breaching)}",
    ]

    if report.labels.get("sufficient_labels"):
        lines += [
            "# HELP rolling_pr_auc PR-AUC on matured labels in the current window.",
            "# TYPE rolling_pr_auc gauge",
            f"rolling_pr_auc {report.labels['observed_pr_auc']:.6f}",
        ]

    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log.info("drift_metrics_exported", path=str(target), n_features=len(report.features))
    return target


__all__ = [
    "baseline_pr_auc",
    "export_to_prometheus",
    "load_model",
    "load_reference",
    "run_replay",
]
