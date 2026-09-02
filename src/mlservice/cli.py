"""Typer CLI — every pipeline stage is runnable and CI-callable.

One entrypoint for the whole system means CI, the Makefile, the scheduled
monitoring job and a human at a terminal all invoke the same code path. Commands
are added by the phase that implements them; the stubs below name what is
coming so the surface is visible from the start.

    uv run mlservice config      # show resolved configuration
    uv run mlservice doctor      # check the environment is fit to run
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from mlservice import __version__
from mlservice.config import (
    PROJECT_ROOT,
    Settings,
    get_settings,
    get_thresholds,
    stray_env_vars,
)
from mlservice.logging_ import configure_logging, get_logger, request_context

app = typer.Typer(
    name="mlservice",
    help="Readmission risk service — data, training, serving and monitoring.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="DEBUG logging.")] = False,
) -> None:
    if verbose:
        import os

        os.environ["MLSERVICE_LOGGING__LEVEL"] = "DEBUG"
        get_settings.cache_clear()

    # Logging configuration reads settings, so a broken config would raise here
    # — before any command runs. That is wrong for `doctor`, whose entire job is
    # to diagnose a broken environment: a diagnostic that crashes on the fault
    # it exists to report is useless. Fall back to unconfigured logging and let
    # each command decide how to handle the failure.
    try:
        configure_logging(force=True)
    except Exception:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


data_app = typer.Typer(help="Dataset download, audit and split.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@data_app.command("download")
def data_download(
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
) -> None:
    """Download the dataset and verify it against the committed checksum."""
    from mlservice.data.download import download

    with request_context():
        result = download(force=force)
    typer.secho(
        f"  OK    {result.archive.name} ({'cached' if result.was_cached else 'downloaded'})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"        sha256 {result.sha256}")


@data_app.command("audit")
def data_audit(
    write_splits: Annotated[
        bool, typer.Option("--write-splits/--no-write-splits", help="Persist train/val/test.")
    ] = True,
) -> None:
    """Run the full data audit and write the findings.

    Emits reports/data_audit.json, which docs/DATA_AUDIT.md is generated from —
    so the document cannot drift away from what the code actually computed.
    """
    from mlservice.data.audit import load_raw, run_audit

    settings = get_settings()

    with request_context():
        raw = load_raw()
        report, split_result = run_audit(raw)

        out = settings.paths.reports / "data_audit.json"
        report.to_json(out)

        if write_splits:
            processed = settings.paths.data_processed
            processed.mkdir(parents=True, exist_ok=True)
            for name, frame in (
                ("train", split_result.train),
                ("val", split_result.val),
                ("test", split_result.test),
            ):
                frame.to_parquet(processed / f"{name}.parquet", index=False)

            # The drift baseline: written once, then left alone. A reference
            # window that silently tracks recent data cannot detect drift,
            # because it drifts along with it.
            reference = settings.paths.data_reference
            reference.mkdir(parents=True, exist_ok=True)
            split_result.train.to_parquet(reference / "reference_window.parquet", index=False)

    proxy = report.time_proxy
    colour = typer.colors.GREEN if proxy["passed"] else typer.colors.YELLOW
    typer.secho(
        f"  {'OK  ' if proxy['passed'] else 'WARN'}  time proxy: {proxy['claim']}",
        fg=colour,
        bold=True,
    )
    typer.echo(f"        {proxy['n_trending']}/{proxy['n_signals']} signals trend monotonically")

    sep = report.separability
    typer.secho(
        f"  {'FAIL' if sep['alarm_triggered'] else 'OK  '}  separability: "
        f"test ROC-AUC {sep['test_roc_auc']} (alarm above {sep['alarm_threshold']})",
        fg=typer.colors.RED if sep["alarm_triggered"] else typer.colors.GREEN,
    )

    lk = report.leakage
    typer.echo(f"        cleaning removed {lk['rows_removed']:,} rows ({lk['pct_removed']}%)")
    typer.echo(f"        report -> {out}")

    from mlservice.data.render_audit import render_to_file

    doc = render_to_file()
    typer.echo(f"        docs   -> {doc}")


train_app = typer.Typer(help="Model training, evaluation and registration.", no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("run")
def train_run(
    register_model: Annotated[
        bool, typer.Option("--register/--no-register", help="Register the champion.")
    ] = True,
) -> None:
    """Train candidates, calibrate, evaluate and register the champion."""
    from mlservice.models.train import run_training

    with request_context():
        summary = run_training(register_model=register_model)

    typer.secho(f"  OK    champion: {summary['champion']}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"        {summary['selection_rationale']['rule']}")

    for c in summary["candidates"]:
        pr = c["pr_auc"]
        mark = ">" if c["name"] == summary["champion"] else " "
        typer.echo(
            f"      {mark} {c['name']:24s} PR-AUC {pr['point']:.4f} "
            f"[{pr['lower']:.4f}-{pr['upper']:.4f}]  "
            f"recall {c['recall']:.3f}  prec {c['precision']:.3f}  "
            f"lift {c['lift_over_prevalence']:.2f}x"
        )

    from mlservice.models.render_card import render_to_file

    card = render_to_file()
    typer.echo(f"        model card -> {card}")

    if not summary["tracking_is_server"]:
        typer.secho(
            "  note  MLflow server unreachable — tracked to the local file store",
            fg=typer.colors.YELLOW,
        )


monitor_app = typer.Typer(help="Drift detection and monitoring.", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("calibrate")
def monitor_calibrate(
    write: Annotated[
        bool, typer.Option("--write/--dry-run", help="Write thresholds into configs/.")
    ] = True,
) -> None:
    """Derive per-feature drift thresholds from the training period's own churn.

    Replaces the PLACEHOLDER in configs/thresholds.yaml with a MEASURED value
    per feature: that feature's own 99th-percentile PSI between adjacent stable
    training windows.
    """
    import pandas as pd

    from mlservice.monitoring import null_calibration

    settings = get_settings()
    with request_context():
        train = pd.read_parquet(settings.paths.data_processed / "train.parquet")
        result = null_calibration.calibrate(train)
        null_calibration.save_report(result)
        if write:
            null_calibration.write_thresholds(result)

    summary = result.summary()
    typer.secho(
        f"  OK    calibrated {summary['n_features']} features over {summary['n_windows']} windows",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"        {summary['n_unclamped']} set their own threshold from measured churn")
    typer.echo(
        f"        {len(summary['clamped_to_floor'])} clamped to the {summary['floor']} floor"
    )
    typer.echo(
        f"        {len(summary['clamped_to_ceiling'])} clamped to the {summary['ceiling']} ceiling"
    )
    if not write:
        typer.secho("  note  dry run — nothing written", fg=typer.colors.YELLOW)


@monitor_app.command("replay")
def monitor_replay(
    induce_drift: Annotated[
        bool, typer.Option("--induce-drift", help="Deliberately manipulate later windows.")
    ] = False,
    inducer: Annotated[
        str, typer.Option("--inducer", help="age | utilisation | specialty")
    ] = "age",
    window_rows: Annotated[int | None, typer.Option("--window-rows")] = None,
) -> None:
    """Replay windows through the drift detectors.

    Without --induce-drift this replays the held-out test split untouched, so
    any drift found is REAL. With it, later windows are deliberately shifted and
    every artefact is labelled induced.
    """
    from mlservice.monitoring import replay as replay_mod
    from mlservice.monitoring import reports as reports_mod

    mode = "induced" if induce_drift else "real"
    with request_context():
        result = reports_mod.run_replay(mode=mode, inducer=inducer, window_rows=window_rows)
        path = replay_mod.save_result(result)

    if result.drift_origin == "induced":
        typer.secho(
            f"  NOTE  drift in later windows is ARTIFICIAL (inducer: {inducer}) — "
            "a demonstration, not a finding",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    else:
        typer.secho(
            "  NOTE  no manipulation — any drift below is real change in the 1999-2008 data",
            fg=typer.colors.CYAN,
        )

    typer.echo()
    typer.echo(f"      {'window':>6} {'origin':>9} {'breaching':>10}  features")
    for w in result.windows:
        d = w["drift"]
        names = ", ".join(d["breaching_features"][:4]) or "-"
        typer.echo(
            f"      {w['index']:>6} {w['drift_origin']:>9} "
            f"{d['n_breaching']:>4}/{d['n_features']:<5} {names}"
        )

    typer.echo()
    alert = result.alert
    colour = typer.colors.RED if alert["confirmed"] else typer.colors.GREEN
    typer.secho(
        f"  {'ALERT' if alert['confirmed'] else 'OK   '} {alert['reason']}", fg=colour, bold=True
    )
    if result.first_detection_window is not None:
        typer.echo(f"        first confirmed at window {result.first_detection_window}")
    typer.echo(f"        report -> {path}")


@monitor_app.command("check")
def monitor_check() -> None:
    """Analyse the most recent prediction-log window against the reference."""
    from mlservice.monitoring import drift as drift_mod
    from mlservice.monitoring import prediction_log
    from mlservice.monitoring import reports as reports_mod

    with request_context():
        records = prediction_log.read_records()
        if not records:
            typer.secho(
                "  note  prediction log is empty — nothing to check", fg=typer.colors.YELLOW
            )
            raise typer.Exit(code=0)

        import pandas as pd

        frame = pd.DataFrame([r["features_raw"] for r in records])
        reference = reports_mod.load_reference()
        report = drift_mod.analyse_window(reference=reference, current=frame)
        path = drift_mod.save_report(report)
        reports_mod.export_to_prometheus(report)

    typer.secho(
        f"  {'DRIFT' if report.breaching else 'OK   '} "
        f"{len(report.breaching)}/{len(report.features)} features breaching",
        fg=typer.colors.RED if report.breaching else typer.colors.GREEN,
        bold=True,
    )
    for f in report.breaching:
        typer.echo(f"        {f.feature:26s} psi {f.psi:.4f} > {f.threshold:.4f}")
    typer.echo(f"        report -> {path}")


# =============================================================================
# Retraining (Phase 7)
# =============================================================================

retrain_app = typer.Typer(
    help="Retraining triggers, promotion gates, promote and rollback.",
    no_args_is_help=True,
)
app.add_typer(retrain_app, name="retrain")


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _last_trained() -> datetime | None:
    """When the serving model was registered, read from the registry itself.

    Not from a hand-maintained date file: a freshness check whose input can
    drift out of sync with reality is worse than no check at all, because it
    reports confidently while being wrong.
    """
    from mlservice.models import registry as registry_mod

    try:
        from mlflow.tracking import MlflowClient

        registry_mod.setup_tracking()
        settings = get_settings()
        version = MlflowClient().get_model_version_by_alias(
            settings.model.name, settings.model.serving_alias
        )
        return datetime.fromtimestamp(version.creation_timestamp / 1000, tz=UTC)
    except Exception:
        return None


@retrain_app.command("check")
def retrain_check(
    drift_alert: Annotated[
        Path | None,
        typer.Option("--drift-alert", help="JSON alert state; default reads saved reports."),
    ] = None,
    label_drift: Annotated[
        Path | None, typer.Option("--label-drift", help="JSON label-drift result.")
    ] = None,
    manual: Annotated[bool, typer.Option("--manual", help="Force a manual trigger.")] = False,
    requested_by: Annotated[str | None, typer.Option("--by", help="Who requested it.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Evaluate every retraining trigger and report which, if any, fired."""
    from mlservice.monitoring import drift as drift_mod
    from mlservice.retraining import trigger as trigger_mod

    with request_context():
        reports = drift_mod.load_reports()

        alert = _load_json(drift_alert)
        if not alert:
            counts = drift_mod.breaching_counts(reports)
            alert = drift_mod.alert_state_from_counts(counts) if counts else {}

        labels = _load_json(label_drift)
        if not labels:
            for report in reversed(reports):
                if report.get("labels"):
                    labels = report["labels"]
                    break

        decision = trigger_mod.evaluate_triggers(
            last_trained=_last_trained(),
            drift_alert=alert,
            label_drift=labels,
            manual=manual,
            requested_by=requested_by,
        )

    if as_json:
        typer.echo(json.dumps(decision.to_dict(), indent=2))
        raise typer.Exit(code=0)

    if not reports:
        typer.secho(
            "  note  no drift reports on disk — the drift and performance "
            "triggers had no evidence to evaluate, which is not the same as "
            "finding no drift. Run `mlservice monitor check` first.",
            fg=typer.colors.YELLOW,
        )

    typer.secho(
        f"  RETRAIN: {'YES' if decision.retrain else 'no '}  --  {decision.reason}",
        fg=typer.colors.YELLOW if decision.retrain else typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"        evaluated {len(reports)} drift window(s)")
    for t in decision.triggers:
        typer.echo(f"        {'FIRED' if t.fired else '  -  '}  {t.name:12s} {t.reason}")
    typer.echo()
    typer.echo("        deliberately NOT triggers:")
    for item in trigger_mod.NOT_TRIGGERS:
        typer.echo(f"          - {item}")


@retrain_app.command("evidence")
def retrain_evidence(
    summary: Annotated[
        Path | None, typer.Option("--summary", help="Training summary JSON.")
    ] = None,
    behavioral: Annotated[
        Path | None, typer.Option("--behavioral", help="pytest --json-report for tests/behavior.")
    ] = None,
    data_quality: Annotated[
        Path | None, typer.Option("--data-quality", help="pytest --json-report for tests/data.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Where to write the evidence.")] = None,
) -> None:
    """Assemble the evidence the promotion gates read, from a training run.

    Gates that are unit-tested but never wired to the pipeline's real output are
    gates that have never blocked anything. This is the wiring.

    Evidence that cannot be collected is left **absent, not assumed** — the
    corresponding gate then blocks with "the suite did not run", which is the
    correct verdict and a different message from "a test failed".
    """
    from mlservice.retraining import evidence as evidence_mod

    with request_context():
        payload = evidence_mod.build_evidence(
            summary=evidence_mod.load_summary(summary),
            behavioral=(evidence_mod.parse_pytest_report(behavioral) if behavioral else None),
            data_quality=(
                {"suite_passed": evidence_mod.parse_pytest_report(data_quality)["failures"] == []}
                if data_quality
                else None
            ),
            run_canary=True,
        )
        path = evidence_mod.save_evidence(payload, out)

    loads = payload["artifact_loads"]
    typer.secho(
        f"  {'OK   ' if loads else 'FAIL '} canary: {payload['canary']}",
        fg=typer.colors.GREEN if loads else typer.colors.RED,
        bold=True,
    )
    typer.echo(f"        model {payload['model']}  schema {payload['feature_schema_hash']}")
    typer.echo(
        f"        PR-AUC {payload['pr_auc']['point']:.4f}  "
        f"Brier {payload['brier']['point']:.4f}  "
        f"ECE {payload['calibration']['ece']:.4f}"
    )

    for gap in payload.get("evidence_gaps", []):
        typer.secho(
            f"  gap   {gap} evidence absent — that gate will BLOCK, by design",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"        evidence -> {path}")


@retrain_app.command("gates")
def retrain_gates(
    challenger: Annotated[Path, typer.Option("--challenger", help="Challenger metrics JSON.")],
    incumbent: Annotated[Path, typer.Option("--incumbent", help="Incumbent metrics JSON.")],
    out: Annotated[Path | None, typer.Option("--out", help="Write the decision JSON here.")] = None,
) -> None:
    """Run every promotion gate. Exits non-zero if the challenger is blocked."""
    from mlservice.retraining import gates as gates_mod

    with request_context():
        decision = gates_mod.evaluate_promotion(
            challenger=_load_json(challenger), incumbent=_load_json(incumbent)
        )

    typer.secho(
        f"  {'PROMOTE' if decision.promote else 'BLOCKED'}  --  {decision.reason}",
        fg=typer.colors.GREEN if decision.promote else typer.colors.RED,
        bold=True,
    )
    for g in decision.gates:
        mark = "pass" if g.passed else ("FAIL" if g.blocking else "warn")
        typer.echo(f"        {mark}  {g.name:14s} {g.reason}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
        typer.echo(f"        decision -> {out}")

    raise typer.Exit(code=0 if decision.promote else 1)


@retrain_app.command("promote")
def retrain_promote(
    version: Annotated[str, typer.Option("--version", help="Model version to promote.")],
    decision: Annotated[
        Path, typer.Option("--decision", help="Gate decision JSON from `retrain gates`.")
    ],
    trigger: Annotated[str, typer.Option("--trigger", help="What caused this retrain.")] = "manual",
    approver: Annotated[str, typer.Option("--approver")] = "automated",
) -> None:
    """Point the serving alias at a version — only if the gates passed.

    The gate decision is a required argument rather than an optional flag.
    Promoting without a recorded decision is precisely the failure mode the
    gates exist to prevent, so the CLI does not offer a way to do it.
    """
    from mlservice.models import registry as registry_mod
    from mlservice.retraining import promote as promote_mod

    payload = _load_json(decision)

    with request_context():
        registry_mod.setup_tracking()
        if not payload.get("promote"):
            typer.secho(
                f"  BLOCKED  gates did not pass: {payload.get('reason', 'unknown')}",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

        entry = promote_mod.promote(
            version=version,
            trigger=trigger,
            gates_passed=True,
            approver=approver,
            detail={"gate_decision": payload.get("reason")},
        )

    typer.secho(
        f"  PROMOTED  {entry.from_version} -> {entry.to_version}  (alias {entry.alias})",
        fg=typer.colors.GREEN,
        bold=True,
    )


@retrain_app.command("rollback")
def retrain_rollback(
    reason: Annotated[str, typer.Option("--reason", help="Why. Recorded in the audit trail.")],
    approver: Annotated[str, typer.Option("--approver")] = "operator",
) -> None:
    """Point the serving alias back at the previous version."""
    from mlservice.models import registry as registry_mod
    from mlservice.retraining import promote as promote_mod

    with request_context():
        registry_mod.setup_tracking()
        entry = promote_mod.rollback(reason=reason, approver=approver)

    typer.secho(
        f"  ROLLED BACK  {entry.from_version} -> {entry.to_version}",
        fg=typer.colors.YELLOW,
        bold=True,
    )


@retrain_app.command("verify-rollback")
def retrain_verify_rollback() -> None:
    """Exercise a real promote -> rollback cycle against the registry.

    Run in CI and before any demo. A rollback path that has never been
    exercised is a plan, not a capability — and the moment you need it is the
    worst possible time to find out it does not work.
    """
    from mlservice.retraining import promote as promote_mod

    with request_context():
        result = promote_mod.verify_rollback_path()

    for step in result["steps"]:
        typer.echo(
            f"        {step['step']:9s} -> {step.get('to')}   alias now: {step.get('alias_now')}"
        )
    typer.secho(
        f"  {'VERIFIED' if result['verified'] else 'FAILED  '}  {result['reason']}",
        fg=typer.colors.GREEN if result["verified"] else typer.colors.RED,
        bold=True,
    )
    raise typer.Exit(code=0 if result["verified"] else 1)


@retrain_app.command("history")
def retrain_history(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    """Show the promotion and rollback audit trail."""
    from mlservice.retraining import promote as promote_mod

    entries = promote_mod.history()
    if not entries:
        typer.secho("  note  no promotions recorded yet", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.echo(f"  {len(entries)} entries, showing the last {min(limit, len(entries))}:")
    for e in entries[-limit:]:
        colour = typer.colors.YELLOW if e["action"] == "rollback" else typer.colors.GREEN
        typer.secho(
            f"        {e['timestamp_utc'][:19]}  {e['action']:8s} "
            f"{e['from_version']} -> {e['to_version']}  "
            f"trigger={e['trigger']} by={e['approver']}",
            fg=colour,
        )


@app.command("config")
def show_config(
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    show_thresholds: Annotated[
        bool, typer.Option("--thresholds", help="Include operational thresholds.")
    ] = False,
) -> None:
    """Show the fully resolved configuration.

    Useful in an incident: it prints what the process *actually* resolved after
    YAML layering and environment overrides, which is frequently not what the
    person debugging assumes.
    """
    settings = get_settings()
    payload: dict[str, object] = json.loads(settings.model_dump_json())
    if show_thresholds:
        payload["thresholds"] = get_thresholds().model_dump()

    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    typer.secho(f"mlservice {__version__}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"root: {PROJECT_ROOT}")
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def doctor() -> None:
    """Check that this environment can actually run the service.

    Exits non-zero if anything required is missing, so CI and the Makefile can
    gate on it rather than failing later with a less obvious error.
    """
    problems: list[str] = []
    notes: list[str] = []

    if sys.version_info[:2] != (3, 11):
        problems.append(
            f"Python 3.11 required, running {sys.version_info.major}.{sys.version_info.minor}"
        )

    # Resolve settings BEFORE touching the logger: get_logger() configures
    # logging, which itself reads settings. On a broken config that would raise
    # out of doctor instead of being reported by it.
    settings: Settings | None
    try:
        settings = get_settings()
        notes.append(f"config resolved for env={settings.env}")
    except Exception as exc:
        problems.append(f"configuration failed to resolve: {_first_line(exc)}")
        settings = None

    # Structured logging only once we know it can be configured.
    log = get_logger(__name__) if settings is not None else None

    with request_context() as rid:
        if log:
            log.info("doctor_started", request_id=rid)

        try:
            thresholds = get_thresholds()
            raw = thresholds.model_dump()
            placeholders = _find_placeholders(raw)
            notes.append(f"thresholds schema_version={raw.get('schema_version')}")
            if placeholders:
                notes.append(
                    f"{len(placeholders)} PLACEHOLDER threshold block(s) still unmeasured: "
                    + ", ".join(sorted(placeholders))
                )
        except Exception as exc:
            problems.append(f"thresholds failed to load: {exc}")

        if settings is not None:
            for name in ("data_raw", "data_processed", "reports", "logs"):
                path = getattr(settings.paths, name)
                if not path.exists():
                    notes.append(f"path missing (created on first use): {name}={path}")

        # Env-var typos cannot be caught by validation — pydantic-settings never
        # reads a variable that maps to no field, so the override silently does
        # nothing. Surfacing them here is the only place this gets noticed.
        for name in stray_env_vars():
            notes.append(f"unrecognised env var, override will NOT apply: {name}")

        for line in notes:
            typer.secho(f"  note  {line}", fg=typer.colors.YELLOW)

        if problems:
            for line in problems:
                typer.secho(f"  FAIL  {line}", fg=typer.colors.RED, bold=True)
            # `log` is None precisely when settings failed to resolve, which is
            # the most common reason to be here. The terminal report above is
            # the real output; structured logging is a bonus when available.
            if log:
                log.error("doctor_failed", problem_count=len(problems))
            raise typer.Exit(code=1)

        typer.secho("  OK    environment is fit to run", fg=typer.colors.GREEN, bold=True)
        if log:
            log.info("doctor_passed", note_count=len(notes))


def _first_line(exc: Exception) -> str:
    """Condense an exception to something a human scans in one glance.

    A pydantic ValidationError renders as ~40 lines with a docs URL. In a
    diagnostic summary that buries the other findings, so keep the offending
    field and its message and drop the rest.
    """
    text = str(exc).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return exc.__class__.__name__
    useful = [ln for ln in lines if not ln.startswith("For further information")]
    return " | ".join(useful[:3])


def _find_placeholders(node: object, path: str = "") -> list[str]:
    """Walk the thresholds tree and collect blocks still tagged PLACEHOLDER.

    Phase 8 turns this into a hard CI gate: no PLACEHOLDER may survive into the
    v1.0.0 tag, because an unmeasured threshold that looks configured is worse
    than an obviously empty one.
    """
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("provenance") == "PLACEHOLDER":
            found.append(path or "<root>")
        for key, value in node.items():
            found.extend(_find_placeholders(value, f"{path}.{key}" if path else str(key)))
    return found


if __name__ == "__main__":  # pragma: no cover
    app()
