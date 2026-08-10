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
from typing import Annotated

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
