"""Machine-readable output must be redirectable.

A ``--json`` flag exists for automation. If logs share the stream, redirecting
it produces a file that is not JSON, and every caller breaks — which is exactly
what happened the first time the retraining workflow ran:

    mlservice retrain check --json > triggers.json
    json.decoder.JSONDecodeError: Extra data: line 1 column 5 (char 4)

Diagnostics belong on stderr precisely so a program's actual output stays
pipeable. These tests run the CLI as a subprocess, because that is the only way
to observe real stream separation — an in-process runner that captures both
into one buffer would pass while the bug was still there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "MLSERVICE_MLFLOW__TRACKING_URI": "",
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [sys.executable, "-m", "mlservice.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


class TestJsonGoesToStdoutAlone:
    def test_retrain_check_json_is_parseable_from_stdout(self) -> None:
        """The regression test for the workflow failure."""
        result = _run("retrain", "check", "--json")
        assert result.returncode == 0, result.stderr[-500:]

        payload = json.loads(result.stdout)  # must not raise
        assert "retrain" in payload
        assert "triggers" in payload

    def test_logs_are_present_but_on_stderr(self) -> None:
        """Separation, not suppression.

        The logs must still be emitted — losing them to make the JSON parse
        would trade one problem for a worse one.
        """
        result = _run("retrain", "check", "--json")
        assert result.stderr.strip(), "diagnostics disappeared entirely"
        # The marker is the structured-logging signature rather than a specific
        # event name: which events fire depends on log level and on whether a
        # registry happens to be reachable, and pinning one of them would make
        # this test fail for reasons unrelated to stream separation.
        assert "service=mlservice" in result.stderr or "mlservice" in result.stderr

    def test_stdout_contains_nothing_but_the_document(self) -> None:
        """No banner, no progress line, no trailing note."""
        result = _run("retrain", "check", "--json")
        assert result.stdout.lstrip().startswith("{")
        assert result.stdout.rstrip().endswith("}")
