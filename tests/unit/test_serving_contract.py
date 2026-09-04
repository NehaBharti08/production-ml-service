"""The model's serving contract must travel with the model.

Regression tests for the most consequential bug in this project, which shipped
through CI, a full unit suite, and a behaviour suite without one of them
noticing — and was found by making a single HTTP request to a running
container.

**What happened.** The decision threshold and feature schema hash lived only in
``reports/training_summary.json``, a build-time report that does not travel with
the artifact. The container mounts ``models/`` and nothing else, so inside it
that file did not exist and the API fell back to the config placeholder of 0.5
against a model tuned to 0.1011.

**Why nothing caught it.** Nothing raised. The service returned 200 with a
plausible probability and ``flagged: false`` for every patient — because scores
cluster near 0.1 and almost nothing clears 0.5. A screening model that never
flags anyone, reporting itself perfectly healthy.

The unit tests all ran on a dev machine where ``reports/`` exists, so they
exercised the path that works. CI's container smoke test asserted only that an
unready service *reports itself unready* — a correct test of the failure path,
leaving the success path untested. Every check verified the world in which the
bug is invisible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlservice.api.model_loader import ModelStore

pytestmark = pytest.mark.unit

TRAINED_THRESHOLD = 0.10106382978723404
SCHEMA_HASH = "06f5f0b873ca95f6"  # pragma: allowlist secret


@pytest.fixture
def artifact_dir(tmp_path: Path, env: Any) -> Path:
    """A models/champion directory with no reports/ anywhere near it.

    This is the container's situation, which no test previously reproduced.
    """
    models = tmp_path / "models"
    champion = models / "champion"
    champion.mkdir(parents=True)
    (champion / "model.joblib").write_bytes(b"not a real model")

    env(
        MLSERVICE_PATHS__MODELS=str(models),
        MLSERVICE_MODEL__DECISION_THRESHOLD="0.5",  # the placeholder that shipped
        # No reports/ here. This is precisely the container's situation, and
        # not isolating it is why the first version of this test passed by
        # reading the real repository's summary.
        MLSERVICE_PATHS__REPORTS=str(tmp_path / "no-reports-here"),
    )
    return champion


def _write_sidecar(directory: Path, **overrides: Any) -> None:
    payload = {
        "champion": "logistic_l2",
        "champion_threshold": TRAINED_THRESHOLD,
        "feature_schema_hash": SCHEMA_HASH,
        **overrides,
    }
    (directory / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


class TestThresholdTravelsWithTheArtifact:
    def test_sidecar_supplies_the_trained_threshold(self, artifact_dir: Path) -> None:
        """The fix, stated as a test.

        With only the artifact and its sidecar — no reports/ — the served
        threshold must still be the trained one.
        """
        _write_sidecar(artifact_dir)
        assert ModelStore._decision_threshold() == pytest.approx(TRAINED_THRESHOLD)

    def test_sidecar_supplies_the_schema_hash(self, artifact_dir: Path) -> None:
        _write_sidecar(artifact_dir)
        assert ModelStore._schema_hash() == SCHEMA_HASH

    def test_without_a_sidecar_it_falls_back_to_config(self, artifact_dir: Path) -> None:
        """The old behaviour, pinned so the danger stays visible.

        This is not a bug being preserved — a deployment with neither sidecar
        nor report has no better answer available. It is pinned so that the
        0.5 fallback is a *tested, documented* degradation rather than a
        surprise, and the loader logs a warning when it happens.
        """
        assert ModelStore._decision_threshold() == pytest.approx(0.5)

    def test_the_placeholder_and_the_trained_value_genuinely_differ(self) -> None:
        """A meta-test: the fixture must be capable of catching the bug.

        If the config placeholder ever equalled the trained threshold, every
        test above would pass while proving nothing — the same way the original
        rollback verification passed by never moving the alias.
        """
        assert TRAINED_THRESHOLD != 0.5
        assert abs(TRAINED_THRESHOLD - 0.5) > 0.3, (
            "the placeholder must be far enough from the trained value that "
            "confusing them changes the flagged decision"
        )


class TestSidecarPrecedence:
    def test_sidecar_wins_over_a_stale_report(self, artifact_dir: Path, tmp_path: Path) -> None:
        """The artifact's own contract beats a report that may describe a
        different model entirely.

        A stale reports/ describing a previously trained model is exactly the
        situation where trusting the report serves one model's coefficients
        with another model's operating point.
        """
        _write_sidecar(artifact_dir, champion_threshold=0.42)
        assert ModelStore._decision_threshold() == pytest.approx(0.42)

    def test_a_corrupt_sidecar_does_not_break_loading(self, artifact_dir: Path) -> None:
        """Degrade, do not crash. Refusing to serve over unreadable metadata
        would be the worse trade — but the config fallback must then apply."""
        (artifact_dir / "metadata.json").write_text("{not json", encoding="utf-8")
        assert ModelStore._decision_threshold() == pytest.approx(0.5)


class TestRegistryWritesTheSidecar:
    def test_metadata_is_written_next_to_the_artifact(self, tmp_path: Path, env: Any) -> None:
        from mlservice.models import registry

        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "training_summary.json").write_text(
            json.dumps(
                {
                    "champion": "logistic_l2",
                    "champion_threshold": TRAINED_THRESHOLD,
                    "feature_schema_hash": SCHEMA_HASH,
                }
            ),
            encoding="utf-8",
        )
        env(MLSERVICE_PATHS__REPORTS=str(reports))

        out = registry.write_artifact_metadata(tmp_path / "champion")
        assert out is not None
        assert out.is_file()

        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["champion_threshold"] == pytest.approx(TRAINED_THRESHOLD)
        assert written["feature_schema_hash"] == SCHEMA_HASH

    def test_no_summary_returns_none_rather_than_writing_junk(
        self, tmp_path: Path, env: Any
    ) -> None:
        from mlservice.models import registry

        env(MLSERVICE_PATHS__REPORTS=str(tmp_path / "empty"))
        assert registry.write_artifact_metadata(tmp_path / "champion") is None
