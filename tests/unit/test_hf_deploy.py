"""The Hugging Face Space deployment contract.

Three couplings here are invisible at review time and fail silently in
production, which is the worst combination. Each gets a test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SPACE_README = ROOT / "deploy" / "hf-space" / "README.md"
DOCKERFILE_HF = ROOT / "deploy" / "docker" / "Dockerfile.hf"


def _frontmatter() -> dict[str, Any]:
    text = SPACE_README.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, "the Space README must open with YAML frontmatter"
    return dict(yaml.safe_load(match.group(1)))


class TestSpaceFrontmatter:
    def test_declares_the_docker_sdk(self) -> None:
        assert _frontmatter()["sdk"] == "docker"

    def test_app_port_matches_the_dockerfile(self) -> None:
        """A mismatch here shows as a permanent "connection errored out".

        HF routes to the port declared in the frontmatter. If the container
        listens elsewhere the Space builds, starts, passes every local check,
        and is unreachable — with no error attributable to either file alone.
        """
        declared = _frontmatter()["app_port"]
        dockerfile = DOCKERFILE_HF.read_text(encoding="utf-8")

        exposed = re.search(r"^EXPOSE\s+(\d+)", dockerfile, re.M)
        assert exposed, "Dockerfile.hf must EXPOSE a port"
        assert int(exposed.group(1)) == declared

        served = re.search(r"--port\s+(\d+)", dockerfile)
        assert served, "the CMD must pass --port"
        assert int(served.group(1)) == declared

    def test_carries_the_non_clinical_disclaimer(self) -> None:
        """The disclaimer is required on every public surface, and a Space is
        the most public one this project has."""
        assert "NOT FOR CLINICAL USE" in SPACE_README.read_text(encoding="utf-8")


class TestModelPathContract:
    def test_fetch_target_matches_the_loader_fallback(self, env: Any) -> None:
        """The coupling that would break the Space silently.

        ``fetch_model.py`` writes the artifact and ``model_loader`` reads it,
        computing the path *independently* — one from ``paths.models``, the
        other from ``model.local_fallback`` resolved against
        ``paths.models.parent``. They agree today. If they ever stop agreeing,
        the Space downloads the model successfully and then reports 503
        forever, with both halves looking correct in isolation.
        """
        env(MLSERVICE_ENV="hf", MLSERVICE_MLFLOW__TRACKING_URI="")

        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import fetch_model

            from mlservice.api.model_loader import ModelStore
            from mlservice.config import get_settings

            written = fetch_model.target_path().resolve()
            read = ModelStore._fallback_path(get_settings()).resolve()
        finally:
            sys.path.remove(str(ROOT / "scripts"))

        assert written == read, (
            f"fetch_model writes {written} but model_loader reads {read} — "
            "the Space would download the model and still report itself unready"
        )


class TestFetchModelFailsOpen:
    def test_missing_repo_variable_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No model configured must not crash loop.

        An unready service that explains itself is debuggable. A container that
        restarts forever is not, and on a Space the logs scroll away with it.
        """
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import fetch_model

            monkeypatch.delenv(fetch_model.REPO_ENV, raising=False)
            assert fetch_model.main() == 0
        finally:
            sys.path.remove(str(ROOT / "scripts"))
