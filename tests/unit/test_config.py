"""Configuration behaviour.

These tests exist because config bugs are silent. A wrong decision threshold or
a dropped override does not raise — it just serves subtly wrong predictions, and
the mistake surfaces much later as a metric nobody can explain.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlservice.config import (
    PROJECT_ROOT,
    DataSettings,
    Settings,
    get_settings,
    get_thresholds,
    stray_env_vars,
)

pytestmark = pytest.mark.unit


class TestLayering:
    def test_resolves_under_test_env(self) -> None:
        settings = get_settings()
        assert settings.env == "test"

    def test_env_overlay_beats_base_yaml(self) -> None:
        # base.yaml sets api.port 8000; test.yaml overrides to 8001.
        assert get_settings().api.port == 8001

    def test_env_var_beats_yaml(self, env: object) -> None:
        settings = env(MLSERVICE_API__PORT="9999")  # type: ignore[operator]
        assert settings.api.port == 9999

    def test_settings_are_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_relative_paths_resolve_against_project_root(self) -> None:
        paths = get_settings().paths
        assert paths.data_raw.is_absolute()
        assert paths.data_raw == PROJECT_ROOT / "data" / "raw"


class TestFailFast:
    """A misconfigured service must refuse to start, not start and be wrong."""

    def test_rejects_unknown_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLSERVICE_ENV", "production")
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="MLSERVICE_ENV must be one of"):
            get_settings()

    def test_rejects_nested_key_typo(self) -> None:
        # `prot` for `port`. Without extra="forbid" this is silently discarded
        # and the operator believes an override took effect that did not.
        with pytest.raises(ValidationError, match="extra_forbidden"):
            Settings(api={"prot": 8000})  # type: ignore[arg-type]

    def test_rejects_split_fractions_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
            DataSettings(train_fraction=0.7, val_fraction=0.2, test_fraction=0.2)

    def test_accepts_split_fractions_that_do_sum_to_one(self) -> None:
        data = DataSettings(train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
        assert data.train_fraction == 0.6

    @pytest.mark.parametrize("bad", ["", "   ", "demo only"])
    def test_refuses_a_blanked_disclaimer(self, bad: str) -> None:
        # The non-clinical disclaimer is a requirement of this project, not a
        # label to be trimmed away for a cleaner API response.
        with pytest.raises(ValidationError, match="disclaimer"):
            Settings(api={"disclaimer": bad})  # type: ignore[arg-type]

    def test_disclaimer_mentions_non_clinical_use(self) -> None:
        assert "NOT FOR CLINICAL USE" in get_settings().api.disclaimer.upper()


class TestStrayEnvVars:
    """The one gap extra="forbid" cannot close.

    A root-level env typo maps to no field, so pydantic-settings never reads it
    and validation never sees it — the override silently does nothing.
    """

    def test_detects_root_level_typo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLSERVICE_DEBUGG", "true")
        assert "MLSERVICE_DEBUGG" in stray_env_vars()

    def test_accepts_known_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLSERVICE_DEBUG", "true")
        monkeypatch.setenv("MLSERVICE_API__PORT", "8080")
        strays = stray_env_vars()
        assert "MLSERVICE_DEBUG" not in strays
        assert "MLSERVICE_API__PORT" not in strays

    def test_ignores_unprefixed_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH_LIKE_THING", "x")
        assert "PATH_LIKE_THING" not in stray_env_vars()


class TestThresholds:
    """Operational numbers must be traceable, not merely present."""

    def test_loads(self) -> None:
        assert get_thresholds().model_dump()["schema_version"] == 1

    def test_every_block_declares_provenance(self) -> None:
        """No unjustified number. This is the whole point of the file.

        A threshold without a provenance tag is indistinguishable from one
        copied off a blog post, which is exactly the failure mode the project
        is meant to avoid.
        """
        raw = get_thresholds().model_dump()
        missing = list(_blocks_without_provenance(raw))
        assert not missing, f"threshold blocks without a provenance tag: {missing}"

    def test_provenance_values_are_from_the_known_set(self) -> None:
        allowed = {"MEASURED", "DERIVED", "STANDARD", "PLACEHOLDER"}
        found = set(_provenance_values(get_thresholds().model_dump()))
        assert found <= allowed, f"unknown provenance tags: {found - allowed}"

    def test_placeholders_name_the_phase_that_replaces_them(self) -> None:
        """A PLACEHOLDER without an owner becomes permanent."""
        raw = get_thresholds().model_dump()
        orphaned = [
            path
            for path, block in _walk_blocks(raw)
            if block.get("provenance") == "PLACEHOLDER" and "replaced_in_phase" not in block
        ]
        assert not orphaned, f"PLACEHOLDER blocks with no owning phase: {orphaned}"

    def test_calibration_gate_is_present_and_binding(self) -> None:
        """Calibration is a deployment gate, not a report. See ADR 0002."""
        gate = get_thresholds().model_dump()["promotion"]["calibration"]
        assert gate["max_brier_ratio_vs_incumbent"] <= 1.05
        assert 0 < gate["max_ece"] <= 0.10

    def test_drift_alerting_requires_confirmation(self) -> None:
        """Single-window blips must not page — that is how a pager gets ignored."""
        alert = get_thresholds().model_dump()["drift"]["alert"]["data_drift"]
        assert alert["consecutive_windows"] >= 2
        assert alert["min_features_breaching"] >= 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: Structural keys that group other blocks rather than defining a threshold.
_CONTAINER_KEYS = {"slo", "drift", "alert", "promotion", "retraining", "rollout", "labels"}


def _walk_blocks(node: object, path: str = "") -> list[tuple[str, dict]]:
    """Yield every mapping in the tree, with its dotted path."""
    out: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        out.append((path or "<root>", node))
        for key, value in node.items():
            out.extend(_walk_blocks(value, f"{path}.{key}" if path else str(key)))
    return out


def _blocks_without_provenance(root: dict) -> list[str]:
    """Blocks that set values with no provenance on them or on an ancestor.

    Provenance is inherited. ``slo.burn_rate`` declares ``STANDARD`` and its
    ``page``/``ticket`` children are the detail of that one decision — requiring
    them to repeat the tag would be noise, and noisy requirements get satisfied
    by copy-paste rather than by thought.
    """
    tagged: set[str] = {path for path, block in _walk_blocks(root) if "provenance" in block}

    def has_tagged_ancestor(path: str) -> bool:
        parts = path.split(".")
        return any(".".join(parts[:i]) in tagged for i in range(1, len(parts)))

    missing = []
    for path, block in _walk_blocks(root):
        if path == "<root>" or path.split(".")[-1] in _CONTAINER_KEYS:
            continue
        if "provenance" in block or has_tagged_ancestor(path):
            continue
        if any(not isinstance(v, dict) for v in block.values()):
            missing.append(path)
    return missing


def _provenance_values(root: dict) -> list[str]:
    return [b["provenance"] for _, b in _walk_blocks(root) if "provenance" in b]
