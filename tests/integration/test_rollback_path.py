"""Promote and roll back against a **real** MLflow registry.

The unit tests in ``tests/unit/test_retraining.py`` use a fake client. That is
the right tool for testing the logic and the wrong tool for the question this
file asks: does the rollback path work against a real registry, with its real
types and its real ordering guarantees?

Both Phase 7 bugs lived exactly in that gap, and no amount of mocking would have
surfaced either:

*   ``search_model_versions`` returns ``.version`` as an **int** while
    ``get_model_version_by_alias`` returns a **str**, so ``"2" == 2`` was False
    and a working cycle reported ``promotion did not take effect: alias is 2,
    expected 2``.
*   ``search_model_versions`` makes no ordering guarantee, so promoting
    "whichever version came back first" promoted the one already serving and
    recorded ``2 -> 2`` — a no-op that reported ``VERIFIED: True``.

So this runs the whole cycle on a real SQLite-backed registry in a temp
directory. No server, no network, no dataset — CI-runnable, and it fails if
either bug returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def real_registry(tmp_path: Path, env: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """A real MLflow registry with two registered versions of one model.

    SQLite rather than the file store: MLflow's filesystem backend never
    supported the model registry, which is the thing under test.
    """
    mlflow = pytest.importorskip("mlflow")
    from sklearn.dummy import DummyClassifier

    store = tmp_path / "mlruns"
    store.mkdir()
    uri = f"sqlite:///{(store / 'registry.db').resolve().as_posix()}"

    env(
        MLSERVICE_MLFLOW__TRACKING_URI=uri,
        MLSERVICE_PATHS__REPORTS=str(tmp_path / "reports"),
        MLSERVICE_MODEL__NAME="rollback-test-model",
        MLSERVICE_MODEL__SERVING_ALIAS="champion",
    )

    # setup_tracking() would fall back to the project's own store; the point of
    # this fixture is that the registry is disposable.
    from mlservice.models import registry as registry_mod

    monkeypatch.setattr(registry_mod, "setup_tracking", lambda *a, **k: (uri, False))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("rollback-test")

    model = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    for _ in range(2):
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(model, name="model")
            mlflow.register_model(info.model_uri, "rollback-test-model")

    return uri


def test_verify_rollback_path_moves_the_alias_and_returns_it(real_registry: str) -> None:
    """The end-to-end claim the README makes, executed rather than asserted."""
    from mlservice.retraining import promote as promote_mod

    result = promote_mod.verify_rollback_path()

    assert result["verified"], result["reason"]
    steps = {s["step"]: s for s in result["steps"]}

    # It moved...
    assert steps["promote"]["from"] != steps["promote"]["to"]
    assert steps["promote"]["alias_now"] == result["latest_version"]
    # ...and it came back.
    assert steps["rollback"]["alias_now"] == result["oldest_version"]


def test_version_identifiers_compare_across_both_mlflow_apis(real_registry: str) -> None:
    """Regression test for the int/str mismatch.

    Pinned directly rather than only through ``verify_rollback_path``, because
    the failure mode was a *self-contradictory* message — "alias is 2, expected
    2" — and a test that only checks the verdict would not say why.
    """
    from mlflow.tracking import MlflowClient

    from mlservice.retraining import promote as promote_mod

    client = MlflowClient()
    versions = client.search_model_versions("name='rollback-test-model'")
    latest = str(max(versions, key=lambda v: int(v.version)).version)

    promote_mod.promote(version=latest, trigger="test", gates_passed=True)

    champion = promote_mod.current_champion()
    assert champion == latest
    assert isinstance(champion, str)


def test_a_blocked_challenger_never_reaches_the_registry(real_registry: str) -> None:
    """The gate refusal must happen before the alias moves, not after."""
    from mlservice.retraining import promote as promote_mod

    before = promote_mod.current_champion()

    with pytest.raises(ValueError, match="gates did not pass"):
        promote_mod.promote(version="2", trigger="drift", gates_passed=False)

    assert promote_mod.current_champion() == before


def test_the_audit_trail_survives_a_full_cycle(real_registry: str) -> None:
    """Promotion history is the answer to "how did this ship?"."""
    from mlservice.retraining import promote as promote_mod

    promote_mod.verify_rollback_path()
    entries = promote_mod.history()

    actions = [e["action"] for e in entries]
    assert "promote" in actions
    assert "rollback" in actions

    rollbacks = [e for e in entries if e["action"] == "rollback"]
    assert rollbacks[-1]["gates_passed"] is False, "a rollback is deliberately not gated"
