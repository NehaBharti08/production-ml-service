"""Fetch the serving artifact from a Hugging Face Model repo.

**Why this exists.** The model binary must never live in git — it is in
`.gitignore` alongside the MLflow artifact store, and committing it would be one
of the things this project promised not to do. But a Hugging Face Space builds
from a git repo, so the artifact cannot simply be baked in at build time either.

So the registry pattern splits in two, which is what a real deployment does
anyway:

    HF Model repo   = the artifact store (the binary lives here)
    HF Space        = the runtime       (downloads it at startup)

**What this deliberately does NOT do: change the serving code.** It writes the
artifact to the path `model_loader` already treats as its local fallback, then
gets out of the way. The API is identical in the Space and on a laptop, which is
the property that makes the Space evidence of anything at all.

**On failure it does not raise.** If the download fails the service still starts
and reports itself *unready* with a reason — because `/health/ready` already
exists to say "cannot serve traffic", and a crash loop would say the same thing
far less usefully. A Space that boots and honestly reports 503 is debuggable; a
Space that restarts forever is not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variables, set in the Space's Settings.
#: HF_TOKEN is only needed for a private model repo; a public one needs none.
REPO_ENV = "HF_MODEL_REPO"
FILE_ENV = "HF_MODEL_FILE"
TOKEN_ENV = "HF_TOKEN"

DEFAULT_FILE = "model.joblib"


def target_path() -> Path:
    """Where model_loader looks for its local fallback."""
    from mlservice.config import get_settings

    return Path(get_settings().paths.models) / "champion" / "model.joblib"


def _fetch_contract(repo: str, directory: Path, token: str | None) -> None:
    """Download ``metadata.json`` — the artifact's serving contract.

    **Without this the model loads and is quietly wrong.** The threshold falls
    back to the config placeholder of 0.5 against a model tuned to 0.1011, and
    the service returns 200 with a plausible probability while flagging nobody.

    That bug has now appeared five times in this project, in five different
    places. It reappeared *here* the first time this image was run against the
    real model repo: the sidecar mechanism existed and worked, and the fetcher
    simply never brought the sidecar along. A contract that travels with the
    artifact only helps if everything that moves the artifact moves both.

    A missing contract is loud rather than fatal. The service can still serve —
    and an operator who reads one line of logs can see exactly why the
    threshold looks wrong — but nothing about it is silent.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=repo, filename="metadata.json", token=token)
        (directory / "metadata.json").write_bytes(Path(path).read_bytes())
        print(f"[fetch_model] wrote {directory / 'metadata.json'}", file=sys.stderr)
    except Exception as exc:
        print(
            f"[fetch_model] WARNING: no metadata.json in {repo} ({type(exc).__name__}).",
            file=sys.stderr,
        )
        print(
            "[fetch_model] WARNING: the serving contract is MISSING, so the "
            "decision threshold falls back to config and predictions will be "
            "labelled against the wrong operating point.",
            file=sys.stderr,
        )


def main() -> int:
    repo = os.environ.get(REPO_ENV, "").strip()
    if not repo:
        print(
            f"[fetch_model] {REPO_ENV} is not set — no artifact to download.\n"
            f"[fetch_model] The API will start and report itself UNREADY, which is\n"
            f"[fetch_model] the correct behaviour for a service with no model.",
            file=sys.stderr,
        )
        return 0

    filename = os.environ.get(FILE_ENV, DEFAULT_FILE).strip() or DEFAULT_FILE
    token = os.environ.get(TOKEN_ENV) or None

    try:
        from huggingface_hub import hf_hub_download

        print(f"[fetch_model] downloading {repo}/{filename}", file=sys.stderr)
        downloaded = hf_hub_download(repo_id=repo, filename=filename, token=token)

        destination = target_path()
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Copy rather than symlink: the hub cache is a separate layer on the
        # host's ephemeral disk, and a dangling symlink fails at load time
        # rather than here, where the error would still be readable.
        destination.write_bytes(Path(downloaded).read_bytes())

        size_kb = round(destination.stat().st_size / 1e3)
        print(f"[fetch_model] wrote {destination} ({size_kb} kB)", file=sys.stderr)

        _fetch_contract(repo, destination.parent, token)
        return 0
    except Exception as exc:
        # Deliberately swallowed. See the module docstring: an unready service
        # that explains itself beats a crash loop that does not.
        print(
            f"[fetch_model] FAILED: {type(exc).__name__}: {exc}\n"
            f"[fetch_model] Starting anyway; /health/ready will report 503.",
            file=sys.stderr,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
