# Development setup

## Prerequisites

| Tool | Needed from | Why |
|:--|:--|:--|
| [uv](https://docs.astral.sh/uv/) | Phase 0 | Dependency and Python management; fetches Python 3.11 itself |
| Git | Phase 0 | — |
| Docker Desktop | Phase 3 | The compose stack, and the base for `kind` |
| kind + kubectl | Phase 7 | Local Kubernetes demo |
| k6 | Phase 4 | Load testing |

Only `uv` and Git are needed to start. The rest can wait until the phase that
uses them.

## Quick start

```bash
uv sync --all-groups            # venv + every dependency group
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type commit-msg
uv run mlservice doctor         # confirms the environment is fit to run
```

On Windows without `make`, use the shim — same target names:

```powershell
.\make.ps1 setup
.\make.ps1 check
```

## Installing the container toolchain (needed from Phase 3)

These require administrator rights and a reboot, so they are not part of any
scripted setup step.

```powershell
# 1. WSL2 distro (the kernel is usually already enabled; the distro is not)
wsl --install -d Ubuntu

# 2. Docker Desktop  (~5 GB on C:)
winget install --id Docker.DockerDesktop -e

# 3. Kubernetes tooling (Phase 7)
winget install --id Kubernetes.kind -e
winget install --id Kubernetes.kubectl -e

# 4. Load testing (Phase 4)
winget install --id k6.k6 -e
```

Reboot after Docker Desktop, then enable **Settings → Resources → WSL
Integration** for your distro. Verify:

```powershell
docker run --rm hello-world
```

### Point Docker's storage at a drive with room

Docker Desktop defaults to `C:`. If that drive is tight, move the disk image
via **Settings → Resources → Advanced → Disk image location**. Images, volumes
and `kind` clusters together run to tens of GB.

## Resource profiles

The stack is split into profiles that are **mutually exclusive by design**. On
a 16 GB machine, the full observability stack and a `kind` cluster together
will swap.

| Profile | Command | Approx. RAM |
|:--|:--|:--|
| Core | `make serve` | ~1.5 GB |
| Full | `make full` | ~3.5 GB |
| Kubernetes | `make k8s-demo` | ~4 GB (stops compose first) |

`make k8s-demo` runs `docker compose down` before starting, deliberately.

## Configuration

Resolution order, lowest precedence first:

1. Field defaults in `src/mlservice/config.py`
2. `configs/base.yaml`
3. `configs/{MLSERVICE_ENV}.yaml` — `local`, `hf` or `test`
4. Environment variables — `MLSERVICE_` prefix, `__` for nesting
5. `.env`

So `api.port` is overridden by `MLSERVICE_API__PORT=9000`.

```bash
uv run mlservice config --thresholds    # what the process actually resolved
```

### A typo asymmetry worth knowing

Unknown keys are rejected — in YAML, and in *nested* environment variables:

```bash
MLSERVICE_API__PROT=9999 uv run mlservice doctor
#   FAIL  configuration failed to resolve: api.prot | Extra inputs are not permitted
```

But a **root-level** env typo maps to no field at all, so pydantic-settings
never reads it and validation never sees it. The override silently does nothing.
`doctor` catches that case separately:

```bash
MLSERVICE_DEBUGG=true uv run mlservice doctor
#   note  unrecognised env var, override will NOT apply: MLSERVICE_DEBUGG
```

Run `doctor` after editing `.env`.

## Operational thresholds

SLOs, drift thresholds and promotion gates live in `configs/thresholds.yaml`,
separate from application settings because Phase 6 *regenerates* them from data.

Every value carries a provenance tag:

| Tag | Meaning |
|:--|:--|
| `MEASURED` | Observed in this repo; `source` names where |
| `DERIVED` | Computed from a `MEASURED` value by a stated rule |
| `STANDARD` | External citable convention; `source` names it |
| `PLACEHOLDER` | Not yet real. **Must not be used for alerting.** |

`mlservice doctor` reports how many `PLACEHOLDER` blocks remain. None may
survive into the `v1.0.0` tag.

## Git workflow

One branch per phase, Conventional Commits, PR per phase, tag on merge. Direct
commits to `main` are blocked by a pre-commit hook.

```bash
git switch -c phase-N-name
git commit -m "feat(scope): what changed"
```

## Before every commit

```bash
make check      # lint + typecheck + tests
```

Hooks enforce the rest. Two of them are not style tools and must not be
disabled: `nbstripout` (a notebook output cell can contain patient rows) and
`detect-secrets`.
