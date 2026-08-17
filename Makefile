# =============================================================================
# Task runner. Canonical entrypoint for CI and for anyone cloning on
# Linux/macOS. Windows users without `make` have an equivalent shim: run
# `.\make.ps1 <target>` — it mirrors these targets one for one.
#
# `make help` lists everything.
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE       := docker compose
COMPOSE_FULL  := docker compose -f docker-compose.yml -f docker-compose.monitoring.yml

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------------

.PHONY: setup
setup:  ## Create the venv, install deps, install git hooks
	uv sync --all-groups
	uv run pre-commit install --install-hooks
	uv run pre-commit install --hook-type commit-msg
	uv run mlservice doctor

.PHONY: doctor
doctor:  ## Check this environment can run the service
	uv run mlservice doctor

.PHONY: config
config:  ## Print fully resolved configuration
	uv run mlservice config --thresholds

# --- Quality -----------------------------------------------------------------

.PHONY: lint
lint:  ## Lint and format-check (no changes written)
	uv run ruff check src tests
	uv run ruff format --check src tests

.PHONY: format
format:  ## Auto-fix lint and formatting
	uv run ruff check --fix src tests
	uv run ruff format src tests

.PHONY: typecheck
typecheck:  ## Static type check
	uv run mypy src

.PHONY: hooks
hooks:  ## Run every pre-commit hook over the whole repo
	uv run pre-commit run --all-files

# --- Tests -------------------------------------------------------------------
# Suites are separate targets because CI runs them as separate jobs and only
# some need a live container.

.PHONY: test
test: test-unit test-contract test-behavior  ## Everything that needs no container

.PHONY: test-unit
test-unit:  ## Unit tests
	MLSERVICE_ENV=test uv run pytest tests/unit -m unit

.PHONY: test-contract
test-contract:  ## API schema stability
	MLSERVICE_ENV=test uv run pytest tests/contract -m contract

.PHONY: test-behavior
test-behavior:  ## Model invariance and directional expectations
	MLSERVICE_ENV=test uv run pytest tests/behavior -m behavior

.PHONY: test-integration
test-integration:  ## Integration tests — requires a running container
	MLSERVICE_ENV=test uv run pytest tests/integration -m integration

.PHONY: test-all
test-all:  ## Every suite, with coverage
	MLSERVICE_ENV=test uv run pytest --cov --cov-report=term-missing --cov-report=xml

.PHONY: check
check: lint typecheck test  ## What CI runs on every PR

# --- Stack -------------------------------------------------------------------
# Profiles are mutually exclusive by design: `full` and `k8s` together exceed
# what a 16 GB laptop can serve without swapping. See README "Resource footprint".

.PHONY: serve
serve:  ## Core stack: api + mlflow + postgres
	$(COMPOSE) up -d --build
	@echo "api      http://localhost:8000/docs"
	@echo "mlflow   http://localhost:5000"

.PHONY: full
full:  ## Core stack + prometheus + grafana
	$(COMPOSE_FULL) up -d --build
	@echo "api        http://localhost:8000/docs"
	@echo "mlflow     http://localhost:5000"
	@echo "prometheus http://localhost:9090"
	@echo "grafana    http://localhost:3000"

.PHONY: down
down:  ## Stop everything, keep volumes
	$(COMPOSE_FULL) down

.PHONY: clean
clean:  ## Stop everything and DELETE volumes (mlflow runs, dashboards, metrics)
	$(COMPOSE_FULL) down -v

.PHONY: logs
logs:  ## Tail API logs
	$(COMPOSE) logs -f api

# --- Pipeline (filled in by the phase that implements each stage) ------------

.PHONY: data
data:  ## Phase 1 — download and verify the dataset
	uv run mlservice data download

.PHONY: audit
audit:  ## Phase 1 — reproduce the data audit
	uv run mlservice data audit

.PHONY: train
train:  ## Phase 2 — train candidates and register the champion
	uv run mlservice train run

.PHONY: test-behavior-real
test-behavior-real:  ## Behaviour suite against the REAL trained artifact
	uv run pytest tests/behavior -m behavior -rs

.PHONY: loadtest
loadtest:  ## Load test a running service (Locust; see docs/LOAD_TEST_REPORT.md)
	PYTHONUTF8=1 uv run locust -f loadtest/locustfile.py --headless 		--users 6 --spawn-rate 2 --run-time 60s 		--host http://127.0.0.1:8000 --tags ramp

.PHONY: loadtest-ramp
loadtest-ramp:  ## Find the saturation knee (sweeps concurrency)
	@for u in 1 2 5 10 20 40; do 		echo "--- $$u users ---"; 		PYTHONUTF8=1 uv run locust -f loadtest/locustfile.py --headless 			--users $$u --spawn-rate $$u --run-time 25s 			--host http://127.0.0.1:8000 --tags ramp --only-summary 2>&1 			| grep -E "Aggregated" | head -1; 	done

.PHONY: drift
drift:  ## Phase 6 — induce drift and show detection
	uv run mlservice monitor replay --induce-drift

.PHONY: k8s-demo
k8s-demo:  ## Phase 7 — kind cluster, canary, rollback (stops compose first)
	$(COMPOSE_FULL) down
	bash scripts/k8s_rollback_demo.sh
