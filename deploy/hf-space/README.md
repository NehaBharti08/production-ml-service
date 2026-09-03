---
title: Hospital Readmission Risk — Production ML Service
emoji: 🏥
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Hospital Readmission Risk — live demo

> ## ⚠️ NOT FOR CLINICAL USE
>
> This is an **engineering demonstration of ML operations**, not a medical
> device and not a clinical decision support tool. It is trained on a public
> 1999–2008 hospital research dataset, has never been clinically validated, and
> must never be used to inform patient care.

**Source:** https://github.com/NehaBharti08/production-ml-service

---

## What this Space is, and what it is not

**It is** the same FastAPI service the repository builds, running the same
model, so the API contract shown here is the real one.

**It is not** the monitoring substrate. A Space has an ephemeral filesystem: the
prediction log written here is lost on restart, and there is no Prometheus or
Grafana attached. The real observability loop runs locally on compose. That is a
property of the free tier, not an oversight, and pretending otherwise would
misrepresent what a reviewer is looking at.

## Endpoints

| Endpoint | Purpose |
|:--|:--|
| `/` | Static UI with a non-dismissable disclaimer |
| `POST /v1/predict` | Single prediction; returns a `prediction_id` |
| `POST /v1/predict/batch` | Batch prediction |
| `GET /v1/model` | Model version, source, and the disclaimer |
| `GET /health/live` | Process is alive |
| `GET /health/ready` | Model is loaded and passed a canary inference |
| `GET /docs` | OpenAPI |

## Configuration

Set in **Settings → Variables and secrets**:

| Name | Kind | Required | Purpose |
|:--|:--|:--|:--|
| `HF_MODEL_REPO` | variable | yes | Model repo holding `model.joblib` |
| `HF_MODEL_FILE` | variable | no | Defaults to `model.joblib` |
| `HF_TOKEN` | **secret** | only if the model repo is private | Read token |

The model binary is **not** in the git repository — it is gitignored there and
always will be. It lives in a separate HF Model repo which this Space downloads
at startup, which is the same artifact-store/runtime split the MLflow path uses.

## If the Space reports 503

That is the readiness probe doing its job, not a crash. `/health/ready` returns
503 whenever no model is loaded. Check the Space logs for lines beginning
`[fetch_model]` — the download failure is printed there with its reason.

The service deliberately starts and reports itself unready rather than crash
looping, because an unready service that explains itself is debuggable and a
restart loop is not.
