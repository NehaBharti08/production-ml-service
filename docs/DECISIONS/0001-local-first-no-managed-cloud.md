# ADR 0001 — Local-first infrastructure, no managed cloud

- **Status:** Accepted
- **Date:** 2026-08-09
- **Phase:** 0

## Context

This project must demonstrate production ML operations: container
orchestration, metrics, dashboards, alerting, and a publicly reachable
endpoint. The obvious route is a managed cloud — EKS/GKE, managed Prometheus, a
hosted model endpoint.

The constraints are a student budget and a 16 GB laptop, and the audience is a
technical reviewer assessing operational competence.

## Decision

Run the entire stack locally on Docker Compose and `kind`, with a single free-
tier public endpoint on Hugging Face Spaces. No managed cloud services.

## Reasoning

**Free-tier credits expire; bills surprise students.** A portfolio project whose
live demo dies when a trial lapses is worse than one that never had a live demo,
because the dead link is visible on the README.

**Local Kubernetes demonstrates the same competence.** The manifests, probes,
rollout strategy and `kubectl rollout undo` path are identical on `kind` and on
a managed cluster. What a managed cluster additionally demonstrates is the
ability to click through a console, which is not the skill under assessment.

**Cost predictability is itself an operational argument.** "I chose the
architecture that costs nothing to keep running" is a defensible engineering
answer, not an apology.

## Public endpoint: Hugging Face Spaces

Evaluated three options:

| Option | Cost | Cold start | Card required |
|:--|:--|:--|:--|
| **HF Spaces** (`cpu-basic`) | Free | Sleeps after **48 h** idle | No |
| Fly.io | ~$0 with scale-to-zero | 1–3 s | **Yes** |
| Render free tier | Free | Sleeps after **15 min**, 30–60 s wake | No |

Chose **Hugging Face Spaces**: no credit card, 2 vCPU / 16 GB, and a 48-hour
idle window rather than Render's 15 minutes — the difference between a reviewer
clicking the README link and getting a response versus a 60-second spinner.
It is also ML-recruiter-native.

Fly.io is retained as a documented fallback and `Dockerfile.hf` is written to be
portable to it.

## Consequences

**The Space is a demo endpoint, not the monitoring substrate.** Space disk is
ephemeral, so prediction logs written there do not survive a restart. The real
monitoring loop runs locally against the compose stack. The README must say
this plainly rather than implying the public endpoint is a monitored production
service.

**Model binaries go to a Hugging Face Model repo, not git.** The Space pulls the
artifact at startup. This keeps binaries out of version control and happens to
be a genuinely good registry → artifact store → runtime pattern.

**Profiles are mutually exclusive.** The full observability stack (~3.5 GB) and
a `kind` cluster (~4 GB) cannot both run on a 16 GB machine without swapping.
`make k8s-demo` runs `docker compose down` first, deliberately.

**The Kubernetes demo is scoped to Deployment, Service, probes, HPA and a
rollout/rollback demonstration** — not a full in-cluster Prometheus/Grafana/
MLflow stack, which would need 7–9 GB. The rollback demo is the portfolio
signal; running Prometheus twice is not.

## Revisit if

An employer-provided cloud account becomes available, or the Space's free tier
changes terms. Neither changes the manifests — only where they are applied.
