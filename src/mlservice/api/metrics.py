"""Prometheus metrics.

Defined in Phase 3 rather than Phase 5 because the instrumentation points are
inside the request path: adding them later means editing every route again. The
dashboards and alert rules that consume these arrive in Phase 5.

Two decisions that determine whether these metrics are usable at all:

**Latency buckets are chosen for this service, not copied.** The default
Prometheus buckets top out coarsely around 10s, which is useless for an endpoint
whose p99 is expected in the tens of milliseconds — every real observation would
land in the first bucket and the histogram would be unable to resolve p95 from
p99. These buckets are dense between 1 ms and 250 ms, which is where the
distribution actually lives.

**Endpoint labels use the route template, never the raw path.** A label whose
cardinality grows with traffic (an ID in the path) will eventually exhaust
Prometheus's memory. There is no such path parameter today, but the habit is
cheap and the failure is expensive.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

#: A dedicated registry rather than the global default. It keeps the exposition
#: free of unrelated process collectors from libraries, and lets tests build a
#: clean registry instead of fighting module-global state.
REGISTRY = CollectorRegistry()

# --------------------------------------------------------------------------- #
# Request-level: the golden signals
# --------------------------------------------------------------------------- #

http_requests_total = Counter(
    "http_requests_total",
    "HTTP requests by method, endpoint and status class.",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

#: Dense where the distribution lives. p99 is expected in the 25-40 ms range for
#: a logistic regression behind FastAPI, so resolution below 250 ms is what
#: matters; the 1s and 5s buckets exist only to catch pathology.
LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.250,
    0.500,
    1.0,
    5.0,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "End-to-end request latency, including validation and logging.",
    ["method", "endpoint"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

# --------------------------------------------------------------------------- #
# Model-level
# --------------------------------------------------------------------------- #

#: Separate from request latency so a regression can be attributed. If total
#: latency rises while this stays flat, the cause is the API layer or the log
#: write — not the model.
model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Time spent inside predict_proba only.",
    ["model_version"],
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.010, 0.025, 0.050, 0.100),
    registry=REGISTRY,
)

model_predictions_total = Counter(
    "model_predictions_total",
    "Predictions by model version and emitted label.",
    ["model_version", "predicted_label"],
    registry=REGISTRY,
)

#: Buckets are uniform over [0,1] because this is a *distribution* to watch for
#: shift, not a latency to percentile. The score histogram moves before accuracy
#: does — and long before labels mature — which makes it the earliest available
#: signal that the input population has changed.
model_predicted_proba = Histogram(
    "model_predicted_proba",
    "Distribution of predicted probabilities.",
    ["model_version"],
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

#: 1 = a model is loaded and passed its canary inference. This is what readiness
#: reports, so it is the gauge Kubernetes effectively gates traffic on.
model_load_status = Gauge(
    "model_load_status",
    "1 if a model is loaded and serving, else 0.",
    registry=REGISTRY,
)

model_info = Info(
    "model_serving",
    "Identity of the model currently serving.",
    registry=REGISTRY,
)

# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #

#: Labelled by field, which is the point. A spike concentrated on one field is
#: an upstream schema change; a spread across many is a caller sending junk.
#: Those need different responses, and an unlabelled counter cannot distinguish
#: them.
validation_errors_total = Counter(
    "validation_errors_total",
    "Request validation failures by field.",
    ["field"],
    registry=REGISTRY,
)

prediction_log_writes_total = Counter(
    "prediction_log_writes_total",
    "Prediction log write attempts by outcome.",
    ["result"],  # "ok" | "failed"
    registry=REGISTRY,
)

#: Watchdog for the most common silent monitoring failure: labels stop arriving,
#: nothing alerts, because nothing is being measured.
outcomes_recorded_total = Counter(
    "outcomes_recorded_total",
    "Observed outcomes submitted for prediction IDs.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
# Recording helpers — keep instrumentation out of the route bodies
# --------------------------------------------------------------------------- #


def record_request(method: str, endpoint: str, status_code: int, duration_s: float) -> None:
    # Status *class* rather than the exact code: dashboards and the error-rate
    # SLO care about 5xx vs 4xx, and per-code series multiply the label
    # cardinality for no operational gain.
    http_requests_total.labels(
        method=method, endpoint=endpoint, status=f"{status_code // 100}xx"
    ).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration_s)


def record_prediction(
    model_version: str, probability: float, label: int, inference_s: float
) -> None:
    model_inference_duration_seconds.labels(model_version=model_version).observe(inference_s)
    model_predictions_total.labels(model_version=model_version, predicted_label=str(label)).inc()
    model_predicted_proba.labels(model_version=model_version).observe(probability)


def record_validation_error(field: str) -> None:
    validation_errors_total.labels(field=field).inc()


def record_log_write(ok: bool) -> None:
    prediction_log_writes_total.labels(result="ok" if ok else "failed").inc()


def set_model_loaded(loaded: bool, **info: str) -> None:
    model_load_status.set(1 if loaded else 0)
    if loaded and info:
        model_info.info(info)


__all__ = [
    "LATENCY_BUCKETS",
    "REGISTRY",
    "outcomes_recorded_total",
    "record_log_write",
    "record_prediction",
    "record_request",
    "record_validation_error",
    "set_model_loaded",
]
