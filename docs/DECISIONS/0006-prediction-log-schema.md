# ADR 0006 — Prediction log schema and durability

- **Status:** Accepted
- **Date:** 2026-08-17
- **Phase:** 3

## Context

Phases 5, 6 and 7 all read the prediction log: latency percentiles and score
distributions, drift detection against a frozen reference, delayed-label
performance, and the retraining trigger. It is the substrate for everything that
makes this project an operations demonstration rather than a model demo.

A schema that has to be migrated later is worse than one designed slowly now,
because the records written before the migration become unusable — and those are
the earliest production records, which make the best monitoring baseline.

## Decision

Newline-delimited JSON, append-only, one record per prediction, with nullable
outcome columns present from the first write.

## Reasoning

**NDJSON over Parquet.** Parquet compresses far better and reads faster, but it
cannot be appended one line at a time. A monitoring substrate that can lose its
last buffer on a crash is not one you can reason about during an incident.
NDJSON is also greppable at 3am, which is worth more than disk.

**Nullable outcome columns from day one.** A 30-day readmission label cannot
exist at prediction time. Adding the columns later would make every earlier
record unjoinable.

**Raw features, not transformed ones.** Drift must be measured in a space a human
can reason about (`age="[70-80)"`), not in one-hot columns whose meaning depends
on an encoder version. It also lets a stored record be replayed through a
*different* model, which is what the Phase 7 canary comparison needs.

**`feature_schema_hash` per record.** Drift analysis across a schema change is
meaningless — the distributions are not comparable. Without the hash the change
is invisible and the drift report silently lies.

**`model_source` per record**, not only at startup. During an incident the first
question is "which model produced this score", and `registry` versus
`local_fallback` are very different answers. A startup log line from an hour ago
does not answer it for a specific prediction.

**Outcomes are appended, never written back.** Mutating a prediction record would
mean rewriting the file — risking the whole log on a crash and erasing the fact
that the label arrived later, which Phase 6 needs to reason about maturation lag.
The join happens at read time on `prediction_id`.

## Durability: fsync per record, except in batch

Single predictions fsync per record. A container kill would otherwise lose the
tail sitting in the OS buffer, and the records around a crash are precisely the
ones worth having.

**Batch writes fsync once**, and that came from measurement rather than
principle. A 200-item batch spent roughly **400 ms of its ~490 ms in 200
separate fsync calls**, while the model transform cost ~85 ms total. Batching the
write took the same request to **47.7 ms** — a 10× improvement, with per-item
cost (0.24 ms) then matching the raw model benchmark (0.214 ms) exactly, which
confirmed fsync had been the entire gap.

The crash semantics change in a way that does not matter. Per-record fsync
guarantees "every record before the crash is durable"; per-batch fsync guarantees
"every record before the last batch is durable, and the final batch may be
truncated". `read_records` already tolerates a truncated tail — it is the
expected state after an unclean shutdown — so this trades a bounded,
already-handled loss for a large latency win.

## Consequences

**The log grows without bound.** No rotation is implemented. At the demo's
traffic this is irrelevant; a real deployment would need rotation, and the
Phase 8 runbook should say so rather than implying the design is complete.

**Writes are swallowed on failure.** A monitoring write must never turn a served
prediction into a 500: losing a log line is a monitoring gap, failing the request
is an outage. Failures are counted in `prediction_log_writes_total{result="failed"}`
so the gap is visible.

**No PHI in application logs.** Feature values go to the prediction log, which is
a data store; the aggregated application log gets IDs, scores and latencies only.
An aggregated log shipper is the wrong place for a patient record.

## Revisit if

Throughput rises enough that per-record fsync on single predictions becomes the
bottleneck, in which case a bounded write-behind buffer with an explicit
durability window would be the next step — stated as a window, not left implicit.
