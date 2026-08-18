# Hospital Readmission Risk — A Production ML Service

> ## ⚠️ NOT FOR CLINICAL USE
>
> This is an **engineering demonstration of ML operations**, not a medical
> device and not a clinical decision support tool. It is trained on a public
> 1999–2008 hospital research dataset, has never been clinically validated, and
> must never be used to inform patient care. Predictions it returns are
> illustrative of a monitoring and deployment pipeline — nothing more.
>
> This disclaimer appears in the API responses, the model card, and the UI as
> well as here. That repetition is deliberate.

---

## What this project is

Most ML portfolios prove someone can *train* a model. This one exists to prove
something scarcer: that the model can be **operated**.

So the priority is inverted on purpose. The model is deliberately boring — a
regularised logistic regression — and the operations are the product:
monitoring, drift detection, calibration, retraining with promotion gates,
canary rollout, tested rollback, and an incident runbook.

A well-operated logistic regression is a better artifact than an unmonitored
gradient-boosted ensemble. Any effort that would go into squeezing out accuracy
points goes into the observability and retraining layers instead.

### Why hospital readmission, and why this dataset

The widely-circulated symptom-to-disease datasets are synthetically generated
and trivially separable — a decision tree reaches near-100% accuracy on them,
which is a red flag rather than a result.

This project uses **Diabetes 130-US Hospitals (UCI #296)**: 101,766 real
inpatient encounters across 130 hospitals, 1999–2008. Real missingness, real
class imbalance (~11% positive), real fairness dimensions, and a published
performance ceiling around ROC-AUC 0.65–0.68.

That modest ceiling is a **feature, not a limitation.** A model that cannot be
made excellent is a model that must be carefully operated, which is the entire
point.

### What the audit found

The [data audit](docs/DATA_AUDIT.md) is generated from the code, not written by
hand. Four findings changed the pipeline:

- **A deterministic label leak.** 1,652 expired-discharge encounters contain
  **exactly zero** positive labels — a dead patient cannot be readmitted, so a
  model would learn `died → not readmitted` from the discharge code. Removed.
  Hospice discharges are excluded too, but for a *different* reason: they are
  not deterministic (5.6% positive), so calling them leakage would overstate it.
- **The time proxy holds — verified, not assumed.** This dataset has no
  timestamp column at all. 5 of 8 independent signals shift monotonically across
  `encounter_id`, and rosiglitazone prescribing shows a sharp
  **7.7× discontinuity at the 80th percentile** — the 2007 Avandia safety
  collapse. The ordering reproduces a *dated real-world event*, which is far
  stronger evidence than a trend test. See
  [ADR 0004](docs/DECISIONS/0004-temporal-split-proxy.md).
- **Right-censoring at the tail.** A first encounter can only be labelled
  positive if a later encounter exists in the data. The positive rate collapses
  56% in the final 5% of the ordering — labels are *missing, not negative*. A
  censoring buffer now discards that region before splitting.
- **`payer_code` is time-confounded, not merely missing.** Capture goes from
  100% missing to 14% across the period. Under a chronological split the model
  would learn it as a proxy for *era*. Dropped.

The dataset is not trivially separable: an unconstrained decision tree reaches
**0.519 test ROC-AUC**. On the synthetic symptom-to-disease datasets the same
tree reaches ~100%, which is precisely why those datasets prove nothing.

---

## Status

🚧 **Phase 5 of 8 complete.** This README fills in as the phases land.

| Phase | Scope | State |
|:--|:--|:--|
| 0 | Foundation — config, logging, tooling | ✅ done |
| 1 | Data audit and honest baseline | ✅ done |
| 2 | Model, calibration, subgroups, model card | ✅ done |
| 3 | FastAPI serving + prediction log | ✅ done |
| 4 | Tests, behaviour suite, CI, load test | ✅ done |
| 5 | Prometheus + Grafana observability | ✅ done |
| 6 | Drift detection and monitoring | ⬜ |
| 7 | Retraining, promotion gates, rollback | ⬜ |
| 8 | Ship — runbook, live endpoint, docs | ⬜ |

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv fetches it).

```bash
uv sync                      # create the venv and install
uv run mlservice doctor      # verify the environment is fit to run
uv run mlservice config      # show fully resolved configuration
```

Docker, `kind` and `kubectl` are needed from Phase 3 onward — see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

---

## Documentation

| Document | What it covers |
|:--|:--|
| [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md) | Leakage hunt, missingness, imbalance, split verification |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use, **out-of-scope use**, metrics, calibration, subgroups |
| [docs/MONITORING.md](docs/MONITORING.md) | Every metric and threshold, with its derivation |
| [docs/LOAD_TEST_REPORT.md](docs/LOAD_TEST_REPORT.md) | Measured latency profile and where the service breaks |
| [docs/RETRAINING_POLICY.md](docs/RETRAINING_POLICY.md) | Triggers, promotion gates, rollback |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Incident response — what to do when it breaks |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagram and component responsibilities |
| [docs/DECISIONS/](docs/DECISIONS/) | Architecture decision records |

---

## Responsible ML commitments

These are load-bearing requirements of the project, enforced in code rather
than promised in prose:

- **Calibration is a deployment gate, not a report.** A model that ranks better
  but calibrates worse is *blocked* from production. For anything
  health-adjacent, a probability you cannot trust cannot support a decision —
  see the calibration gate in [`configs/thresholds.yaml`](configs/thresholds.yaml).
- **Subgroup performance is reported openly**, across race, gender and age
  bands, including where the results are unflattering.
- **Drift is labelled honestly.** Where drift is deliberately induced to
  demonstrate detection, the documentation says so plainly rather than implying
  a claim the data cannot support.
- **Thresholds are derived, not chosen.** Every operational number carries a
  provenance tag (`MEASURED`, `DERIVED`, `STANDARD`, `PLACEHOLDER`) and a
  written justification.
- **No patient-level data is ever committed.** See [data/README.md](data/README.md).

---

## Data handling

The UCI dataset is public and de-identified, but this repository still never
commits row-level records, model binaries, or MLflow artifact stores. Data is
fetched by script and verified against `data/checksums.txt`, which keeps the
pipeline reproducible without putting clinical records in git history.

---

## Licence

[MIT](LICENSE). The dataset carries its own terms — see [data/README.md](data/README.md).
