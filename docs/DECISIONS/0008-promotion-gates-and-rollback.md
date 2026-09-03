# ADR 0008 — Promotion gates run all-or-nothing, and the rollback path is exercised

- **Status:** Accepted
- **Date:** 2026-08-29
- **Phase:** 7

## Context

Phase 7 needed a mechanism that decides whether a freshly trained challenger
replaces the serving model. The obvious implementations are all subtly wrong in
ways that only show up once something bad ships.

Three design questions had to be answered, and each has a tempting wrong answer:

1. **Should gate evaluation short-circuit on the first failure?** It is cheaper,
   and it is what most validation code does.
2. **Should promotion be forceable?** Every deployment tool grows a `--force`
   flag, usually within a week of the first incident.
3. **What counts as having "tested" rollback?** The default answer is that the
   code exists and has a unit test.

The third question was answered wrongly here first, which is why this ADR exists
rather than being a paragraph in the policy document.

## Decision

### 1. Every gate runs, even after one fails

`evaluate_promotion()` evaluates all six gates and returns every result.

Short-circuiting optimises for the wrong thing. Gate evaluation costs
milliseconds against metrics already computed; the expensive resource is the
operator's attention at the moment a promotion is blocked. "Blocked by
calibration" sends them to investigate calibration. "Blocked by calibration,
subgroup and data quality" tells them the training data is broken — a completely
different diagnosis reached in one step instead of three fix-and-rerun cycles.

The failing gate is named in the log *with the numbers that produced it*. At 3am,
"the calibration gate failed" is not actionable; the challenger's ECE, the
incumbent's, and the limit are.

### 2. There is no promotion bypass

`promote()` raises `ValueError` when `gates_passed` is False, and the CLI takes
the gate decision as a **required argument** rather than an optional flag.

A promote function with a bypass is a promote function with no gates. The flag
would be reached for at precisely the moment judgment is worst: during an
incident, under time pressure, by someone who is confident this particular case
is the exception. If the gates are wrong, the fix is to change the gates in a
reviewed commit — which leaves a record, and makes the next person's decision
easier rather than harder.

The escape hatch already exists and points the other way: **rollback is
ungated.** Getting a model out is always available; getting one in is not.

### 3. Rollback is verified by a cycle that must move the alias

`verify_rollback_path()` seeds the alias to the oldest version, promotes the
latest, rolls back, and asserts the alias both **moved** and **returned** —
failing explicitly on a degenerate cycle. The original target is restored on exit.

This is the decision that came from getting it wrong.

The first implementation promoted whichever version `search_model_versions`
returned first. That happened to be the version already serving, so it recorded:

```
promote  version 2 -> 2   alias now points at: 2
rollback version 2 -> 2   alias now points at: 2
VERIFIED: True
```

It passed. It proved nothing. `from_version == to_version` on both operations
means the alias never moved, so the rollback path was never exercised — while the
output said, in capital letters, that it was.

A verification that cannot fail is decoration, and decoration is worse than
nothing here because it manufactures confidence. The claim "rollback is tested"
in the README has to be backed by something that would have caught a broken
rollback, and that version would not have.

The fix is not more assertions but a differently *constructed* cycle: force a
real two-version transition, then assert the transition happened. The
corresponding unit test, `test_the_alias_genuinely_moved_in_between`, pins the
midpoint rather than only the endpoints — because ending where you started is
satisfied trivially by never leaving.

A second bug surfaced immediately after: `search_model_versions` returns
`.version` as an **int** while `get_model_version_by_alias` returns a **str**, so
`"2" == 2` was False and a working cycle reported
`promotion did not take effect: alias is 2, expected 2`. Normalising at the
boundary fixed it. Worth recording because the failure message was
self-contradictory on its face — the kind of output that means a type is wrong,
not a value.

## Consequences

**Good**

- A blocked promotion produces a complete diagnosis, not the first symptom.
- There is no path to production that skips the gates, so "how did this ship?"
  always has an answer in `promotion_history.ndjson`.
- The rollback claim in the README is backed by an execution that would fail if
  rollback broke. `verify-rollback` runs in CI.
- The audit trail records `from_version` before the flip, so rollback is a lookup
  rather than reconstruction from run history during an incident.

**Costs and limits**

- Running all gates means computing all gate inputs — including subgroup metrics
  and the behaviour suite — even when the first gate already failed. Acceptable
  at this scale; would need revisiting if a gate became expensive.
- No bypass means a genuinely-wrong gate blocks a genuinely-good model until
  someone edits and reviews the threshold. That is the intended trade: slower in
  the rare case, safe in the common one.
- `verify_rollback_path` writes real entries to the promotion history, including
  a synthetic promotion tagged `trigger=rollback_verification`. Filterable, but
  the audit trail is not purely organic — better than a verification that touches
  nothing real.
- **The container-level rollback (`kubectl rollout undo`) remains unverified.**
  Docker is not installed on this machine. The model-level lever is genuinely
  tested; the image-level one is written and not run, and both this ADR and the
  policy say so rather than implying full coverage.

## Reasoning — alternatives rejected

**Weighted gates with a composite score.** Rejected. It permits a model to fail
calibration badly and pass on the strength of discrimination — which is exactly
the failure [ADR 0002](0002-calibration-as-deployment-gate.md) exists to prevent.
Gates are boolean because a veto is not a vote.

**Warn-only gates on first release, hardening later.** Rejected. A gate that only
warns is a log line, and the point at which everyone agrees to harden it never
arrives. All six gates block. `GateResult` carries a `blocking` field so a future
advisory gate can be added deliberately and be visible as such — today nothing
uses it, and no gate is advisory.

**Trusting the unit tests as rollback verification.** Rejected — this is the
substance of decision 3. The unit tests use a fake registry, which is right for
testing the logic and useless for the question actually being asked: does the
rollback path work against the real registry, with its real types and its real
ordering guarantees? Both bugs above lived exactly in that gap, and no amount of
mocking would have surfaced either.

## Revisit if

- **A gate becomes expensive enough that running all six wastes real time.** The
  all-or-nothing evaluation assumes gate inputs are cheap. If a behaviour suite
  grows to minutes, revisit ordering — but keep reporting every result that was
  computed.
- **Blocked promotions become routine rather than exceptional.** That means a
  threshold is wrong, not that the gates are. Fix the threshold in a reviewed
  commit; do not add a bypass.
- **Docker becomes available on the development machine.** Then the container-level
  rollback should be exercised the same way the registry-level one is, and the
  "not verified" note above should be replaced with output — not deleted.
- **The registry gains more than a handful of versions.** `verify_rollback_path`
  assumes oldest and latest are both loadable; with a pruned registry, seed from
  two known-good versions instead.

## Related

- [ADR 0002 — Calibration as a deployment gate](0002-calibration-as-deployment-gate.md)
- [ADR 0007 — Drift thresholds from an empirical null](0007-drift-thresholds.md)
- [RETRAINING_POLICY.md](../RETRAINING_POLICY.md)
