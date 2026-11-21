# 1. Record architecture decisions

- Status: Accepted
- Date: 2026-11-21

## Context
NOVA is built by several teams (human and agent) working in parallel on separate modules. Decisions about
shared contracts — module boundaries, public APIs, dependencies, toolchain — must be explicit, reviewable,
and discoverable later, or parallel work drifts apart.

## Decision
Every architecturally significant decision is captured as an ADR in `docs/adr/NNNN-title.md` using the
template below. ADRs are immutable once accepted; a change of direction is a new ADR that supersedes the old
one. Changes to a shared contract (any `include/nova/<module>/` header or a cross-module rule) require an ADR
or a reference to an existing one in the pull request, plus Architect review.

```
# N. Title
- Status: Proposed | Accepted | Superseded by NNNN
- Date: YYYY-MM-DD
## Context
## Decision
## Consequences
```

## Consequences
- Reviewers can reject contract changes that lack a decision record.
- "Proposed" ADRs are allowed for choices that still depend on a prototype or benchmark (e.g. 0005).
