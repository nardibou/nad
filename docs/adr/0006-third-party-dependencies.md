# 6. Third-party dependencies

- Status: Accepted
- Date: 2026-11-21

## Context
Dependency sprawl, unpinned versions and unclear licenses are listed risks in the master plan.

## Decision
- Every dependency is declared in `cmake/NovaDependencies.cmake`, pinned to an exact release with a
  **SHA-256 hash**, and listed in `THIRD_PARTY.md` with its version and license.
- Prefer an installed package (`find_package(... CONFIG)`) and fall back to `FetchContent` from the pinned
  archive, so offline/vendored builds stay possible.
- Only permissive licenses (MIT, BSD, zlib, Apache-2.0) without review. Anything else needs an ADR.
- Dependencies stay private to the module that wraps them (0003, rule 3).
- Upgrades are their own pull request with the new hash and a note on what changed.

## Consequences
- Adding a dependency is a reviewable, single-file change plus a `THIRD_PARTY.md` entry.
