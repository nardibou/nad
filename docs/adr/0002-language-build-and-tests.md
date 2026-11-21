# 2. Language, build system and tests

- Status: Accepted
- Date: 2026-11-21

## Context
The engine needs predictable performance, direct control over memory and threads, and a build that works
identically on developer machines and CI, on Windows first.

## Decision
- **C++23** for engine, runtime and ECS. Use library features available in MSVC 19.38+ and GCC 13+
  (`std::format`, `std::expected`, `std::source_location`, ranges). Avoid `std::print` and modules until all
  CI compilers support them.
- **CMake (>= 3.25) + Ninja Multi-Config**, driven by `CMakePresets.json` (`windows-msvc`, `linux-gcc`,
  `headless`). No IDE-specific project files are committed.
- **CTest** is the single test entry point. Unit tests use **doctest** (header-only, fast compile);
  `doctest_discover_tests` registers each test case with CTest. End-to-end smoke tests (e.g. the headless
  sandbox) are plain CTest tests.
- Warnings: `/W4` (MSVC) and `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` (GCC/Clang). CI builds with
  `NOVA_WARNINGS_AS_ERRORS=ON`.
- Python is the tooling language (asset pipeline, test tooling). Gameplay scripting language (C# or Lua) is
  decided in a later ADR; only one will be adopted initially.

## Consequences
- Contributors need a C++23-capable compiler; Visual Studio 2022 17.8+ on Windows.
- CI runs the same presets as developers, so "works locally" and "works in CI" mean the same thing.
