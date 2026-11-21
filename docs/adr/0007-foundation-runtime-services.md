# 7. Foundation runtime services: logging, memory, jobs

- Status: Accepted
- Date: 2026-11-21

## Context
Every later system needs logging, assertions, timing, predictable allocation and parallel execution. These
must be simple and correct first, and measurable so they can be optimized with evidence later.

## Decision
- **Logging** (`nova/core/log.hpp`): levels Trace…Fatal, channel names, `std::format` messages through
  `NOVA_INFO("Channel", "fmt {}", x)`-style macros. Formatting is skipped when the level is disabled. Sinks are
  pluggable callbacks serialized by a mutex; a console sink is installed by default. Sinks must not log.
- **Assertions** (`nova/core/assert.hpp`): `NOVA_ASSERT` (Debug only) and `NOVA_VERIFY` (always). Failures
  go to a replaceable handler (default: log Fatal and abort; tests install a throwing handler).
- **Timing** (`nova/core/time.hpp`): `Stopwatch` on `steady_clock`, and `FrameStats`, a rolling window that
  reports avg/min/max, p99 frame time and 1% low FPS — the metrics the performance plan tracks.
- **Memory** (`nova/memory/allocators.hpp`): `LinearAllocator` (per-frame/scratch arenas with markers) and
  `PoolAllocator` (fixed-size blocks, O(1), intrusive free list). Both report usage and peak. Neither is
  thread-safe; use one per thread.
- **Jobs** (`nova/jobs/job_system.hpp`): fixed worker pool (hardware threads − 1), `WaitGroup` completion
  counters, `parallel_for`. Waiting threads execute queued jobs, so nested submit/wait cannot deadlock even
  with one worker.

## Known limitations (deliberate, revisit with profiling data)
- The job system uses one mutex-protected queue. Replace with per-worker work-stealing deques when Tracy
  shows contention in real workloads.
- Jobs are `std::function` (may heap-allocate for large captures). Move to a fixed-size job type if
  allocation shows up in profiles.
- No global allocation tracking yet; to be added together with Tracy memory zones.

## Consequences
- Foundation code is covered by unit tests and runs clean under ThreadSanitizer and
  AddressSanitizer/UBSan; CI keeps the sanitizer jobs.
