# 3. Module architecture and dependency rules

- Status: Accepted
- Date: 2026-11-21

## Context
Systems (rendering, physics, editor, ...) must be developed and tested independently without breaking the
rest of the engine, and owned by separate teams.

## Decision
Each module lives in `engine/<module>/` and is a static library `Nova::<module>`:

```
engine/<module>/
  include/nova/<module>/   public contract: the only headers other modules may include
  src/                     private implementation (never included from outside)
  CMakeLists.txt           nova_add_module(<module> SOURCES ... PUBLIC_DEPS ... PRIVATE_DEPS ...)
```

Rules:
1. **Layering is one-way.** A module may depend only on modules in lower layers. No cycles.

   | Layer | Modules |
   |-------|---------|
   | 0 | core (types, log, assert, time) |
   | 1 | memory, jobs, platform, math |
   | 2 | rhi; ecs, assets, serialization *(planned)* |
   | 3 | renderer; scene, render-graph, physics, animation, audio *(planned)* |
   | 4 | runtime, editor *(planned)* |

2. **Small public surfaces.** Prefer opaque types (pimpl) or interfaces at module boundaries so
   implementations can change without touching dependents (e.g. `JobSystem`, `Window`).
3. **Third-party types do not leak** into public headers. SDL3, Jolt, Vulkan, etc. are private dependencies
   behind engine interfaces (see 0004, 0005).
4. **No direct access to another module's data.** Communication goes through the public API; resources go
   through the Asset/Resource manager once it exists.
5. **Editor and Runtime are separate modules.** They may share a process initially, but the Runtime never
   depends on the Editor, so it can later run out-of-process (IPC) and ship without editor code.
6. Every module has unit tests in `tests/` and builds with warnings-as-errors.

## Consequences
- Changing a header under `include/nova/` is a contract change and needs review (0001).
- A module can be swapped or stubbed (e.g. the Headless window backend) for tests and servers.
