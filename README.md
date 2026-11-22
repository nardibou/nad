# NOVA Engine

A modular, Windows-first 3D game engine targeting FPS/TPS and open-world games. The engine is built in
small, independently testable modules with documented contracts, and every performance claim is backed by a
benchmark.

**Current phase: 2 — Rendering Core** (Vulkan RHI, lit 3D scene, CPU/GPU frame timing).
Phase 1 (Foundation) is complete. See the roadmap below and the decisions in [`docs/adr/`](docs/adr).

## Building

Requirements: CMake ≥ 3.25, Ninja, a C++23 compiler (Visual Studio 2022 17.8+ or GCC 13+), and a Vulkan 1.3
driver to run (not to build). Dependencies (SDL3, Vulkan-Headers, volk, Slang compiler, doctest) are fetched
automatically at pinned versions.

```sh
# Windows (Developer PowerShell for VS 2022)
cmake --preset windows-msvc
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug

# Linux
cmake --preset linux-gcc
cmake --build --preset linux-gcc-debug
ctest --preset linux-gcc-debug

# Without SDL3 (CI, servers)
cmake --preset headless && cmake --build --preset headless-debug && ctest --preset headless-debug
```

No GPU? On Linux install `mesa-vulkan-drivers` (lavapipe) to run the GPU tests and the headless sandbox.

## Sandbox

`build/<preset>/apps/sandbox/Debug/nova_sandbox` renders a grid of lit, animated cubes and logs CPU and GPU
frame time (avg / p99) every second.

| Option | Effect |
|--------|--------|
| `--headless` | render offscreen (no window) |
| `--frames N` | stop after N frames |
| `--width W --height H` | resolution |
| `--no-vsync` | mailbox/immediate present mode |
| `--validation` | Vulkan validation layers (on by default in Debug) |
| `--screenshot out.ppm` | with `--headless`: save the last frame |

Controls: A/D or ←/→ orbit, W/S zoom, Space pause, Esc quit.

## Ray-traced lighting (NOVA Lumina RT)

On GPUs with Vulkan ray queries (RTX / RDNA2+ / Arc, and Mesa lavapipe for CI), `WorldRenderer` replaces
shadow maps and screen-space effects with hybrid ray tracing:

- sun shadows traced towards the solar disc, giving physically correct soft penumbrae that harden at contact
  and need no bias tuning or fixed coverage radius;
- ray-traced AO and one-bounce diffuse GI on the real scene triangles, so there are no screen-space haloes
  and no light leaks through thin walls;
- water reflections that are traced, reusing the on-screen pixel when it is visible;
- temporal accumulation (depth-validated, neighbourhood-clamped) and penumbra-aware / à-trous denoising.

Other GPUs keep the raster path automatically. Settings are in `WorldSettings` (`ray_tracing`,
`sun_angular_radius`, `rt_indirect`, `rt_shadow_distance`, ...). Guerrilla accepts `--no-rt` for A/B
comparisons and `--rt-view 1|2|3` to show the shadow / AO / indirect signals. `NOVA_NO_RT=1` disables it
globally. Design: [ADR 0009](docs/adr/0009-hybrid-ray-traced-lighting.md).

## Temporal anti-aliasing

`WorldSettings::taa` (on by default, both lighting paths) jitters each frame by a sub-pixel Halton(2,3)
offset and resolves it into an HDR history. The history is reprojected through the depth buffer with
closest-depth dilation, fetched with a Catmull-Rom filter, variance-clipped in YCoCg and blended with luma
weights. It replaces FXAA and the CAS sharpening pass. Against a 2× supersampled reference, edge error is
about 8% lower than FXAA's on the raster path, and flat surfaces lose the per-pixel shading noise. Guerrilla
accepts `--no-taa` for A/B comparisons.

Limitation: motion vectors come from the camera only. Objects that move on their own (the first-person
viewmodel, wind-animated foliage) rely on the variance clip, and the viewmodel skips history entirely.

## Layout

```
engine/
  core/       types, logging, assertions, timing & frame statistics
  memory/     linear (frame) and pool allocators
  jobs/       worker pool, wait groups, parallel_for
  platform/   window + input (SDL3 backend, headless backend), Vulkan surface bridge
  math/       vectors, matrices, camera transforms (header-only)
  rhi/        GPU abstraction; Vulkan 1.3 backend (dynamic rendering, sync2, timestamps)
  renderer/   baseline forward renderer, meshes, Slang shaders
apps/sandbox/ test scene with frame-time reporting
tests/        doctest unit tests + GPU tests (offscreen rendering with pixel checks), via CTest
docs/adr/     architecture decision records
cmake/        build helpers, shader compilation, pinned dependencies
```

Module rules (public headers only under `include/nova/<module>/`, one-way layering, no third-party types in
public headers) are in [ADR 0003](docs/adr/0003-module-architecture.md). The RHI design is in
[ADR 0008](docs/adr/0008-rendering-core-rhi.md).

## Roadmap

1. **Foundation** — build/CI, window & input, logging, memory, jobs. *(done)*
2. **Rendering Core** — Vulkan RHI, simple 3D scene, CPU/GPU frame timing. *(in progress: pending validation
   on reference Windows GPUs, shader hot reload)*
3. **Scene & Editor** — ECS, scene save/load, Hierarchy, Inspector, gizmos.
4. **Smart Rendering** — culling, LOD/HLOD, instancing, render graph, streaming (with documented gains).
5. **Advanced Graphics** — shadows, GI, fog, SSR, optional ray tracing.
6. **Physics & Animation** — Jolt behind an interface; FPS/TPS character interacting with the world.
7. **Production** — audio, packaging, tools, sample game.

First vertical slice: create a project, import a glTF model, place it in a scene, apply PBR, add a light,
camera and simple physics, save the scene and run it in a standalone Runtime.

## Contributing

Each task states its scope, the files it owns, the API contract it touches and its success criteria. Work
happens on a branch; a change merges only after it builds with warnings-as-errors, passes `ctest` (including
GPU tests on lavapipe and sanitizer jobs in CI) and is reviewed. Changes to public headers or cross-module
rules need an ADR ([ADR 0001](docs/adr/0001-record-architecture-decisions.md)). Code style: `.clang-format`.
