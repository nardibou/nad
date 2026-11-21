# 8. Rendering Core: RHI shape and first renderer

- Status: Accepted
- Date: 2026-11-21

## Context
Phase 2 needs a lit 3D scene through Vulkan with CPU and GPU frame time measured separately, without
locking the engine into Vulkan types or into designs we have not measured yet.

## Decision
- **`Nova::rhi`** exposes `Device` and `CommandList` interfaces with typed 32-bit handles
  (`BufferHandle`, `ShaderHandle`, `PipelineHandle`). No Vulkan types in public headers.
- **Vulkan 1.3 backend** using dynamic rendering and synchronization2 (no render pass/framebuffer objects),
  2 frames in flight, one acquire semaphore per frame and one present semaphore per swapchain image.
  Entry points are loaded with **volk** so the engine never links against the Vulkan loader; headers come from
  pinned **Vulkan-Headers**.
- **Headless devices** render into an offscreen sRGB target and support `read_back()`. This is how GPU tests
  and CI verify rendering (lavapipe on Linux) and how deterministic screenshots are produced.
- **Validation** is on by default in Debug builds and in GPU tests; any validation error fails the test /
  sandbox run.
- **GPU timing** uses timestamp queries at the top and bottom of each frame, read back one frame-in-flight
  later (`Device::last_gpu_frame_ms()`); CPU frame time is measured by `FrameStats` in the app.
- **Shaders** are written in Slang, compiled at build time to SPIR-V by `nova_add_shader()` and embedded as
  `u32` arrays (no runtime file I/O). Matrices are column-major on both CPU and GPU.
- **`Nova::renderer`**: baseline forward renderer (one pipeline, one directional light, one draw per item,
  128-byte push constants). This is the measured baseline that culling/instancing/render graph must beat.
- **`Nova::math`**: small header-only math (right-handed, column-major, Vulkan clip space with the Y flip in
  `perspective()`), kept in-house until a need for SIMD math is measured.

## Known simplifications (revisit with profiling data)
- One `vkAllocateMemory` per resource and host-visible vertex/index buffers. Replace with VMA (or a custom
  sub-allocator) plus staging uploads when resource counts grow.
- `destroy()` waits for device idle. Replace with per-frame deferred deletion queues.
- No descriptor sets yet; per-draw data goes through push constants. Bindless descriptors arrive with
  materials/textures.
- Swapchain resize is handled (out-of-date/suboptimal, minimized windows) but only exercised manually.

## Consequences
- Everything above the RHI can be tested on machines without a GPU (lavapipe) and in CI.
- ADR 0005 stays Proposed until the scene is validated on the reference Windows GPUs.
