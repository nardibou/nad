# 5. Graphics API and shading language

- Status: Proposed — to be confirmed by the Rendering Core prototype
- Date: 2026-11-21

## Context
The renderer targets GPU-driven rendering (culling, LOD/HLOD, instancing, streaming, render graph) on
Windows. The choice must be validated on target hardware before the engine commits to it.

## Decision (proposed)
- **Vulkan 1.3** first (dynamic rendering, synchronization2, descriptor indexing), behind an internal RHI so a
  D3D12 backend remains possible.
- **Slang** for shaders, compiling to SPIR-V; HLSL accepted where needed.
- Validation layers are on in Debug; RenderDoc and Tracy integrations are part of Rendering Core.

## Acceptance criteria for moving to Accepted
1. Prototype renders a lit 3D scene through the RHI on the reference Windows machines.
2. CPU and GPU frame times are measured separately (budget: 16.67 ms at 60 FPS) with a fixed benchmark scene.
3. Slang → SPIR-V compilation is integrated into the build and hot-reload loop.

## Consequences
- No graphics code is merged before the RHI interface is reviewed under this ADR.

## Progress
- Criteria 2 and 3 are met by the Rendering Core work (ADR 0008): CPU/GPU frame times are reported by the
  sandbox, and Slang → SPIR-V runs in the build. Hot reload is still to do.
- Criterion 1 is verified on lavapipe (Linux, software) and an X11 swapchain, not yet on the reference
  Windows GPUs.
