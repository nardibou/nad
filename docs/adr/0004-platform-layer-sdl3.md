# 4. Platform layer: Windows-first with SDL3

- Status: Accepted
- Date: 2026-11-21

## Context
The first target is Windows desktop. We still want Linux builds for CI and dedicated servers, and we do not
want OS or SDL types spread through the engine.

## Decision
- `Nova::platform` exposes `Window`, `WindowDesc` and a backend-independent `Input` snapshot
  (`Key`, `MouseButton`, edge-triggered `was_pressed`/`was_released`, per-frame mouse deltas).
- **SDL3** (pinned 3.2.8, static) is the windowing/input backend, compiled only when `NOVA_WITH_SDL3=ON`.
  SDL headers are private to `engine/platform/src`.
- A **Headless** backend is always available, for unit tests, CI and servers (`--headless` in the sandbox).
- Windows are created with Vulkan surface support by default. If that fails (no Vulkan loader), the backend
  falls back to a plain window and logs a warning so non-rendering work is not blocked; the RHI reports the
  hard error once it exists.
- Focus loss resets input state so keys do not stick.

## Consequences
- The RHI receives the native window through `Window::native_handle()` (an `SDL_Window*` for SDL3) and is the
  only other place allowed to interpret it.
- Gamepad, text input, multi-window and DPI handling are follow-up work in this module.
