#pragma once

#include "nova/core/types.hpp"
#include "nova/platform/input.hpp"

#include <memory>
#include <string>

namespace nova::platform {

enum class Backend : u8 {
    Default,  // SDL3 when built with it, otherwise Headless
    Headless, // no OS window; for tests, CI and dedicated servers
    SDL3,
};

struct WindowDesc {
    std::string title = "NOVA Engine";
    u32 width = 1280;
    u32 height = 720;
    bool resizable = true;
    bool vulkan = true; // create the OS window with Vulkan surface support
    Backend backend = Backend::Default;
};

struct WindowSize {
    u32 width = 0;
    u32 height = 0;
};

class Window {
public:
    virtual ~Window() = default;

    // Pumps OS events into `input`. Returns false once the window has been asked to close.
    virtual bool poll_events(Input& input) = 0;
    [[nodiscard]] virtual WindowSize size() const noexcept = 0;
    [[nodiscard]] virtual bool should_close() const noexcept = 0;
    virtual void request_close() noexcept = 0;
    virtual void set_title(const std::string& title) = 0;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    // Backend window handle (SDL_Window* for SDL3, nullptr for Headless). For the RHI only.
    [[nodiscard]] virtual void* native_handle() const noexcept = 0;
    // FPS-style mouse: hides the cursor and reports unbounded relative motion. No-op when headless.
    virtual void set_mouse_captured(bool /*captured*/) {}
    virtual void minimize() {}
    virtual void toggle_maximize() {}
    virtual void toggle_fullscreen() {}
    [[nodiscard]] virtual bool is_fullscreen() const noexcept { return false; }
    [[nodiscard]] virtual bool is_maximized() const noexcept { return false; }
};

// Returns nullptr (and logs) if the requested backend is unavailable or creation fails.
[[nodiscard]] std::unique_ptr<Window> create_window(const WindowDesc& desc);
[[nodiscard]] bool backend_available(Backend backend) noexcept;

} // namespace nova::platform
