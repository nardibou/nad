#include "backends.hpp"

#include "nova/core/log.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>

namespace nova::platform::detail {
namespace {

// SDL_Init/SDL_Quit are global; reference-count them across windows.
std::atomic<int> g_sdl_users{0};

bool acquire_sdl() {
    if (g_sdl_users.fetch_add(1) == 0 && !SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        NOVA_ERROR("Platform", "SDL_InitSubSystem(VIDEO) failed: {}", SDL_GetError());
        g_sdl_users.fetch_sub(1);
        return false;
    }
    return true;
}

void release_sdl() {
    if (g_sdl_users.fetch_sub(1) == 1) {
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        SDL_Quit();
    }
}

Key translate(SDL_Scancode sc) noexcept {
    if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) {
        return static_cast<Key>(static_cast<u16>(Key::A) + (sc - SDL_SCANCODE_A));
    }
    if (sc >= SDL_SCANCODE_1 && sc <= SDL_SCANCODE_9) {
        return static_cast<Key>(static_cast<u16>(Key::Num1) + (sc - SDL_SCANCODE_1));
    }
    if (sc >= SDL_SCANCODE_F1 && sc <= SDL_SCANCODE_F12) {
        return static_cast<Key>(static_cast<u16>(Key::F1) + (sc - SDL_SCANCODE_F1));
    }
    switch (sc) {
    case SDL_SCANCODE_0: return Key::Num0;
    case SDL_SCANCODE_ESCAPE: return Key::Escape;
    case SDL_SCANCODE_RETURN: return Key::Enter;
    case SDL_SCANCODE_TAB: return Key::Tab;
    case SDL_SCANCODE_BACKSPACE: return Key::Backspace;
    case SDL_SCANCODE_DELETE: return Key::Delete;
    case SDL_SCANCODE_SPACE: return Key::Space;
    case SDL_SCANCODE_MINUS: return Key::Minus;
    case SDL_SCANCODE_EQUALS: return Key::Equal;
    case SDL_SCANCODE_LEFT: return Key::Left;
    case SDL_SCANCODE_RIGHT: return Key::Right;
    case SDL_SCANCODE_UP: return Key::Up;
    case SDL_SCANCODE_DOWN: return Key::Down;
    case SDL_SCANCODE_LSHIFT: return Key::LeftShift;
    case SDL_SCANCODE_RSHIFT: return Key::RightShift;
    case SDL_SCANCODE_LCTRL: return Key::LeftCtrl;
    case SDL_SCANCODE_RCTRL: return Key::RightCtrl;
    case SDL_SCANCODE_LALT: return Key::LeftAlt;
    case SDL_SCANCODE_RALT: return Key::RightAlt;
    default: return Key::Unknown;
    }
}

bool translate(Uint8 sdl_button, MouseButton& out) noexcept {
    switch (sdl_button) {
    case SDL_BUTTON_LEFT: out = MouseButton::Left; return true;
    case SDL_BUTTON_RIGHT: out = MouseButton::Right; return true;
    case SDL_BUTTON_MIDDLE: out = MouseButton::Middle; return true;
    case SDL_BUTTON_X1: out = MouseButton::X1; return true;
    case SDL_BUTTON_X2: out = MouseButton::X2; return true;
    default: return false;
    }
}

class Sdl3Window final : public Window {
public:
    explicit Sdl3Window(SDL_Window* window) : window_(window) {}
    ~Sdl3Window() override {
        SDL_DestroyWindow(window_);
        release_sdl();
    }

    bool poll_events(Input& input) override {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED: close_requested_ = true; break;
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP:
                if (!e.key.repeat) {
                    input.on_key(translate(e.key.scancode), e.type == SDL_EVENT_KEY_DOWN);
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                MouseButton b;
                if (translate(e.button.button, b)) {
                    input.on_mouse_button(b, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION: {
                const f32 density = SDL_GetWindowPixelDensity(window_);
                const f32 scale = density > 0.0f ? density : 1.0f;
                input.on_mouse_move(e.motion.x * scale, e.motion.y * scale, e.motion.xrel, e.motion.yrel);
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL: input.on_mouse_wheel(e.wheel.y); break;
            case SDL_EVENT_DROP_FILE:
                if (e.drop.data) {
                    input.on_file_drop(e.drop.data);
                }
                break;
            case SDL_EVENT_WINDOW_FOCUS_LOST: input.reset(); break;
            default: break;
            }
        }
        return !close_requested_;
    }

    WindowSize size() const noexcept override {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(window_, &w, &h);
        return {static_cast<u32>(w), static_cast<u32>(h)};
    }
    bool should_close() const noexcept override { return close_requested_; }
    void request_close() noexcept override { close_requested_ = true; }
    void set_title(const std::string& title) override { SDL_SetWindowTitle(window_, title.c_str()); }
    Backend backend() const noexcept override { return Backend::SDL3; }
    void* native_handle() const noexcept override { return window_; }
    void set_mouse_captured(bool captured) override { SDL_SetWindowRelativeMouseMode(window_, captured); }
    void minimize() override { SDL_MinimizeWindow(window_); }
    void toggle_maximize() override {
        if (is_fullscreen()) {
            SDL_SetWindowFullscreen(window_, false);
        }
        if (is_maximized()) {
            SDL_RestoreWindow(window_);
        } else {
            SDL_MaximizeWindow(window_);
        }
    }
    void toggle_fullscreen() override {
        SDL_SetWindowFullscreen(window_, !is_fullscreen());
    }
    bool is_fullscreen() const noexcept override {
        return (SDL_GetWindowFlags(window_) & SDL_WINDOW_FULLSCREEN) != 0;
    }
    bool is_maximized() const noexcept override {
        return (SDL_GetWindowFlags(window_) & SDL_WINDOW_MAXIMIZED) != 0;
    }

private:
    SDL_Window* window_;
    bool close_requested_ = false;
};

} // namespace

std::unique_ptr<Window> create_sdl3_window(const WindowDesc& desc) {
    if (!acquire_sdl()) {
        return nullptr;
    }
    SDL_WindowFlags flags = desc.resizable ? SDL_WINDOW_RESIZABLE : 0;
    int w = static_cast<int>(desc.width);
    int h = static_cast<int>(desc.height);

    // Keep windowed mode comfortably inside the desktop usable area (so titlebar & taskbar stay accessible;
    // Fullscreen is always an explicit user choice via the FS button or F11).
    SDL_Rect usable{};
    const SDL_DisplayID display = SDL_GetPrimaryDisplay();
    if (display != 0 && SDL_GetDisplayUsableBounds(display, &usable) && usable.w > 320 && usable.h > 240) {
        if (w >= usable.w || h >= usable.h) {
            w = std::min(w, std::max(960, usable.w - 48));
            h = std::min(h, std::max(540, usable.h - 72));
        }
    }

    SDL_Window* window = SDL_CreateWindow(desc.title.c_str(), w, h, flags | (desc.vulkan ? SDL_WINDOW_VULKAN : 0));
    if (!window && desc.vulkan) {
        // Keeps the window usable on machines without a Vulkan loader; the RHI will report it.
        NOVA_WARN("Platform", "Vulkan window creation failed ({}); retrying without Vulkan", SDL_GetError());
        window = SDL_CreateWindow(desc.title.c_str(), w, h, flags);
    }
    if (!window) {
        NOVA_ERROR("Platform", "SDL_CreateWindow failed: {}", SDL_GetError());
        release_sdl();
        return nullptr;
    }
    NOVA_INFO("Platform", "SDL3 window {}x{} (video driver: {})", w, h, SDL_GetCurrentVideoDriver());
    return std::make_unique<Sdl3Window>(window);
}

} // namespace nova::platform::detail
