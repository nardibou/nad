#include "backends.hpp"

#include "nova/core/log.hpp"

namespace nova::platform {
namespace detail {
namespace {

class HeadlessWindow final : public Window {
public:
    explicit HeadlessWindow(const WindowDesc& desc) : size_{desc.width, desc.height}, title_(desc.title) {}

    bool poll_events(Input&) override { return !close_requested_; }
    WindowSize size() const noexcept override { return size_; }
    bool should_close() const noexcept override { return close_requested_; }
    void request_close() noexcept override { close_requested_ = true; }
    void set_title(const std::string& title) override { title_ = title; }
    Backend backend() const noexcept override { return Backend::Headless; }
    void* native_handle() const noexcept override { return nullptr; }

private:
    WindowSize size_;
    std::string title_;
    bool close_requested_ = false;
};

} // namespace

std::unique_ptr<Window> create_headless_window(const WindowDesc& desc) {
    return std::make_unique<HeadlessWindow>(desc);
}

} // namespace detail

bool backend_available(Backend backend) noexcept {
    switch (backend) {
    case Backend::Default:
    case Backend::Headless: return true;
    case Backend::SDL3:
#if defined(NOVA_HAS_SDL3)
        return true;
#else
        return false;
#endif
    }
    return false;
}

std::unique_ptr<Window> create_window(const WindowDesc& desc) {
    Backend backend = desc.backend;
    if (backend == Backend::Default) {
        backend = backend_available(Backend::SDL3) ? Backend::SDL3 : Backend::Headless;
    }
    switch (backend) {
    case Backend::Headless: return detail::create_headless_window(desc);
    case Backend::SDL3:
#if defined(NOVA_HAS_SDL3)
        return detail::create_sdl3_window(desc);
#else
        NOVA_ERROR("Platform", "SDL3 backend requested but engine was built with NOVA_WITH_SDL3=OFF");
        return nullptr;
#endif
    case Backend::Default: break;
    }
    return nullptr;
}

} // namespace nova::platform
