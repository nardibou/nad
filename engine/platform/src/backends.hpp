#pragma once

#include "nova/platform/window.hpp"

namespace nova::platform::detail {

std::unique_ptr<Window> create_headless_window(const WindowDesc& desc);

#if defined(NOVA_HAS_SDL3)
std::unique_ptr<Window> create_sdl3_window(const WindowDesc& desc);
#endif

} // namespace nova::platform::detail
