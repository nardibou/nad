#include "nova/platform/vulkan.hpp"

#include "nova/core/log.hpp"

#if defined(NOVA_HAS_SDL3)
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#endif

namespace nova::platform {

std::vector<const char*> vulkan_instance_extensions(const Window& window) {
    std::vector<const char*> out;
#if defined(NOVA_HAS_SDL3)
    if (window.backend() == Backend::SDL3 &&
        (SDL_GetWindowFlags(static_cast<SDL_Window*>(window.native_handle())) & SDL_WINDOW_VULKAN)) {
        Uint32 count = 0;
        const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
        out.assign(names, names + count);
    }
#else
    (void)window;
#endif
    return out;
}

u64 create_vulkan_surface(const Window& window, void* vk_instance) {
#if defined(NOVA_HAS_SDL3)
    if (window.backend() == Backend::SDL3) {
        auto* sdl_window = static_cast<SDL_Window*>(window.native_handle());
        if (!(SDL_GetWindowFlags(sdl_window) & SDL_WINDOW_VULKAN)) {
            NOVA_ERROR("Platform", "window was created without Vulkan support");
            return 0;
        }
        VkSurfaceKHR surface{};
        if (!SDL_Vulkan_CreateSurface(sdl_window, static_cast<VkInstance>(vk_instance), nullptr, &surface)) {
            NOVA_ERROR("Platform", "SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
            return 0;
        }
        return reinterpret_cast<u64>(surface);
    }
#else
    (void)vk_instance;
#endif
    (void)window;
    NOVA_ERROR("Platform", "backend cannot create a Vulkan surface");
    return 0;
}

} // namespace nova::platform
