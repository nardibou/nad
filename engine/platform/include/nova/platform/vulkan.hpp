#pragma once

// Bridge between a platform Window and the Vulkan RHI. Types are erased so the platform
// contract does not depend on Vulkan or SDL headers (ADR 0003, 0004).

#include "nova/core/types.hpp"
#include "nova/platform/window.hpp"

#include <vector>

namespace nova::platform {

// Instance extensions the window system needs (empty for Headless or non-Vulkan windows).
[[nodiscard]] std::vector<const char*> vulkan_instance_extensions(const Window& window);

// Creates a VkSurfaceKHR for `window`. `vk_instance` is a VkInstance. Returns the surface
// handle as an integer, or 0 on failure (logged).
[[nodiscard]] u64 create_vulkan_surface(const Window& window, void* vk_instance);

} // namespace nova::platform
