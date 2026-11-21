nova_add_module(rhi
    SOURCES
        src/vulkan_device.cpp
    PUBLIC_DEPS
        Nova::core
    PRIVATE_DEPS
        Nova::platform
        nova_volk)
