nova_add_module(platform
    SOURCES
        src/input.cpp
        src/vulkan.cpp
        src/window.cpp
    PUBLIC_DEPS
        Nova::core)

if(NOVA_WITH_SDL3)
    target_sources(nova_platform PRIVATE src/sdl3_window.cpp)
    target_link_libraries(nova_platform PRIVATE SDL3::SDL3)
    target_compile_definitions(nova_platform PRIVATE NOVA_HAS_SDL3=1)
endif()
