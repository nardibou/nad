cmake_minimum_required(VERSION 3.25)

project(NovaEngine
    VERSION 0.1.0
    DESCRIPTION "NOVA Engine - modular 3D game engine"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(NOVA_BUILD_TESTS "Build unit tests" ON)
option(NOVA_BUILD_SANDBOX "Build the sandbox application" ON)
option(NOVA_WITH_SDL3 "Build the SDL3 platform backend (otherwise headless only)" ON)
option(NOVA_WITH_VULKAN "Build the Vulkan RHI, renderer and sandbox (downloads Vulkan headers, volk, slangc)" ON)
option(NOVA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

include(NovaModule)
include(NovaDependencies)

add_subdirectory(engine/core)
add_subdirectory(engine/memory)
add_subdirectory(engine/jobs)
add_subdirectory(engine/platform)
add_subdirectory(engine/math)

if(NOVA_WITH_VULKAN)
    add_subdirectory(engine/rhi)
    add_subdirectory(engine/renderer)
endif()

if(NOVA_BUILD_SANDBOX AND NOVA_WITH_VULKAN)
    add_subdirectory(apps/sandbox)
    add_subdirectory(apps/village)
    add_subdirectory(apps/guerrilla)
endif()

if(NOVA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
