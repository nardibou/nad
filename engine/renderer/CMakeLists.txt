nova_add_module(renderer
    SOURCES
        src/mesh.cpp
        src/renderer.cpp
        src/world_renderer.cpp
        src/ui_overlay.cpp
        src/animation.cpp
        src/gltf_loader.cpp
    PUBLIC_DEPS
        Nova::math
        Nova::rhi
    PRIVATE_DEPS
        nova_cgltf
        nova_stb)

nova_add_shader(nova_renderer shaders/forward.slang forward_spv)
nova_add_shader(nova_renderer shaders/world.slang world_spv)
nova_add_shader(nova_renderer shaders/ui.slang ui_spv)
