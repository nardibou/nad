add_executable(nova_guerrilla main.cpp viewmodel.cpp rig_viewmodel.cpp)
target_link_libraries(nova_guerrilla PRIVATE Nova::core Nova::platform Nova::renderer nova_village_scene)
# Fallback location of assets/ when the game is not started from the repository root.
target_compile_definitions(nova_guerrilla PRIVATE NOVA_GUERRILLA_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
nova_set_warnings(nova_guerrilla)
