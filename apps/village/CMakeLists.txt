add_library(nova_village_scene STATIC village_scene.cpp)
target_include_directories(nova_village_scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nova_village_scene PUBLIC Nova::core Nova::renderer)
nova_set_warnings(nova_village_scene)

add_executable(nova_village main.cpp)
target_link_libraries(nova_village PRIVATE Nova::core Nova::platform Nova::renderer nova_village_scene)
nova_set_warnings(nova_village)
