# Header-only module.
add_library(nova_math INTERFACE)
add_library(Nova::math ALIAS nova_math)
target_include_directories(nova_math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nova_math INTERFACE Nova::core)
