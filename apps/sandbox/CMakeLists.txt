add_executable(nova_sandbox main.cpp)
target_link_libraries(nova_sandbox PRIVATE Nova::core Nova::platform Nova::renderer)
nova_set_warnings(nova_sandbox)
