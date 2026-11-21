add_executable(nova_tests
    main.cpp
    core_tests.cpp
    memory_tests.cpp
    jobs_tests.cpp
    platform_tests.cpp)
target_link_libraries(nova_tests PRIVATE
    Nova::core Nova::memory Nova::jobs Nova::platform Nova::math doctest::doctest)
if(NOVA_WITH_VULKAN)
    target_sources(nova_tests PRIVATE math_tests.cpp model_tests.cpp animation_tests.cpp)
    target_compile_definitions(nova_tests PRIVATE NOVA_TEST_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
    target_link_libraries(nova_tests PRIVATE Nova::renderer nova_stb)
endif()
nova_set_warnings(nova_tests)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(nova_tests)

if(NOVA_WITH_VULKAN)
    # Real rendering through the RHI; exit code 77 = no Vulkan device => skipped.
    add_executable(nova_gpu_tests gpu_tests.cpp)
    target_link_libraries(nova_gpu_tests PRIVATE Nova::core Nova::renderer doctest::doctest)
    nova_set_warnings(nova_gpu_tests)
    add_test(NAME gpu.rendering COMMAND nova_gpu_tests)
    set_tests_properties(gpu.rendering PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Smoke test: the sandbox renders offscreen end-to-end.
if(TARGET nova_sandbox)
    add_test(NAME sandbox.headless_smoke COMMAND nova_sandbox --headless --frames 120)
    set_tests_properties(sandbox.headless_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Smoke test: the village scene builds and renders offscreen with validation on.
if(TARGET nova_village)
    add_test(NAME village.headless_smoke
             COMMAND nova_village --headless --validation --frames 4 --width 640 --height 360)
    set_tests_properties(village.headless_smoke PROPERTIES SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION
                         "Validation Error")
endif()

# Smoke test: the Guerrilla demo plays a scripted burst (fire, recoil, impacts) with validation on.
if(TARGET nova_guerrilla)
    add_test(NAME guerrilla.headless_smoke
             COMMAND nova_guerrilla --headless --validation --frames 40 --fire --width 640 --height 360)
    set_tests_properties(guerrilla.headless_smoke PROPERTIES SKIP_RETURN_CODE 77 FAIL_REGULAR_EXPRESSION
                         "Validation Error")
endif()
