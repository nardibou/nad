# Shared helpers for engine modules.
#
# nova_add_module(<name> SOURCES ... [PUBLIC_DEPS ...] [PRIVATE_DEPS ...])
#   Creates static library `nova_<name>` with alias `Nova::<name>`.
#   Public headers live in <module>/include/nova/<name>/ and are the module's only
#   contract; everything under src/ is private (see docs/adr/0003).

function(nova_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
        # Portable C functions (fopen, getenv) over MSVC-only *_s variants.
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        if(NOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
            -Wno-sign-conversion
            -Wno-missing-field-initializers) # idiomatic `VkFoo info{VK_STRUCTURE_TYPE_FOO}`
        if(NOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(nova_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})
    set(target nova_${name})
    add_library(${target} STATIC ${ARG_SOURCES})
    add_library(Nova::${name} ALIAS ${target})
    target_include_directories(${target}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_definitions(${target} PUBLIC $<$<CONFIG:Debug>:NOVA_ENABLE_ASSERTS=1>)
    if(ARG_PUBLIC_DEPS)
        target_link_libraries(${target} PUBLIC ${ARG_PUBLIC_DEPS})
    endif()
    if(ARG_PRIVATE_DEPS)
        target_link_libraries(${target} PRIVATE ${ARG_PRIVATE_DEPS})
    endif()
    nova_set_warnings(${target})
endfunction()

# nova_add_shader(<target> <file.slang> <symbol> [PROFILE <spirv_x_y>] [DEFINES <NAME=VALUE>...])
#   Compiles all entry points of a Slang file to one SPIR-V module and embeds it as
#   `inline constexpr std::uint32_t <symbol>[]` in <symbol>.hpp, reachable from <target>.
function(nova_add_shader target source symbol)
    cmake_parse_arguments(ARG "" "PROFILE" "DEFINES" ${ARGN})
    if(NOT ARG_PROFILE)
        set(ARG_PROFILE spirv_1_3)
    endif()
    set(defines "")
    foreach(d IN LISTS ARG_DEFINES)
        list(APPEND defines "-D${d}")
    endforeach()
    get_filename_component(src "${source}" ABSOLUTE)
    set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(spv "${out_dir}/${symbol}.spv")
    set(hpp "${out_dir}/${symbol}.hpp")
    add_custom_command(
        OUTPUT "${hpp}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
        COMMAND "${NOVA_SLANGC}" "${src}" -target spirv -profile ${ARG_PROFILE} ${defines}
                -matrix-layout-column-major -fvk-use-entrypoint-name -O2 -o "${spv}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${hpp} -DSYMBOL=${symbol}
                -P "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        DEPENDS "${src}" "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        COMMENT "Compiling shader ${source}"
        VERBATIM)
    target_sources(${target} PRIVATE "${hpp}")
    target_include_directories(${target} PRIVATE "${out_dir}")
endfunction()
