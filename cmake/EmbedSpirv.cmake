# Converts a SPIR-V binary into a C++ header with a u32 array.
#   cmake -DINPUT=<file.spv> -DOUTPUT=<file.hpp> -DSYMBOL=<name> -P EmbedSpirv.cmake
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" len)
math(EXPR rem "${len} % 8")
if(NOT rem EQUAL 0)
    message(FATAL_ERROR "${INPUT} is not a whole number of 32-bit words")
endif()

set(words "")
set(count 0)
math(EXPR last "${len} - 8")
foreach(i RANGE 0 ${last} 8)
    # SPIR-V words are little-endian in the file.
    string(SUBSTRING "${hex}" ${i} 2 b0)
    math(EXPR o1 "${i} + 2")
    math(EXPR o2 "${i} + 4")
    math(EXPR o3 "${i} + 6")
    string(SUBSTRING "${hex}" ${o1} 2 b1)
    string(SUBSTRING "${hex}" ${o2} 2 b2)
    string(SUBSTRING "${hex}" ${o3} 2 b3)
    string(APPEND words "0x${b3}${b2}${b1}${b0}u,")
    math(EXPR count "${count} + 1")
    math(EXPR col "${count} % 8")
    if(col EQUAL 0)
        string(APPEND words "\n    ")
    endif()
endforeach()

file(WRITE "${OUTPUT}" "// Generated from ${INPUT}. Do not edit.
#pragma once
#include <cstdint>

inline constexpr std::uint32_t ${SYMBOL}[] = {
    ${words}
};
")
