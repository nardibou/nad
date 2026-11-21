#pragma once

#include <source_location>
#include <string_view>

namespace nova {

// Called when an assertion fails. The default handler logs a Fatal record and aborts.
// Tests may install a handler that throws instead.
using AssertHandler = void (*)(std::string_view expression, std::string_view message,
                               const std::source_location& location);

void set_assert_handler(AssertHandler handler) noexcept;
[[noreturn]] void assert_failed(std::string_view expression, std::string_view message,
                                std::source_location location = std::source_location::current());

} // namespace nova

// NOVA_VERIFY is always checked; NOVA_ASSERT only when NOVA_ENABLE_ASSERTS is set (Debug builds).
#define NOVA_VERIFY(cond, msg)                                                                               \
    do {                                                                                                     \
        if (!(cond)) [[unlikely]] {                                                                          \
            ::nova::assert_failed(#cond, (msg));                                                             \
        }                                                                                                    \
    } while (false)

#if defined(NOVA_ENABLE_ASSERTS)
#define NOVA_ASSERT(cond, msg) NOVA_VERIFY(cond, msg)
#else
#define NOVA_ASSERT(cond, msg) ((void)0)
#endif
