#include "nova/core/assert.hpp"
#include "nova/core/log.hpp"

#include <atomic>
#include <cstdlib>

namespace nova {
namespace {

void default_handler(std::string_view expression, std::string_view message,
                     const std::source_location& location) {
    log::write(log::Level::Fatal, "Assert",
               std::format("{} ({}) at {}:{}", message, expression, location.file_name(), location.line()),
               location);
}

std::atomic<AssertHandler> g_handler{&default_handler};

} // namespace

void set_assert_handler(AssertHandler handler) noexcept {
    g_handler.store(handler ? handler : &default_handler);
}

void assert_failed(std::string_view expression, std::string_view message, std::source_location location) {
    g_handler.load()(expression, message, location);
    std::abort(); // reached only if the handler returns
}

} // namespace nova
