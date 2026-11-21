#define DOCTEST_CONFIG_IMPLEMENT
#include "nova/core/assert.hpp"
#include "nova/core/log.hpp"

#include <doctest/doctest.h>

#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void throwing_assert_handler(std::string_view expr, std::string_view msg, const std::source_location&) {
    throw std::logic_error(std::string(msg) + " (" + std::string(expr) + ")");
}

} // namespace

int main(int argc, char** argv) {
    // Assertion failures become test failures instead of aborting the runner.
    nova::set_assert_handler(&throwing_assert_handler);
    nova::log::set_console_enabled(false);

    doctest::Context context(argc, argv);
    return context.run();
}
