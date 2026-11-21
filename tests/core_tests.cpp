#include "nova/core/assert.hpp"
#include "nova/core/log.hpp"
#include "nova/core/time.hpp"

#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace nova;

TEST_CASE("log: sinks receive formatted records at or above the level") {
    std::vector<std::string> got;
    const auto id = log::add_sink([&](const log::Record& r) {
        got.push_back(std::string(r.channel) + ":" + std::string(r.message));
    });
    const auto old = log::level();
    log::set_level(log::Level::Info);

    NOVA_DEBUG("Test", "hidden {}", 1);
    NOVA_INFO("Test", "value={} name={}", 42, "nova");
    NOVA_ERROR("Test", "boom");

    log::set_level(old);
    log::remove_sink(id);
    NOVA_INFO("Test", "after removal");

    REQUIRE(got.size() == 2);
    CHECK(got[0] == "Test:value=42 name=nova");
    CHECK(got[1] == "Test:boom");
}

TEST_CASE("log: Off disables everything") {
    int calls = 0;
    const auto id = log::add_sink([&](const log::Record&) { ++calls; });
    const auto old = log::level();
    log::set_level(log::Level::Off);
    NOVA_FATAL("Test", "nope");
    log::set_level(old);
    log::remove_sink(id);
    CHECK(calls == 0);
}

TEST_CASE("assert: NOVA_VERIFY routes failures to the installed handler") {
    CHECK_NOTHROW(NOVA_VERIFY(1 + 1 == 2, "math works"));
    CHECK_THROWS_AS(NOVA_VERIFY(1 + 1 == 3, "math is broken"), std::logic_error);
}

TEST_CASE("time: FrameStats computes avg, min, max and p99") {
    FrameStats stats(100);
    for (int i = 1; i <= 100; ++i) {
        stats.record(static_cast<double>(i)); // 1..100 ms
    }
    const auto s = stats.summarize();
    CHECK(s.frame_count == 100);
    CHECK(s.avg_ms == doctest::Approx(50.5));
    CHECK(s.min_ms == doctest::Approx(1.0));
    CHECK(s.max_ms == doctest::Approx(100.0));
    CHECK(s.p99_ms == doctest::Approx(99.0));
    CHECK(s.one_percent_low_fps == doctest::Approx(1000.0 / 99.0));
}

TEST_CASE("time: FrameStats keeps only the rolling window") {
    FrameStats stats(4);
    for (double v : {100.0, 100.0, 1.0, 2.0, 3.0, 4.0}) {
        stats.record(v);
    }
    const auto s = stats.summarize();
    CHECK(s.frame_count == 4);
    CHECK(s.max_ms == doctest::Approx(4.0));
    CHECK(s.avg_ms == doctest::Approx(2.5));

    stats.clear();
    CHECK(stats.summarize().frame_count == 0);
}

TEST_CASE("time: Stopwatch is monotonic") {
    Stopwatch sw;
    const double a = sw.elapsed_ms();
    const double b = sw.elapsed_ms();
    CHECK(a >= 0.0);
    CHECK(b >= a);
}
