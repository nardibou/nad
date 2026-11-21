#include "nova/jobs/job_system.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

using namespace nova;
using namespace nova::jobs;

TEST_CASE("jobs: every submitted job runs exactly once") {
    JobSystem js(4);
    CHECK(js.worker_count() == 4);

    std::atomic<int> counter{0};
    WaitGroup wg;
    for (int i = 0; i < 1000; ++i) {
        js.submit([&] { counter.fetch_add(1, std::memory_order_relaxed); }, &wg);
    }
    js.wait(wg);
    CHECK(wg.done());
    CHECK(counter.load() == 1000);
}

TEST_CASE("jobs: parallel_for covers the range without overlap") {
    JobSystem js(3);
    std::vector<int> hits(10'007, 0);
    js.parallel_for(static_cast<u32>(hits.size()), 64, [&](u32 begin, u32 end) {
        for (u32 i = begin; i < end; ++i) {
            ++hits[i];
        }
    });
    CHECK(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; }));

    bool called = false;
    js.parallel_for(0, 16, [&](u32, u32) { called = true; });
    CHECK_FALSE(called);
}

TEST_CASE("jobs: nested waits do not deadlock with a single worker") {
    JobSystem js(1);
    std::atomic<int> leaves{0};
    WaitGroup outer;
    for (int i = 0; i < 8; ++i) {
        js.submit([&] {
            WaitGroup inner;
            for (int j = 0; j < 8; ++j) {
                js.submit([&] { leaves.fetch_add(1); }, &inner);
            }
            js.wait(inner);
        }, &outer);
    }
    js.wait(outer);
    CHECK(leaves.load() == 64);
}

TEST_CASE("jobs: destructor drains queued work") {
    std::atomic<int> ran{0};
    {
        JobSystem js(2);
        for (int i = 0; i < 200; ++i) {
            js.submit([&] { ran.fetch_add(1); });
        }
    }
    CHECK(ran.load() == 200);
}

TEST_CASE("jobs: default worker count is at least one") {
    JobSystem js;
    CHECK(js.worker_count() >= 1);
}
