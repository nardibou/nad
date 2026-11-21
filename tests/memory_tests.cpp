#include "nova/memory/allocators.hpp"

#include <doctest/doctest.h>

#include <cstdint>
#include <set>
#include <vector>

using namespace nova;
using namespace nova::memory;

namespace {
bool aligned(const void* p, usize a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }
} // namespace

TEST_CASE("linear: respects alignment and capacity") {
    LinearAllocator arena(256);
    void* a = arena.allocate(1, 1);
    void* b = arena.allocate(8, 64);
    REQUIRE(a);
    REQUIRE(b);
    CHECK(aligned(b, 64));
    CHECK(arena.used() <= arena.capacity());

    CHECK(arena.allocate(1024) == nullptr); // too big, arena unchanged
    const usize before = arena.used();
    CHECK(arena.allocate(arena.capacity()) == nullptr);
    CHECK(arena.used() == before);
}

TEST_CASE("linear: markers rewind and reset reuses memory") {
    LinearAllocator arena(128);
    void* first = arena.allocate(16);
    const auto m = arena.marker();
    void* second = arena.allocate(32);
    arena.rewind(m);
    CHECK(arena.allocate(32) == second);
    CHECK(arena.peak() >= arena.used());

    arena.reset();
    CHECK(arena.used() == 0);
    CHECK(arena.allocate(16) == first);
}

TEST_CASE("linear: create constructs objects in place") {
    struct Vec3 {
        float x, y, z;
    };
    LinearAllocator arena(64);
    Vec3* v = arena.create<Vec3>(1.0f, 2.0f, 3.0f);
    REQUIRE(v);
    CHECK(v->y == 2.0f);
    CHECK(aligned(v, alignof(Vec3)));
}

TEST_CASE("pool: hands out distinct aligned blocks until exhausted") {
    PoolAllocator pool(24, 8, 32);
    CHECK(pool.block_size() % 32 == 0);

    std::set<void*> blocks;
    for (int i = 0; i < 8; ++i) {
        void* p = pool.allocate();
        REQUIRE(p);
        CHECK(aligned(p, 32));
        CHECK(pool.owns(p));
        blocks.insert(p);
    }
    CHECK(blocks.size() == 8);
    CHECK(pool.allocate() == nullptr);
    CHECK(pool.used_blocks() == 8);

    void* any = *blocks.begin();
    pool.free(any);
    CHECK(pool.used_blocks() == 7);
    CHECK(pool.allocate() == any); // LIFO reuse
    CHECK(pool.peak_blocks() == 8);
}

TEST_CASE("pool: rejects foreign pointers") {
    PoolAllocator pool(16, 4);
    int local = 0;
    CHECK_FALSE(pool.owns(&local));
    auto* p = static_cast<std::byte*>(pool.allocate());
    CHECK_FALSE(pool.owns(p + 1)); // not on a block boundary
    pool.free(nullptr); // no-op
}
