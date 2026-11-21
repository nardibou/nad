#pragma once

#include "nova/core/types.hpp"

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace nova::memory {

inline constexpr usize kDefaultAlignment = alignof(std::max_align_t);

[[nodiscard]] constexpr bool is_power_of_two(usize v) noexcept { return v != 0 && (v & (v - 1)) == 0; }
[[nodiscard]] constexpr usize align_up(usize v, usize alignment) noexcept {
    return (v + alignment - 1) & ~(alignment - 1);
}

// Bump allocator over an owned buffer. Individual frees are not supported: rewind to a
// marker or reset() the whole arena (typical use: per-frame scratch memory).
// Not thread-safe; use one arena per thread.
class LinearAllocator {
public:
    using Marker = usize;

    explicit LinearAllocator(usize capacity);
    LinearAllocator(const LinearAllocator&) = delete;
    LinearAllocator& operator=(const LinearAllocator&) = delete;
    LinearAllocator(LinearAllocator&&) noexcept = default;
    LinearAllocator& operator=(LinearAllocator&&) noexcept = default;
    ~LinearAllocator();

    // Returns nullptr when the arena is exhausted. `alignment` must be a power of two.
    [[nodiscard]] void* allocate(usize size, usize alignment = kDefaultAlignment) noexcept;

    template <typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* p = allocate(sizeof(T), alignof(T));
        return p ? ::new (p) T(std::forward<Args>(args)...) : nullptr;
    }

    [[nodiscard]] Marker marker() const noexcept { return offset_; }
    // Discards every allocation made after `m`. Destructors are not run.
    void rewind(Marker m) noexcept;
    void reset() noexcept { offset_ = 0; }

    [[nodiscard]] usize capacity() const noexcept { return capacity_; }
    [[nodiscard]] usize used() const noexcept { return offset_; }
    [[nodiscard]] usize peak() const noexcept { return peak_; }

private:
    std::unique_ptr<std::byte[]> buffer_;
    usize capacity_ = 0;
    usize offset_ = 0;
    usize peak_ = 0;
};

// Fixed-size block allocator with an intrusive free list: O(1) allocate/free, no fragmentation.
// Not thread-safe.
class PoolAllocator {
public:
    PoolAllocator(usize block_size, usize block_count, usize alignment = kDefaultAlignment);
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    ~PoolAllocator();

    // Returns nullptr when every block is in use.
    [[nodiscard]] void* allocate() noexcept;
    // `p` must come from this pool (checked in Debug builds).
    void free(void* p) noexcept;
    [[nodiscard]] bool owns(const void* p) const noexcept;

    [[nodiscard]] usize block_size() const noexcept { return block_size_; }
    [[nodiscard]] usize block_count() const noexcept { return block_count_; }
    [[nodiscard]] usize used_blocks() const noexcept { return used_; }
    [[nodiscard]] usize peak_blocks() const noexcept { return peak_; }

private:
    struct FreeNode {
        FreeNode* next;
    };

    std::byte* storage_ = nullptr;
    usize block_size_ = 0;
    usize block_count_ = 0;
    usize alignment_ = 0;
    FreeNode* free_list_ = nullptr;
    usize used_ = 0;
    usize peak_ = 0;
};

} // namespace nova::memory
