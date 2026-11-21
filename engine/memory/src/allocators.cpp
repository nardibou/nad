#include "nova/memory/allocators.hpp"

#include "nova/core/assert.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>

namespace nova::memory {

// ---- LinearAllocator ----

LinearAllocator::LinearAllocator(usize capacity)
    : buffer_(new std::byte[capacity]), capacity_(capacity) {}

LinearAllocator::~LinearAllocator() = default;

void* LinearAllocator::allocate(usize size, usize alignment) noexcept {
    NOVA_ASSERT(is_power_of_two(alignment), "alignment must be a power of two");
    // Align the absolute address, not just the offset: the buffer is only max_align_t aligned.
    const auto base = reinterpret_cast<std::uintptr_t>(buffer_.get());
    const usize aligned = align_up(base + offset_, alignment) - base;
    if (aligned > capacity_ || size > capacity_ - aligned) {
        return nullptr;
    }
    offset_ = aligned + size;
    peak_ = std::max(peak_, offset_);
    return buffer_.get() + aligned;
}

void LinearAllocator::rewind(Marker m) noexcept {
    NOVA_ASSERT(m <= offset_, "cannot rewind forward");
    offset_ = m;
}

// ---- PoolAllocator ----

PoolAllocator::PoolAllocator(usize block_size, usize block_count, usize alignment)
    : block_count_(block_count), alignment_(std::max(alignment, alignof(FreeNode))) {
    NOVA_VERIFY(is_power_of_two(alignment_), "alignment must be a power of two");
    NOVA_VERIFY(block_count > 0, "pool needs at least one block");
    block_size_ = align_up(std::max(block_size, sizeof(FreeNode)), alignment_);
    storage_ = static_cast<std::byte*>(
        ::operator new(block_size_ * block_count_, std::align_val_t{alignment_}));

    // Thread the free list through the blocks in address order.
    for (usize i = block_count_; i-- > 0;) {
        auto* node = reinterpret_cast<FreeNode*>(storage_ + i * block_size_);
        node->next = free_list_;
        free_list_ = node;
    }
}

PoolAllocator::~PoolAllocator() { ::operator delete(storage_, std::align_val_t{alignment_}); }

void* PoolAllocator::allocate() noexcept {
    if (!free_list_) {
        return nullptr;
    }
    FreeNode* node = free_list_;
    free_list_ = node->next;
    ++used_;
    peak_ = std::max(peak_, used_);
    return node;
}

void PoolAllocator::free(void* p) noexcept {
    if (!p) {
        return;
    }
    NOVA_ASSERT(owns(p), "pointer does not belong to this pool");
    NOVA_ASSERT(used_ > 0, "pool double free");
    auto* node = static_cast<FreeNode*>(p);
    node->next = free_list_;
    free_list_ = node;
    --used_;
}

bool PoolAllocator::owns(const void* p) const noexcept {
    const auto* b = static_cast<const std::byte*>(p);
    const std::byte* end = storage_ + block_size_ * block_count_;
    // std::less gives a total order even for unrelated pointers.
    return !std::less<const std::byte*>{}(b, storage_) && std::less<const std::byte*>{}(b, end) &&
           static_cast<usize>(b - storage_) % block_size_ == 0;
}

} // namespace nova::memory
