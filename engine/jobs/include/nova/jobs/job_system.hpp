#pragma once

#include "nova/core/types.hpp"

#include <atomic>
#include <functional>
#include <memory>

namespace nova::jobs {

using Job = std::function<void()>;

// Tracks completion of a group of jobs. Must outlive every job submitted against it.
class WaitGroup {
public:
    WaitGroup() = default;
    WaitGroup(const WaitGroup&) = delete;
    WaitGroup& operator=(const WaitGroup&) = delete;

    [[nodiscard]] bool done() const noexcept { return pending_.load(std::memory_order_acquire) == 0; }
    [[nodiscard]] u32 pending() const noexcept { return pending_.load(std::memory_order_acquire); }

private:
    friend class JobSystem;
    std::atomic<u32> pending_{0};
};

// Fixed pool of worker threads fed from a shared queue.
// wait() lets the calling thread run queued jobs, so jobs may submit and wait on nested
// work without deadlocking. Jobs must not throw (an escaping exception terminates).
class JobSystem {
public:
    // worker_count == 0 picks hardware_concurrency() - 1 (at least 1).
    explicit JobSystem(u32 worker_count = 0);
    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;
    // Drains every queued job, then joins the workers.
    ~JobSystem();

    void submit(Job job, WaitGroup* group = nullptr);
    void wait(WaitGroup& group);

    // Splits [0, count) into batches of up to `batch_size` and runs fn(begin, end) for each,
    // blocking until all batches finish.
    void parallel_for(u32 count, u32 batch_size, const std::function<void(u32 begin, u32 end)>& fn);

    [[nodiscard]] u32 worker_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nova::jobs
