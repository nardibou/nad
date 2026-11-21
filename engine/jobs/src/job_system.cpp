#include "nova/jobs/job_system.hpp"

#include "nova/core/assert.hpp"
#include "nova/core/log.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace nova::jobs {

struct JobSystem::Impl {
    struct Task {
        Job job;
        WaitGroup* group;
    };

    std::mutex mutex;
    std::condition_variable work_cv; // signalled when work arrives or on shutdown
    std::condition_variable done_cv; // signalled when any group may have completed
    std::deque<Task> queue;
    std::vector<std::thread> workers;
    bool stopping = false;

    // Pops one task; caller must hold `lock`. Returns false if the queue is empty.
    bool try_pop(Task& out) {
        if (queue.empty()) {
            return false;
        }
        out = std::move(queue.front());
        queue.pop_front();
        return true;
    }

    void run(Task& task) {
        task.job();
        if (task.group && task.group->pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Lock so a waiter cannot miss the notification between its check and its wait.
            std::scoped_lock lock(mutex);
            done_cv.notify_all();
        }
    }

    void worker_loop() {
        for (;;) {
            Task task;
            {
                std::unique_lock lock(mutex);
                work_cv.wait(lock, [&] { return stopping || !queue.empty(); });
                if (!try_pop(task)) {
                    return; // stopping and drained
                }
            }
            run(task);
        }
    }
};

JobSystem::JobSystem(u32 worker_count) : impl_(std::make_unique<Impl>()) {
    if (worker_count == 0) {
        const u32 hw = std::thread::hardware_concurrency();
        worker_count = std::max<u32>(1, hw > 1 ? hw - 1 : 1);
    }
    impl_->workers.reserve(worker_count);
    for (u32 i = 0; i < worker_count; ++i) {
        impl_->workers.emplace_back([impl = impl_.get()] { impl->worker_loop(); });
    }
    NOVA_DEBUG("Jobs", "started {} worker threads", worker_count);
}

JobSystem::~JobSystem() {
    {
        std::scoped_lock lock(impl_->mutex);
        impl_->stopping = true;
    }
    impl_->work_cv.notify_all();
    for (auto& t : impl_->workers) {
        t.join();
    }
}

void JobSystem::submit(Job job, WaitGroup* group) {
    NOVA_ASSERT(job, "cannot submit an empty job");
    if (group) {
        group->pending_.fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::scoped_lock lock(impl_->mutex);
        NOVA_ASSERT(!impl_->stopping, "submit after shutdown");
        impl_->queue.push_back({std::move(job), group});
    }
    impl_->work_cv.notify_one();
    // Threads blocked in wait() can help too; this matters when every worker is waiting.
    impl_->done_cv.notify_all();
}

void JobSystem::wait(WaitGroup& group) {
    while (!group.done()) {
        Impl::Task task;
        {
            std::unique_lock lock(impl_->mutex);
            if (!impl_->try_pop(task)) {
                // Nothing to help with: sleep until a group completes or new work arrives.
                impl_->done_cv.wait(lock, [&] { return group.done() || !impl_->queue.empty(); });
                continue;
            }
        }
        impl_->run(task);
    }
}

void JobSystem::parallel_for(u32 count, u32 batch_size, const std::function<void(u32, u32)>& fn) {
    if (count == 0) {
        return;
    }
    batch_size = std::max<u32>(1, batch_size);
    WaitGroup group;
    for (u32 begin = 0; begin < count; begin += batch_size) {
        const u32 end = std::min(count, begin + batch_size);
        submit([&fn, begin, end] { fn(begin, end); }, &group);
    }
    wait(group);
}

u32 JobSystem::worker_count() const noexcept { return static_cast<u32>(impl_->workers.size()); }

} // namespace nova::jobs
