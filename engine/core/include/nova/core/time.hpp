#pragma once

#include "nova/core/types.hpp"

#include <chrono>
#include <vector>

namespace nova {

// Monotonic high-resolution clock used for all engine timing.
class Stopwatch {
public:
    using Clock = std::chrono::steady_clock;

    Stopwatch() noexcept : start_(Clock::now()) {}
    void reset() noexcept { start_ = Clock::now(); }
    [[nodiscard]] f64 elapsed_ms() const noexcept {
        return std::chrono::duration<f64, std::milli>(Clock::now() - start_).count();
    }
    // Returns elapsed time and restarts the stopwatch.
    f64 lap_ms() noexcept {
        const auto now = Clock::now();
        const f64 ms = std::chrono::duration<f64, std::milli>(now - start_).count();
        start_ = now;
        return ms;
    }

private:
    Clock::time_point start_;
};

struct FrameStatsSummary {
    u32 frame_count = 0;
    f64 avg_ms = 0.0;
    f64 min_ms = 0.0;
    f64 max_ms = 0.0;
    f64 p99_ms = 0.0;       // 99th percentile frame time
    f64 avg_fps = 0.0;
    f64 one_percent_low_fps = 0.0; // 1000 / p99_ms
};

// Rolling window of frame times. 60 FPS => 16.67 ms budget.
class FrameStats {
public:
    explicit FrameStats(u32 window_size = 240);

    void record(f64 frame_ms);
    void clear() noexcept;
    [[nodiscard]] u32 size() const noexcept { return count_; }
    [[nodiscard]] FrameStatsSummary summarize() const;

private:
    std::vector<f64> samples_;
    u32 next_ = 0;
    u32 count_ = 0;
};

} // namespace nova
