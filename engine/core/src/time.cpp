#include "nova/core/assert.hpp"
#include "nova/core/time.hpp"

#include <algorithm>

namespace nova {

FrameStats::FrameStats(u32 window_size) : samples_(window_size, 0.0) {
    NOVA_VERIFY(window_size > 0, "FrameStats window must be non-empty");
}

void FrameStats::record(f64 frame_ms) {
    samples_[next_] = frame_ms;
    next_ = (next_ + 1) % static_cast<u32>(samples_.size());
    count_ = std::min<u32>(count_ + 1, static_cast<u32>(samples_.size()));
}

void FrameStats::clear() noexcept {
    next_ = 0;
    count_ = 0;
}

FrameStatsSummary FrameStats::summarize() const {
    FrameStatsSummary s;
    if (count_ == 0) {
        return s;
    }
    // While the ring buffer is filling, valid samples occupy [0, count_).
    std::vector<f64> sorted(samples_.begin(), samples_.begin() + count_);
    std::ranges::sort(sorted);

    f64 total = 0.0;
    for (f64 v : sorted) {
        total += v;
    }
    s.frame_count = count_;
    s.avg_ms = total / count_;
    s.min_ms = sorted.front();
    s.max_ms = sorted.back();
    // Nearest-rank percentile.
    const auto rank = static_cast<usize>((99 * static_cast<u64>(count_) + 99) / 100);
    s.p99_ms = sorted[std::clamp<usize>(rank, 1, count_) - 1];
    s.avg_fps = s.avg_ms > 0.0 ? 1000.0 / s.avg_ms : 0.0;
    s.one_percent_low_fps = s.p99_ms > 0.0 ? 1000.0 / s.p99_ms : 0.0;
    return s;
}

} // namespace nova
