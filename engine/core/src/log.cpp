#include "nova/core/log.hpp"

#include <algorithm>
#include <cstdio>
#include <atomic>
#include <mutex>
#include <vector>

namespace nova::log {
namespace {

struct SinkEntry {
    SinkId id;
    Sink sink;
};

struct Registry {
    std::mutex mutex;
    std::vector<SinkEntry> sinks;
    SinkId next_id = 1;
    SinkId console_id = 0;
};

void console_sink(const Record& r) {
    std::FILE* out = r.level >= Level::Warn ? stderr : stdout;
    std::fprintf(out, "[%-5.*s] [%.*s] %.*s\n", static_cast<int>(to_string(r.level).size()),
                 to_string(r.level).data(), static_cast<int>(r.channel.size()), r.channel.data(),
                 static_cast<int>(r.message.size()), r.message.data());
    if (r.level >= Level::Error) {
        std::fflush(out);
    }
}

Registry& registry() {
    static Registry* reg = [] {
        auto* r = new Registry(); // intentionally leaked: usable during static destruction
        r->console_id = r->next_id++;
        r->sinks.push_back({r->console_id, &console_sink});
        return r;
    }();
    return *reg;
}

#if defined(NDEBUG)
std::atomic<Level> g_level{Level::Info};
#else
std::atomic<Level> g_level{Level::Debug};
#endif

} // namespace

std::string_view to_string(Level level) noexcept {
    switch (level) {
    case Level::Trace: return "trace";
    case Level::Debug: return "debug";
    case Level::Info: return "info";
    case Level::Warn: return "warn";
    case Level::Error: return "error";
    case Level::Fatal: return "fatal";
    case Level::Off: return "off";
    }
    return "?";
}

SinkId add_sink(Sink sink) {
    auto& reg = registry();
    std::scoped_lock lock(reg.mutex);
    const SinkId id = reg.next_id++;
    reg.sinks.push_back({id, std::move(sink)});
    return id;
}

void remove_sink(SinkId id) {
    auto& reg = registry();
    std::scoped_lock lock(reg.mutex);
    std::erase_if(reg.sinks, [id](const SinkEntry& e) { return e.id == id; });
    if (id == reg.console_id) {
        reg.console_id = 0;
    }
}

void set_console_enabled(bool enabled) {
    auto& reg = registry();
    std::scoped_lock lock(reg.mutex);
    const bool present = reg.console_id != 0;
    if (enabled && !present) {
        reg.console_id = reg.next_id++;
        reg.sinks.push_back({reg.console_id, &console_sink});
    } else if (!enabled && present) {
        const SinkId id = reg.console_id;
        std::erase_if(reg.sinks, [id](const SinkEntry& e) { return e.id == id; });
        reg.console_id = 0;
    }
}

void set_level(Level level) noexcept { g_level.store(level, std::memory_order_relaxed); }
Level level() noexcept { return g_level.load(std::memory_order_relaxed); }

void write(Level lvl, std::string_view channel, std::string_view message, std::source_location location) {
    if (!enabled(lvl)) {
        return;
    }
    const Record record{lvl, channel, message, location};
    auto& reg = registry();
    std::scoped_lock lock(reg.mutex);
    for (const auto& entry : reg.sinks) {
        entry.sink(record);
    }
}

} // namespace nova::log
