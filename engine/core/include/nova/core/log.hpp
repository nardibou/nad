#pragma once

#include "nova/core/types.hpp"

#include <format>
#include <functional>
#include <source_location>
#include <string_view>

namespace nova::log {

enum class Level : u8 { Trace, Debug, Info, Warn, Error, Fatal, Off };

[[nodiscard]] std::string_view to_string(Level level) noexcept;

struct Record {
    Level level;
    std::string_view channel;
    std::string_view message;
    std::source_location location;
};

// Sinks may be invoked from any thread but never concurrently (the logger serializes them).
using Sink = std::function<void(const Record&)>;
using SinkId = u32;

SinkId add_sink(Sink sink);
void remove_sink(SinkId id);
// Adds/removes the default stderr sink, which is installed at startup.
void set_console_enabled(bool enabled);

void set_level(Level level) noexcept;
[[nodiscard]] Level level() noexcept;
[[nodiscard]] inline bool enabled(Level l) noexcept { return l >= level() && l != Level::Off; }

void write(Level level, std::string_view channel, std::string_view message,
           std::source_location location = std::source_location::current());

template <typename... Args>
void writef(Level level, std::string_view channel, std::source_location location,
            std::format_string<Args...> fmt, Args&&... args) {
    if (!enabled(level)) {
        return;
    }
    write(level, channel, std::format(fmt, std::forward<Args>(args)...), location);
}

} // namespace nova::log

#define NOVA_LOG(level, channel, ...)                                                                        \
    ::nova::log::writef((level), (channel), std::source_location::current(), __VA_ARGS__)
#define NOVA_TRACE(channel, ...) NOVA_LOG(::nova::log::Level::Trace, channel, __VA_ARGS__)
#define NOVA_DEBUG(channel, ...) NOVA_LOG(::nova::log::Level::Debug, channel, __VA_ARGS__)
#define NOVA_INFO(channel, ...) NOVA_LOG(::nova::log::Level::Info, channel, __VA_ARGS__)
#define NOVA_WARN(channel, ...) NOVA_LOG(::nova::log::Level::Warn, channel, __VA_ARGS__)
#define NOVA_ERROR(channel, ...) NOVA_LOG(::nova::log::Level::Error, channel, __VA_ARGS__)
#define NOVA_FATAL(channel, ...) NOVA_LOG(::nova::log::Level::Fatal, channel, __VA_ARGS__)
