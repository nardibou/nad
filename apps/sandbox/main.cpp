// Rendering Core sandbox: lit 3D scene through the RHI with CPU and GPU frame timing.
//
//   nova_sandbox [--headless] [--frames N] [--width W --height H] [--no-vsync]
//                [--validation] [--screenshot out.ppm]
//
// Controls: A/D or Left/Right orbit, W/S zoom, Space pause, Escape quit.
// Exit code 77 means no usable Vulkan device (CTest treats it as "skipped").

#include "nova/core/log.hpp"
#include "nova/core/time.hpp"
#include "nova/platform/window.hpp"
#include "nova/renderer/renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using namespace nova;

namespace {

constexpr int kExitNoGpu = 77;

template <typename T>
void parse_number(const char* s, T& out) {
    std::from_chars(s, s + std::strlen(s), out);
}

bool write_ppm(const std::string& path, const std::vector<u8>& rgba, u32 w, u32 h) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (usize i = 0; i < rgba.size(); i += 4) {
        std::fwrite(&rgba[i], 1, 3, f);
    }
    return std::fclose(f) == 0;
}

struct Scene {
    std::vector<renderer::DrawItem> items;
    std::vector<f32> spin; // per-cube angular speed (rad/s); 0 for static items
};

Scene build_scene(renderer::Renderer& r) {
    const renderer::MeshId cube = r.upload(renderer::make_cube(1.0f));
    const renderer::MeshId plane = r.upload(renderer::make_plane(1.0f));
    Scene s;
    s.items.push_back({plane, math::scale({40, 1, 40}), {0.35f, 0.37f, 0.40f}});
    s.spin.push_back(0.0f);

    constexpr int kGrid = 7;
    for (int z = 0; z < kGrid; ++z) {
        for (int x = 0; x < kGrid; ++x) {
            const f32 fx = static_cast<f32>(x) / (kGrid - 1);
            const f32 fz = static_cast<f32>(z) / (kGrid - 1);
            const math::Vec3 color{0.2f + 0.8f * fx, 0.3f + 0.4f * (1.0f - fz), 0.9f - 0.7f * fx};
            s.items.push_back({cube, math::Mat4::identity(), color});
            s.spin.push_back(0.4f + 0.15f * static_cast<f32>((x + z) % 5));
        }
    }
    return s;
}

void animate(Scene& s, f32 t) {
    constexpr int kGrid = 7;
    constexpr f32 kSpacing = 2.2f;
    for (usize i = 1; i < s.items.size(); ++i) {
        const int idx = static_cast<int>(i - 1);
        const f32 x = (static_cast<f32>(idx % kGrid) - (kGrid - 1) * 0.5f) * kSpacing;
        const f32 z = (static_cast<f32>(idx / kGrid) - (kGrid - 1) * 0.5f) * kSpacing;
        const f32 y = 0.5f + 0.25f * std::sin(t * 1.5f + static_cast<f32>(idx));
        s.items[i].transform = math::translate({x, y + 0.5f, z}) * math::rotate({0.3f, 1.0f, 0.2f}, t * s.spin[i]);
    }
}

} // namespace

int main(int argc, char** argv) {
    platform::WindowDesc wdesc;
    wdesc.title = "NOVA Sandbox";
    rhi::DeviceDesc ddesc;
    u64 max_frames = 0;
    std::string screenshot;
#if defined(NOVA_ENABLE_ASSERTS)
    ddesc.validation = true;
#endif

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--headless") {
            wdesc.backend = platform::Backend::Headless;
        } else if (arg == "--frames" && has_value) {
            parse_number(argv[++i], max_frames);
        } else if (arg == "--width" && has_value) {
            parse_number(argv[++i], wdesc.width);
        } else if (arg == "--height" && has_value) {
            parse_number(argv[++i], wdesc.height);
        } else if (arg == "--no-vsync") {
            ddesc.vsync = false;
        } else if (arg == "--validation") {
            ddesc.validation = true;
        } else if (arg == "--screenshot" && has_value) {
            screenshot = argv[++i];
        }
    }
    const bool headless = wdesc.backend == platform::Backend::Headless;
    if (headless && max_frames == 0) {
        max_frames = 600; // headless has no close button
    }
    if (!screenshot.empty() && !headless) {
        NOVA_WARN("Sandbox", "--screenshot is only supported with --headless; ignoring");
        screenshot.clear();
    }

    auto window = platform::create_window(wdesc);
    if (!window) {
        NOVA_FATAL("Sandbox", "could not create a window");
        return 1;
    }
    ddesc.window = headless ? nullptr : window.get();
    ddesc.width = wdesc.width;
    ddesc.height = wdesc.height;
    auto device = rhi::Device::create(ddesc);
    if (!device) {
        NOVA_ERROR("Sandbox", "no usable Vulkan device");
        return kExitNoGpu;
    }

    renderer::Renderer renderer(*device);
    if (!renderer.valid()) {
        return 1;
    }
    Scene scene = build_scene(renderer);
    renderer::Camera camera;
    renderer::DirectionalLight light;
    platform::Input input;

    FrameStats cpu_stats;
    FrameStats gpu_stats;
    Stopwatch frame_timer;
    Stopwatch report_timer;
    f32 sim_time = 0.0f;
    f32 orbit = 0.6f;
    f32 distance = 16.0f;
    bool paused = false;
    platform::WindowSize last_size = window->size();

    for (u64 frame = 0; max_frames == 0 || frame < max_frames; ++frame) {
        const f64 frame_ms = frame_timer.lap_ms();
        if (frame > 0) {
            cpu_stats.record(frame_ms);
        }
        // Headless runs use a fixed step so screenshots are deterministic.
        const f32 dt = headless ? 1.0f / 60.0f : static_cast<f32>(std::min(frame_ms, 100.0) / 1000.0);

        input.begin_frame();
        if (!window->poll_events(input) || input.was_pressed(platform::Key::Escape)) {
            break;
        }
        if (input.was_pressed(platform::Key::Space)) {
            paused = !paused;
        }
        if (input.is_down(platform::Key::A) || input.is_down(platform::Key::Left)) orbit -= dt;
        if (input.is_down(platform::Key::D) || input.is_down(platform::Key::Right)) orbit += dt;
        if (input.is_down(platform::Key::W)) distance = std::max(4.0f, distance - 8.0f * dt);
        if (input.is_down(platform::Key::S)) distance = std::min(60.0f, distance + 8.0f * dt);

        const platform::WindowSize size = window->size();
        if (!headless && (size.width != last_size.width || size.height != last_size.height)) {
            device->resize(size.width, size.height);
            last_size = size;
        }

        if (!paused) {
            sim_time += dt;
            orbit += dt * 0.15f;
        }
        animate(scene, sim_time);
        camera.position = {std::cos(orbit) * distance, distance * 0.55f, std::sin(orbit) * distance};
        camera.target = {0, 0.5f, 0};

        renderer.render_frame(camera, light, scene.items, {0.05f, 0.07f, 0.10f});
        if (device->last_gpu_frame_ms() >= 0.0) {
            gpu_stats.record(device->last_gpu_frame_ms());
        }

        if (report_timer.elapsed_ms() >= 1000.0) {
            report_timer.reset();
            const auto c = cpu_stats.summarize();
            const auto g = gpu_stats.summarize();
            NOVA_INFO("Sandbox", "CPU frame {:.2f} ms (p99 {:.2f}) | GPU {:.2f} ms (p99 {:.2f}) | {:.0f} fps | {} draws",
                      c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms, c.avg_fps, renderer.stats().draw_calls);
        }
    }

    device->wait_idle();
    int exit_code = 0;
    if (!screenshot.empty()) {
        std::vector<u8> pixels;
        const rhi::Extent e = device->back_buffer_extent();
        if (device->read_back(pixels) && write_ppm(screenshot, pixels, e.width, e.height)) {
            NOVA_INFO("Sandbox", "wrote {}", screenshot);
        } else {
            NOVA_ERROR("Sandbox", "failed to write screenshot {}", screenshot);
            exit_code = 1;
        }
    }

    const auto c = cpu_stats.summarize();
    const auto g = gpu_stats.summarize();
    NOVA_INFO("Sandbox", "exit after {} frames: CPU avg {:.3f} ms p99 {:.3f} ms | GPU avg {:.3f} ms p99 {:.3f} ms",
              c.frame_count + 1, c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms);
    if (device->validation_error_count() > 0) {
        NOVA_ERROR("Sandbox", "{} Vulkan validation errors", device->validation_error_count());
        exit_code = 1;
    }
    return exit_code;
}
