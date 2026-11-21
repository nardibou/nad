// NOVA village demo: a procedurally built mountain village at golden hour, rendered with the
// WorldRenderer (sun shadows, HDR, aerial perspective, procedural materials, ACES + grading).
//
//   nova_village [--headless] [--frames N] [--width W --height H] [--no-vsync] [--validation]
//                [--screenshot out.ppm]
//                [--model file.glb|.gltf [--model-at X Z] [--model-scale S] [--model-yaw DEG]]
//
// Window controls: hold right mouse to look, WASD move, Q/E down/up, Shift faster, Esc quit.
// Exit code 77 means no usable Vulkan device.

#include "nova/core/log.hpp"
#include "nova/core/time.hpp"
#include "nova/platform/window.hpp"
#include "nova/renderer/model.hpp"
#include "nova/renderer/png_image.hpp"
#include "nova/renderer/world_renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include "village_scene.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using namespace nova;
using math::Mat4;
using math::Vec3;
using math::Vec4;
using renderer::Material;
using renderer::MeshData;

#if defined(_WIN32)
// Ask hybrid-graphics laptop drivers to run this process on the dedicated NVIDIA / AMD GPU.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

constexpr int kExitNoGpu = 77;


template <typename T>
void parse_number(const char* s, T& out) {
    std::from_chars(s, s + std::strlen(s), out);
}

bool write_ppm(const std::string& path, const std::vector<u8>& rgba, u32 w, u32 h) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".png") {
        return image::save_png(path, rgba.data(), w, h);
    }
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

} // namespace

int main(int argc, char** argv) {
    platform::WindowDesc wdesc;
    wdesc.title = "NOVA Village (2K)";
    wdesc.width = 2560;
    wdesc.height = 1440;
    rhi::DeviceDesc ddesc;
    u64 max_frames = 0;
    std::string screenshot;
    std::string model_path;
    f32 model_x = 0.4f, model_z = 5.0f, model_scale = 1.0f, model_yaw = 0.0f;
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
        } else if (arg == "--model" && has_value) {
            model_path = argv[++i];
        } else if (arg == "--model-at" && i + 2 < argc) {
            model_x = std::strtof(argv[++i], nullptr);
            model_z = std::strtof(argv[++i], nullptr);
        } else if (arg == "--model-scale" && has_value) {
            model_scale = std::strtof(argv[++i], nullptr);
        } else if (arg == "--model-yaw" && has_value) {
            model_yaw = std::strtof(argv[++i], nullptr);
        } else if (arg == "--screenshot" && has_value) {
            screenshot = argv[++i];
        }
    }
    const bool headless = wdesc.backend == platform::Backend::Headless;
    if (headless && max_frames == 0) {
        max_frames = 60;
    }

    auto window = platform::create_window(wdesc);
    if (!window) {
        NOVA_FATAL("Village", "could not create a window");
        return 1;
    }
    ddesc.window = headless ? nullptr : window.get();
    ddesc.width = wdesc.width;
    ddesc.height = wdesc.height;
    auto device = rhi::Device::create(ddesc);
    if (!device) {
        NOVA_ERROR("Village", "no usable Vulkan device");
        return kExitNoGpu;
    }
    // Hardware GPUs get a 4K shadow map (crisper, longer shadows); software rasterisers stay at 2K.
    renderer::WorldRenderer world(*device, device->info().software ? 2048u : 4096u);
    if (!world.valid()) {
        return 1;
    }

    Stopwatch build_timer;
    const village::Scene builder = village::build_scene();
    const village::VillageMaterials mats = village::upload_village_materials(world);
    const MeshData& terrain = builder.terrain;
    std::vector<renderer::WorldDrawItem> items = {
        {world.upload(terrain)},
        {world.upload(builder.opaque)},
        {world.upload(builder.foliage)},
        {world.upload(builder.two_sided), Mat4::identity(), true},
        {world.upload(builder.water), Mat4::identity(), false, false},
        {world.upload(builder.stone_walls), Mat4::identity(), false, true, mats.stone_wall},
        {world.upload(builder.dirt_road), Mat4::identity(), false, false, mats.dirt_road},
        {world.upload(builder.mud_patches), Mat4::identity(), false, false, mats.mud_patches},
        {world.upload(builder.forest_floor), Mat4::identity(), false, false, mats.forest_floor},
        {world.upload(builder.wood_props), Mat4::identity(), false, true, mats.wood_props},
        {world.upload(builder.tree_bark), Mat4::identity(), false, true, mats.tree_bark},
    };
    usize tris = (terrain.indices.size() + builder.opaque.indices.size() + builder.foliage.indices.size() +
                  builder.two_sided.indices.size() + builder.water.indices.size() +
                  builder.stone_walls.indices.size() + builder.dirt_road.indices.size() +
                  builder.mud_patches.indices.size() + builder.forest_floor.indices.size() +
                  builder.wood_props.indices.size() + builder.tree_bark.indices.size()) / 3;
    if (!model_path.empty()) {
        // Imported glTF asset, standing on the terrain at (model_x, model_z).
        std::string error;
        const auto model = renderer::load_gltf(model_path, &error);
        if (!model) {
            NOVA_ERROR("Village", "could not load {}: {}", model_path, error);
            return 1;
        }
        const Vec3 ground{model_x, village::terrain_height(model_x, model_z), model_z};
        const Mat4 placement = math::translate(ground) * math::rotate({0, 1, 0}, math::radians(model_yaw)) *
                               math::scale({model_scale, model_scale, model_scale}) *
                               math::translate({0, -model->bounds_min.y, 0});
        for (const renderer::WorldDrawItem& item : world.upload_model(*model, placement)) {
            items.push_back(item);
        }
        for (const renderer::ModelPrimitive& p : model->primitives) {
            tris += p.mesh.indices.size() / 3;
        }
        NOVA_INFO("Village", "loaded {}: {} primitives, {} materials, {} images", model_path,
                  model->primitives.size(), model->materials.size(), model->images.size());
    }
    NOVA_INFO("Village", "built scene in {:.0f} ms: {} triangles in {} draws", build_timer.elapsed_ms(), tris,
              items.size());

    renderer::Camera camera;
    camera.position = {1.1f, 1.7f, 8.5f};
    camera.fov_y = math::radians(62.0f);
    camera.z_far = 2000.0f;
    f32 yaw = math::radians(-93.5f);  // looking down the street (-Z)
    f32 pitch = math::radians(1.5f);
    renderer::WorldSettings settings;
    settings.shadow_center = {0, 0, -12};
    settings.shadow_radius = 42.0f;

    platform::Input input;
    FrameStats cpu_stats;
    FrameStats gpu_stats;
    Stopwatch frame_timer;
    Stopwatch report_timer;
    f32 time = 3.0f;

    for (u64 frame = 0; max_frames == 0 || frame < max_frames; ++frame) {
        const f64 frame_ms = frame_timer.lap_ms();
        if (frame > 0) {
            cpu_stats.record(frame_ms);
        }
        const f32 dt = headless ? 1.0f / 60.0f : static_cast<f32>(std::min(frame_ms, 100.0) / 1000.0);
        time += dt;

        input.begin_frame();
        if (!window->poll_events(input) || input.was_pressed(platform::Key::Escape)) {
            break;
        }
        if (input.is_down(platform::MouseButton::Right)) {
            yaw += input.mouse_dx() * 0.003f;
            pitch = std::clamp(pitch - input.mouse_dy() * 0.003f, -1.4f, 1.4f);
        }
        const Vec3 forward{std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw)};
        const Vec3 right = math::normalize(math::cross(forward, {0, 1, 0}));
        const f32 speed = (input.is_down(platform::Key::LeftShift) ? 20.0f : 5.0f) * dt;
        if (input.is_down(platform::Key::W)) camera.position += forward * speed;
        if (input.is_down(platform::Key::S)) camera.position -= forward * speed;
        if (input.is_down(platform::Key::D)) camera.position += right * speed;
        if (input.is_down(platform::Key::A)) camera.position -= right * speed;
        if (input.is_down(platform::Key::E)) camera.position.y += speed;
        if (input.is_down(platform::Key::Q)) camera.position.y -= speed;
        camera.target = camera.position + forward;

        const platform::WindowSize size = window->size();
        if (!headless && (size.width != device->back_buffer_extent().width ||
                          size.height != device->back_buffer_extent().height)) {
            device->resize(size.width, size.height);
        }

        settings.time = time;
        world.render_frame(camera, settings, items);
        if (device->last_gpu_frame_ms() >= 0.0) {
            gpu_stats.record(device->last_gpu_frame_ms());
        }
        if (report_timer.elapsed_ms() >= 1000.0) {
            report_timer.reset();
            const auto c = cpu_stats.summarize();
            const auto g = gpu_stats.summarize();
            NOVA_INFO("Village", "CPU frame {:.2f} ms (p99 {:.2f}) | GPU {:.2f} ms (p99 {:.2f}) | {:.0f} fps",
                      c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms, c.avg_fps);
        }
    }

    device->wait_idle();
    int exit_code = 0;
    if (!screenshot.empty()) {
        std::vector<u8> pixels;
        const rhi::Extent e = device->back_buffer_extent();
        if (headless && device->read_back(pixels) && write_ppm(screenshot, pixels, e.width, e.height)) {
            NOVA_INFO("Village", "wrote {}", screenshot);
        } else {
            NOVA_ERROR("Village", "failed to write screenshot {} (requires --headless)", screenshot);
            exit_code = 1;
        }
    }
    const auto c = cpu_stats.summarize();
    const auto g = gpu_stats.summarize();
    NOVA_INFO("Village", "exit: CPU avg {:.2f} ms p99 {:.2f} ms | GPU avg {:.2f} ms p99 {:.2f} ms | {} draws, {} tris",
              c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms, world.stats().draw_calls, world.stats().triangles);
    if (device->validation_error_count() > 0) {
        NOVA_ERROR("Village", "{} Vulkan validation errors", device->validation_error_count());
        exit_code = 1;
    }
    return exit_code;
}
