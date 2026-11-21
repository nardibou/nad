// NOVA "Guerrilla" demo game: first-person walk through the mountain village at golden hour with
// an AK rifle, shooting steel popper targets. Built on WorldRenderer (shadows, contact AO, SSAO,
// colour bleed, bloom, FXAA) plus the PBR asset path for the hands and rifle.
//
//   nova_guerrilla [--quality ultra|high|medium|low] [--render-scale S] [--width W --height H] [--no-vsync]
//                  [--viewmodel file.glb | --procedural-viewmodel] [--validation]
//                  [--headless --frames N --screenshot out.ppm [--ads] [--fire] [--reload]]
//
// Controls: mouse look, WASD move, Shift sprint, Space jump, LMB fire (full auto), RMB aim down
// sights, R reload, Esc quit. Exit code 77 means no usable Vulkan device.

#include "nova/core/log.hpp"
#include "nova/core/time.hpp"
#include "nova/platform/window.hpp"
#include "nova/renderer/png_image.hpp"
#include "nova/renderer/ui_overlay.hpp"
#include "nova/renderer/world_renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include "nova_craft.hpp"
#include "rig_viewmodel.hpp"
#include "viewmodel.hpp"
#include "village_scene.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace nova;
using math::Mat4;
using math::Vec3;
using math::Vec4;
using renderer::MeshData;
using renderer::WorldDrawItem;

#if defined(_WIN32)
// Ask hybrid-graphics laptop drivers to run this process on the dedicated NVIDIA / AMD GPU.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

constexpr int kExitNoGpu = 77;

// ------------------------------------------------------------------ small helpers

struct Rng {
    u32 state;
    explicit Rng(u32 seed) : state(seed) {}
    f32 uniform() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<f32>(state & 0xFFFFFF) / static_cast<f32>(0x1000000);
    }
    f32 range(f32 a, f32 b) { return a + (b - a) * uniform(); }
};

f32 approach(f32 current, f32 target, f32 rate, f32 dt) {
    return current + (target - current) * (1.0f - std::exp(-rate * dt));
}
Vec3 approach(const Vec3& current, const Vec3& target, f32 rate, f32 dt) {
    return math::lerp(current, target, 1.0f - std::exp(-rate * dt));
}
f32 smooth01(f32 x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// Columns = basis vectors, i.e. local-to-world for an object with that orientation.
Mat4 basis(const Vec3& x, const Vec3& y, const Vec3& z, const Vec3& origin) {
    Mat4 m = Mat4::identity();
    const Vec3 cols[4] = {x, y, z, origin};
    for (int c = 0; c < 4; ++c) {
        m.m[c][0] = cols[c].x;
        m.m[c][1] = cols[c].y;
        m.m[c][2] = cols[c].z;
    }
    return m;
}

Vec3 rotate_y(const Vec3& v, f32 angle) {
    const f32 c = std::cos(angle), s = std::sin(angle);
    return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z}; // same sense as math::rotate({0,1,0}, angle)
}

// ------------------------------------------------------------------ collision and ray casts

struct Hit {
    f32 t = 1e9f;
    Vec3 normal{0, 1, 0};
    enum class Kind : u8 { None, Terrain, Wall, Target } kind = Kind::None;
    int target = -1;
};

// Slab test in the collider's local frame.
bool ray_obb(const Vec3& o, const Vec3& d, const village::Collider& c, f32& t_out, Vec3& n_out) {
    const Vec3 lo = rotate_y(o - c.center, -c.yaw);
    const Vec3 ld = rotate_y(d, -c.yaw);
    const f32 lo_a[3] = {lo.x, lo.y, lo.z}, ld_a[3] = {ld.x, ld.y, ld.z};
    const f32 h[3] = {c.half_extents.x, c.half_extents.y, c.half_extents.z};
    f32 t0 = 0.0f, t1 = 1e9f;
    int axis = -1;
    f32 sign = 1.0f;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(ld_a[i]) < 1e-8f) {
            if (std::abs(lo_a[i]) > h[i]) {
                return false;
            }
            continue;
        }
        const f32 inv = 1.0f / ld_a[i];
        f32 ta = (-h[i] - lo_a[i]) * inv, tb = (h[i] - lo_a[i]) * inv;
        f32 s = -1.0f;
        if (ta > tb) {
            std::swap(ta, tb);
            s = 1.0f;
        }
        if (ta > t0) {
            t0 = ta;
            axis = i;
            sign = s;
        }
        t1 = std::min(t1, tb);
        if (t0 > t1) {
            return false;
        }
    }
    if (axis < 0) {
        return false; // ray starts inside
    }
    Vec3 ln{0, 0, 0};
    (axis == 0 ? ln.x : axis == 1 ? ln.y : ln.z) = sign;
    t_out = t0;
    n_out = rotate_y(ln, c.yaw);
    return true;
}

bool ray_terrain(const Vec3& o, const Vec3& d, f32 max_t, f32& t_out) {
    constexpr f32 kStep = 0.2f;
    f32 prev_t = 0.0f;
    for (f32 t = kStep; t <= max_t; t += kStep * (1.0f + t * 0.02f)) {
        const Vec3 p = o + d * t;
        if (p.y < village::terrain_height(p.x, p.z)) {
            f32 a = prev_t, b = t;
            for (int i = 0; i < 12; ++i) {
                const f32 m = 0.5f * (a + b);
                const Vec3 q = o + d * m;
                (q.y < village::terrain_height(q.x, q.z) ? b : a) = m;
            }
            t_out = b;
            return true;
        }
        prev_t = t;
    }
    return false;
}

Vec3 terrain_normal(f32 x, f32 z) {
    constexpr f32 e = 0.1f;
    return math::normalize(Vec3{village::terrain_height(x - e, z) - village::terrain_height(x + e, z), 2.0f * e,
                            village::terrain_height(x, z - e) - village::terrain_height(x, z + e)});
}

// ------------------------------------------------------------------ steel popper targets

struct Target {
    Vec3 base;        // ground point under the post
    f32 yaw = 0.0f;   // plate faces +Z rotated by yaw (towards the shooter)
    f32 angle = 0.0f; // knock-down rotation about the base hinge (0 = upright)
    f32 velocity = 0.0f;
    bool down = false;
};

constexpr f32 kPlateRadius = 0.28f;
constexpr f32 kPlateHeight = 1.15f; // plate centre above the hinge

Mat4 target_frame(const Target& t) {
    return math::translate(t.base) * math::rotate({0, 1, 0}, t.yaw) * math::rotate({1, 0, 0}, -t.angle);
}

bool ray_target(const Vec3& o, const Vec3& d, const Target& t, f32& t_out, Vec3& n_out) {
    if (t.down) {
        return false;
    }
    const Mat4 f = target_frame(t);
    const Vec3 c = math::transform_point(f, {0, kPlateHeight, 0});
    const Vec3 n = math::normalize(math::transform_normal(f, {0, 0, 1}));
    const f32 denom = math::dot(d, n);
    if (std::abs(denom) < 1e-5f) {
        return false;
    }
    const f32 tt = math::dot(c - o, n) / denom;
    const Vec3 p = o + d * tt;
    if (tt <= 0.0f || math::length(p - c) > kPlateRadius) {
        return false;
    }
    t_out = tt;
    n_out = denom < 0.0f ? n : -n;
    return true;
}

// ------------------------------------------------------------------ debris and sparks

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 spin_axis{0, 1, 0};
    f32 spin = 0.0f;
    f32 size = 0.02f;
    f32 life = 0.0f;
    f32 max_life = 1.0f;
    bool spark = false;
};

// ------------------------------------------------------------------ command line

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

struct Quality {
    f32 render_scale;
    u32 shadow_map;
    f32 ssao;
    f32 bounce;
};

std::optional<Quality> quality_preset(std::string_view name, bool software) {
    // Native 2K (2560x1440) at 1.0 render_scale with 4K shadows.
    if (name == "ultra") return Quality{software ? 1.0f : 1.0f, 4096, 0.85f, 0.5f};
    if (name == "high") return Quality{1.0f, 4096, 0.8f, 0.45f};
    if (name == "medium") return Quality{1.0f, 2048, 0.6f, 0.3f};
    if (name == "low") return Quality{0.75f, 2048, 0.0f, 0.0f};
    return std::nullopt;
}

} // namespace

int main(int argc, char** argv) {
    platform::WindowDesc wdesc;
    wdesc.title = "NOVA Guerrilla (2K)";
    wdesc.width = 2560;
    wdesc.height = 1440;
    rhi::DeviceDesc ddesc;
    u64 max_frames = 0;
    std::string screenshot;
    std::string quality_name = "ultra";
    std::string viewmodel_path;
    std::string drop_model_path;
    bool procedural_viewmodel = false;
    f32 render_scale_override = 0.0f;
    bool script_ads = false;
    bool script_fire = false;
    bool script_reload = false;
    bool force_editor = false;
    bool force_play = false;
    bool start_new_scene = false;
    bool start_clay_scene = false;
    bool start_hide_ui = false;
    bool force_village = false;
    bool has_cam_override = false;
    Vec3 cam_pos_override{0.5f, 0.0f, 9.0f};
    f32 cam_yaw_deg = -90.0f;
    f32 cam_pitch_deg = 0.5f;
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
        } else if (arg == "--render-scale" && has_value) {
            parse_number(argv[++i], render_scale_override);
        } else if (arg == "--viewmodel" && has_value) {
            viewmodel_path = argv[++i];
        } else if (arg == "--drop-model" && has_value) {
            drop_model_path = argv[++i];
        } else if (arg == "--procedural-viewmodel") {
            procedural_viewmodel = true;
        } else if (arg == "--quality" && has_value) {
            quality_name = argv[++i];
        } else if (arg == "--screenshot" && has_value) {
            screenshot = argv[++i];
        } else if (arg == "--ads") {
            script_ads = true;
        } else if (arg == "--reload") {
            script_reload = true;
        } else if (arg == "--fire") {
            script_fire = true;
        } else if (arg == "--editor") {
            force_editor = true;
        } else if (arg == "--play") {
            force_play = true;
        } else if (arg == "--new-scene") {
            start_new_scene = true;
        } else if (arg == "--clay-test") {
            start_clay_scene = true;
        } else if (arg == "--hide-ui") {
            start_hide_ui = true;
        } else if (arg == "--village") {
            force_village = true;
        } else if (arg == "--cam" && i + 5 < argc) {
            parse_number(argv[++i], cam_pos_override.x);
            parse_number(argv[++i], cam_pos_override.y);
            parse_number(argv[++i], cam_pos_override.z);
            parse_number(argv[++i], cam_yaw_deg);
            parse_number(argv[++i], cam_pitch_deg);
            has_cam_override = true;
        }
    }
    const bool headless = wdesc.backend == platform::Backend::Headless;
    if (headless && max_frames == 0) {
        max_frames = 90;
    }

    auto window = platform::create_window(wdesc);
    if (!window) {
        NOVA_FATAL("Guerrilla", "could not create a window");
        return 1;
    }
    ddesc.window = headless ? nullptr : window.get();
    ddesc.width = wdesc.width;
    ddesc.height = wdesc.height;
    auto device = rhi::Device::create(ddesc);
    if (!device) {
        NOVA_ERROR("Guerrilla", "no usable Vulkan device");
        return kExitNoGpu;
    }
    const auto quality = quality_preset(quality_name, device->info().software);
    if (!quality) {
        NOVA_ERROR("Guerrilla", "unknown --quality '{}' (ultra, high, medium, low)", quality_name);
        return 1;
    }
    renderer::WorldRenderer world(*device, quality->shadow_map);
    if (!world.valid()) {
        return 1;
    }

    // ---- level & NOVA Craft Studio
    Stopwatch build_timer;
    const village::Scene scene = village::build_scene();
    const village::VillageMaterials mats = village::upload_village_materials(world);
    const renderer::MeshId wall_mesh   = world.upload(scene.stone_walls);
    const renderer::MeshId dirt_mesh   = world.upload(scene.dirt_road);
    const renderer::MeshId mud_mesh    = world.upload(scene.mud_patches);
    const renderer::MeshId forest_mesh = world.upload(scene.forest_floor);
    const renderer::MeshId wood_mesh   = world.upload(scene.wood_props);
    const renderer::MeshId bark_mesh   = world.upload(scene.tree_bark);

    // Base procedural village layers; the 6 PBR layers + dropped prefabs are live-editable inside studio.entities!
    std::vector<WorldDrawItem> level = {
        {world.upload(scene.terrain)},
        {world.upload(scene.opaque)},
        {world.upload(scene.foliage)},
        {world.upload(scene.two_sided), Mat4::identity(), true},
        {world.upload(scene.water), Mat4::identity(), false, false},
    };

    renderer::UiOverlay ui_overlay(*device);
    craft::Studio studio;
    studio.init(world, mats, wall_mesh, dirt_mesh, mud_mesh, forest_mesh, wood_mesh, bark_mesh);
    // Interactive desktop opens straight into the Architectural Clay GI Lighting Benchmark in NOVA PRISM Studio,
    // with 1-click +NEW and VILLAGE buttons in the top bar!
    studio.editor_mode = force_editor ? true : (force_play ? false : !headless);
    if (start_clay_scene || (!headless && !start_new_scene && !force_village && !force_play)) {
        studio.load_clay_gi_scene(world);
    } else if (start_new_scene) {
        studio.create_new_scene(world, true);
    }
    if (start_hide_ui) {
        studio.ui_visible = false;
    }
    if (!drop_model_path.empty()) {
        const f32 gy = studio.ground_height_at(0.0f, 0.0f);
        studio.import_gltf_model(drop_model_path, world, {0.0f, gy + 0.01f, 0.0f}, 2.4f);
    }
    if (has_cam_override) {
        studio.requested_cam_pos = {cam_pos_override.x,
                                    studio.ground_height_at(cam_pos_override.x, cam_pos_override.z) + cam_pos_override.y,
                                    cam_pos_override.z};
        studio.requested_cam_yaw = math::radians(cam_yaw_deg);
        studio.requested_cam_pitch = math::radians(cam_pitch_deg);
        studio.camera_reset_requested = true;
    }

    // ---- targets: painted steel poppers on wooden posts down the street and on the slope
    std::vector<Target> targets;
    for (const auto& [x, z] : std::initializer_list<std::pair<f32, f32>>{
             {-1.0f, -6.0f}, {1.4f, -11.0f}, {-0.6f, -17.5f}, {1.8f, -24.0f}, {-1.6f, -30.0f}, {0.4f, -36.0f},
             {-14.0f, -58.0f}, {9.0f, -66.0f}}) {
        Target t;
        t.base = {x, village::terrain_height(x, z), z};
        t.yaw = std::atan2(0.5f - x, 9.0f - z); // face the spawn point
        targets.push_back(t);
    }
    renderer::PbrMaterial steel;
    steel.base_color_factor = {0.80f, 0.78f, 0.70f, 1.0f};
    steel.metallic = 0.15f;
    steel.roughness = 0.55f;
    const renderer::MaterialId steel_mat = world.add_material(steel);
    renderer::PbrMaterial post_wood;
    post_wood.base_color_factor = {0.30f, 0.20f, 0.12f, 1.0f};
    post_wood.metallic = 0.0f;
    post_wood.roughness = 0.8f;
    const renderer::MaterialId post_mat = world.add_material(post_wood);
    MeshData plate_mesh;
    renderer::append(plate_mesh, renderer::make_cylinder(28, kPlateRadius, kPlateRadius, 0.02f),
                     math::translate({0, kPlateHeight, -0.01f}) * math::rotate({1, 0, 0}, math::kPi * 0.5f));
    renderer::append(plate_mesh, renderer::make_cube(1.0f),
                     math::translate({0, (kPlateHeight - kPlateRadius) * 0.5f, -0.02f}) *
                         math::scale({0.09f, kPlateHeight - kPlateRadius, 0.02f}));
    MeshData post_mesh;
    renderer::append(post_mesh, renderer::make_cube(1.0f),
                     math::translate({0, 0.03f, 0}) * math::scale({0.5f, 0.06f, 0.35f}));
    const renderer::MeshId plate_id = world.upload(plate_mesh);
    const renderer::MeshId base_id = world.upload(post_mesh);

    // ---- particles share one small mesh
    MeshData chip_mesh = renderer::make_cube(1.0f);
    const renderer::MeshId chip_id = world.upload(chip_mesh);
    renderer::PbrMaterial dust;
    dust.base_color_factor = {0.45f, 0.36f, 0.26f, 1.0f};
    dust.metallic = 0.0f;
    const renderer::MaterialId dust_mat = world.add_material(dust);
    renderer::PbrMaterial spark;
    spark.base_color_factor = {0.0f, 0.0f, 0.0f, 1.0f};
    spark.metallic = 0.0f;
    spark.emissive_factor = {9.0f, 4.5f, 1.2f};
    const renderer::MaterialId spark_mat = world.add_material(spark);

    // ---- viewmodel: hands + AK
    const guerrilla::ViewmodelAssets vm = guerrilla::build_ak_viewmodel();
    std::array<renderer::MaterialId, static_cast<usize>(guerrilla::ViewMaterial::Count)> vm_materials{};
    for (usize i = 0; i < vm_materials.size(); ++i) {
        const auto& tex = vm.textures[i];
        const auto& fac = vm.factors[i];
        const bool flash = i == static_cast<usize>(guerrilla::ViewMaterial::Flash);
        renderer::PbrMaterial m;
        m.base_color = world.upload_texture(tex.base_color, true);
        m.metallic_roughness = world.upload_texture(tex.metallic_roughness, false);
        m.normal = world.upload_texture(tex.normal, false);
        m.metallic = fac.metallic;
        m.roughness = fac.roughness;
        m.emissive_factor = fac.emissive;
        m.double_sided = fac.double_sided;
        if (flash) {
            // The flash texture doubles as emission; its alpha cuts the star shape.
            m.emissive = m.base_color;
            m.base_color_factor = {0.0f, 0.0f, 0.0f, 1.0f};
            m.alpha_cutoff = 0.08f;
        }
        vm_materials[i] = world.add_material(m);
    }
    struct VmPart {
        renderer::MeshId mesh;
        renderer::MaterialId material;
        bool two_sided;
    };
    auto upload_pieces = [&](const std::vector<guerrilla::ViewPiece>& pieces) {
        std::vector<VmPart> parts;
        for (const auto& p : pieces) {
            const usize mi = static_cast<usize>(p.material);
            parts.push_back({world.upload(p.mesh), vm_materials[mi], vm.factors[mi].double_sided});
        }
        return parts;
    };
    const std::vector<VmPart> vm_weapon = upload_pieces(vm.weapon);
    const std::vector<VmPart> vm_mag = upload_pieces(vm.magazine);
    const std::vector<VmPart> vm_right = upload_pieces(vm.right_hand);
    const std::vector<VmPart> vm_left = upload_pieces(vm.left_hand);
    const std::vector<VmPart> vm_flash = upload_pieces(vm.flash);

    // Animated arms + AK-74M rig (glTF); the procedural rifle stays as a fallback.
    std::unique_ptr<guerrilla::RigViewmodel> rig;
    if (!procedural_viewmodel) {
        if (viewmodel_path.empty()) {
            for (const std::filesystem::path& p :
                 {std::filesystem::path("assets/viewmodels/fps_ak74m.glb"),
                  std::filesystem::path(NOVA_GUERRILLA_ASSET_DIR) / "viewmodels/fps_ak74m.glb"}) {
                if (std::filesystem::exists(p)) {
                    viewmodel_path = p.string();
                    break;
                }
            }
        }
        if (!viewmodel_path.empty()) {
            std::string error;
            rig = guerrilla::RigViewmodel::load(viewmodel_path, world, &error);
            if (rig) {
                NOVA_INFO("Guerrilla", "viewmodel {}: {} triangles, skinned + animated", viewmodel_path,
                          rig->triangle_count());
            } else {
                NOVA_WARN("Guerrilla", "could not use viewmodel {} ({}); using the procedural rifle", viewmodel_path,
                          error);
            }
        }
    }
    NOVA_INFO("Guerrilla", "level + viewmodel ready in {:.0f} ms | quality {} (render scale {:.2f}, shadows {})",
              build_timer.elapsed_ms(), quality_name, quality->render_scale, quality->shadow_map);

    // ---- player
    renderer::Camera camera;
    camera.fov_y = math::radians(68.0f);
    camera.z_near = 0.02f;
    camera.z_far = 2000.0f;
    Vec3 feet{0.5f, village::terrain_height(0.5f, 9.0f), 9.0f};
    Vec3 velocity{0, 0, 0};
    f32 yaw = math::radians(-90.0f); // looking down the street (-Z)
    f32 pitch = math::radians(0.5f);
    if (has_cam_override) {
        feet = {cam_pos_override.x, village::terrain_height(cam_pos_override.x, cam_pos_override.z) + cam_pos_override.y,
                cam_pos_override.z};
        yaw = math::radians(cam_yaw_deg);
        pitch = math::radians(cam_pitch_deg);
    }
    bool grounded = true;
    constexpr f32 kEye = 1.62f;
    constexpr f32 kRadius = 0.35f;
    constexpr f32 kGravity = 9.81f;

    // ---- weapon state
    constexpr int kMagSize = 30;
    constexpr f32 kFireInterval = 60.0f / 600.0f; // 600 rounds per minute
    f32 reload_duration = 2.3f; // procedural viewmodel; the rig uses its reload clip lengths
    int ammo = kMagSize;
    f32 fire_cooldown = 0.0f;
    f32 reload_timer = -1.0f; // < 0: not reloading
    f32 flash_timer = 0.0f;
    f32 flash_roll = 0.0f;
    f32 ads = 0.0f;           // 0 hip .. 1 aiming down sights
    f32 kick = 0.0f;          // visual recoil (decays)
    f32 recoil_pitch = 0.0f;  // camera recoil still to be recovered
    Vec3 sway{0, 0, 0};
    f32 bob_phase = 0.0f;
    f32 bob_amount = 0.0f;
    f32 sprint_blend = 0.0f;
    int shots_fired = 0, hits = 0;
    Rng rng{0x9E3779B9u};
    std::vector<Particle> particles;
    f32 round_time = 0.0f;
    f32 reset_timer = -1.0f;

    auto spawn_impact = [&](const Vec3& p, const Vec3& n, bool metal) {
        const int count = metal ? 14 : 9;
        for (int i = 0; i < count; ++i) {
            Particle q;
            q.position = p + n * 0.02f;
            const Vec3 r{rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)};
            q.velocity = math::normalize(n * 1.4f + r) * (metal ? rng.range(3.0f, 7.0f) : rng.range(1.2f, 3.5f));
            q.spin_axis = math::normalize(r + Vec3{0.01f, 0.02f, 0.03f});
            q.size = metal ? rng.range(0.008f, 0.016f) : rng.range(0.015f, 0.04f);
            q.max_life = metal ? rng.range(0.25f, 0.5f) : rng.range(0.8f, 1.6f);
            q.spark = metal;
            particles.push_back(q);
        }
    };

    renderer::WorldSettings settings;
    // Golden-hour alpine sun nestled in the right mountain pass (casts dappled tree shadows + crisp crepuscular god-rays)
    settings.sun_direction = math::normalize(Vec3{0.58f, 0.24f, -0.78f});
    settings.sun_color = {1.0f, 0.76f, 0.48f};
    settings.sun_intensity = 3.65f;
    settings.sky_zenith = {0.022f, 0.135f, 0.56f};
    settings.sky_horizon = {0.26f, 0.50f, 0.82f};
    settings.ground_color = {0.46f, 0.36f, 0.25f};
    settings.ambient = 0.70f;
    settings.fog_density = 0.00016f;
    settings.fog_height_falloff = 0.020f;
    settings.exposure = 0.68f;
    settings.bloom = 0.15f;
    settings.saturation = 1.12f;
    settings.shadow_center = {0, 0, -10};
    settings.shadow_radius = 40.0f;
    settings.render_scale = render_scale_override > 0.0f ? render_scale_override : quality->render_scale;
    settings.ssao = quality->ssao;
    settings.bounce = std::max(quality->bounce, 0.64f);
    settings.point_light_range = 9.0f;

    bool prev_editor_mode = studio.editor_mode;
    Vec3 editor_cam_pos{0.4f, village::terrain_height(0.4f, 9.2f) + 2.05f, 9.2f};
    f32 editor_yaw = math::radians(-90.0f);
    f32 editor_pitch = math::radians(-6.0f);
    if (!headless) {
        window->set_mouse_captured(!studio.editor_mode);
    }
    platform::Input input;
    FrameStats cpu_stats;
    FrameStats gpu_stats;
    Stopwatch frame_timer;
    Stopwatch title_timer;
    f32 time = 3.0f;
    std::vector<WorldDrawItem> items;

    for (u64 frame = 0; max_frames == 0 || frame < max_frames; ++frame) {
        const f64 frame_ms = frame_timer.lap_ms();
        if (frame > 0) {
            cpu_stats.record(frame_ms);
        }
        const f32 dt = headless ? 1.0f / 60.0f : static_cast<f32>(std::min(frame_ms, 50.0) / 1000.0);
        time += dt;

        input.begin_frame();
        if (!window->poll_events(input)) {
            break;
        }
        if (input.was_pressed(platform::Key::Escape)) {
            if (!studio.editor_mode && !headless) {
                studio.stop_play_mode(world);
            } else {
                break;
            }
        }
        if (input.was_pressed(platform::Key::F11)) {
            window->toggle_fullscreen();
        }
        if (input.was_pressed(platform::Key::F5) || (!studio.editor_mode && input.was_pressed(platform::Key::Tab))) {
            if (studio.editor_mode) {
                studio.enter_play_mode();
            } else {
                studio.stop_play_mode(world);
            }
        }
        if (!headless && studio.editor_mode != prev_editor_mode) {
            window->set_mouse_captured(!studio.editor_mode);
            prev_editor_mode = studio.editor_mode;
        }

        if (studio.camera_reset_requested) {
            editor_cam_pos = studio.requested_cam_pos;
            editor_yaw = studio.requested_cam_yaw;
            editor_pitch = studio.requested_cam_pitch;
            feet = {editor_cam_pos.x, studio.ground_height_at(editor_cam_pos.x, editor_cam_pos.z), editor_cam_pos.z};
            yaw = editor_yaw;
            pitch = 0.0f;
            studio.camera_reset_requested = false;
        }

        // Handle Windows Explorer Drag & Drop (.glb/.gltf 3D models or .png/.jpg/2x2 PBR atlases)
        if (!input.dropped_files().empty()) {
            studio.handle_os_file_drops(input, world, camera);
        }

        // ---- scripted input for headless captures
        bool fire_held = !studio.editor_mode && input.is_down(platform::MouseButton::Left);
        bool aim_held = !studio.editor_mode && input.is_down(platform::MouseButton::Right);
        if (headless) {
            aim_held = script_ads;
            fire_held = script_fire && frame + 14 >= max_frames;
            if (fire_held && frame + 1 == max_frames) {
                fire_cooldown = 0.0f;
            }
        }

        Vec3 forward{0, 0, -1}, right{1, 0, 0}, up{0, 1, 0};
        f32 move_speed = 0.0f;
        bool sprint = false;
        if (studio.editor_mode) {
            // ---- NOVA Craft Free-Fly Editor Camera (hold RMB + WASD/QE + Wheel zoom)
            if (input.is_down(platform::MouseButton::Right)) {
                editor_yaw += input.mouse_dx() * 0.0028f;
                editor_pitch = std::clamp(editor_pitch - input.mouse_dy() * 0.0028f, -1.45f, 1.45f);
            }
            forward = {std::cos(editor_pitch) * std::cos(editor_yaw), std::sin(editor_pitch),
                       std::cos(editor_pitch) * std::sin(editor_yaw)};
            right = math::normalize(math::cross(forward, {0, 1, 0}));
            up = math::cross(right, forward);
            if (input.is_down(platform::MouseButton::Right)) {
                Vec3 fly{0, 0, 0};
                if (input.is_down(platform::Key::W)) fly += forward;
                if (input.is_down(platform::Key::S)) fly -= forward;
                if (input.is_down(platform::Key::D)) fly += right;
                if (input.is_down(platform::Key::A)) fly -= right;
                if (input.is_down(platform::Key::E)) fly += Vec3{0, 1, 0};
                if (input.is_down(platform::Key::Q)) fly -= Vec3{0, 1, 0};
                const f32 fly_speed = input.is_down(platform::Key::LeftShift) ? 14.0f : 6.0f;
                if (math::length(fly) > 0.0f) {
                    editor_cam_pos += math::normalize(fly) * (fly_speed * dt);
                }
            }
            if (!studio.pointer_over_ui() && std::abs(input.wheel_delta()) > 0.01f) {
                editor_cam_pos += forward * (input.wheel_delta() * 1.15f);
            }
            editor_cam_pos.y = std::max(editor_cam_pos.y, studio.ground_height_at(editor_cam_pos.x, editor_cam_pos.z) + 0.35f);
            camera.position = editor_cam_pos;
            camera.target = editor_cam_pos + forward;
            camera.fov_y = studio.clay_scene ? math::radians(52.0f) : math::radians(68.0f);
        } else {
            // ---- Gameplay FPS Look & Move
            const f32 sens = 0.0022f * (1.0f - 0.45f * ads);
            yaw += input.mouse_dx() * sens;
            pitch = std::clamp(pitch - input.mouse_dy() * sens, -1.45f, 1.45f);
            const f32 recover = std::min(recoil_pitch, recoil_pitch * 6.0f * dt);
            recoil_pitch -= recover;
            pitch -= recover * 0.75f;

            const Vec3 flat_fwd{std::cos(yaw), 0, std::sin(yaw)};
            const Vec3 flat_right{-flat_fwd.z, 0, flat_fwd.x};
            Vec3 wish{0, 0, 0};
            if (input.is_down(platform::Key::W)) wish += flat_fwd;
            if (input.is_down(platform::Key::S)) wish -= flat_fwd;
            if (input.is_down(platform::Key::D)) wish += flat_right;
            if (input.is_down(platform::Key::A)) wish -= flat_right;
            sprint = input.is_down(platform::Key::LeftShift) && ads < 0.3f && reload_timer < 0.0f;
            const f32 speed = sprint ? 6.2f : (ads > 0.5f ? 2.2f : 3.6f);
            if (math::length(wish) > 0.0f) {
                wish = math::normalize(wish) * speed;
            }
            const f32 accel = grounded ? 12.0f : 2.0f;
            velocity.x = approach(velocity.x, wish.x, accel, dt);
            velocity.z = approach(velocity.z, wish.z, accel, dt);
            if (grounded && input.was_pressed(platform::Key::Space)) {
                velocity.y = 4.3f;
                grounded = false;
            }
            velocity.y -= kGravity * dt;
            feet += velocity * dt;
            // Push out of village walls/houses + any solid objects dropped in NOVA Craft!
            auto resolve_obb = [&](const Vec3& center, const Vec3& half, f32 cyaw) {
                if (feet.y > center.y + half.y - 0.05f || feet.y + kEye < center.y - half.y) return;
                const Vec3 local = rotate_y(feet - center, -cyaw);
                const Vec3 closest{std::clamp(local.x, -half.x, half.x), 0, std::clamp(local.z, -half.z, half.z)};
                Vec3 delta{local.x - closest.x, 0, local.z - closest.z};
                const f32 dist = math::length(delta);
                if (dist >= kRadius) return;
                if (dist < 1e-4f) {
                    const f32 px = half.x - std::abs(local.x), pz = half.z - std::abs(local.z);
                    delta = px < pz ? Vec3{local.x < 0 ? -1.0f : 1.0f, 0, 0} : Vec3{0, 0, local.z < 0 ? -1.0f : 1.0f};
                    feet += rotate_y(delta * ((px < pz ? px : pz) + kRadius), cyaw);
                } else {
                    feet += rotate_y(delta * ((kRadius - dist) / dist), cyaw);
                }
            };
            if (studio.clay_scene) {
                for (const village::Collider& c : studio.clay_colliders()) {
                    resolve_obb(c.center, c.half_extents, c.yaw);
                }
            } else if (!studio.blank_scene) {
                for (const village::Collider& c : scene.colliders) {
                    resolve_obb(c.center, c.half_extents, c.yaw);
                }
            }
            for (const craft::CraftEntity& e : studio.entities) {
                if (e.has_collider && !e.is_static_village) {
                    resolve_obb(e.world_center(),
                                {e.half_extents.x * e.scale.x, e.half_extents.y * e.scale.y, e.half_extents.z * e.scale.z},
                                math::radians(e.rotation_deg.y));
                }
            }
            const f32 radial = std::sqrt(feet.x * feet.x + feet.z * feet.z);
            if (radial > 160.0f) {
                feet.x *= 160.0f / radial;
                feet.z *= 160.0f / radial;
            }
            const f32 ground = studio.ground_height_at(feet.x, feet.z);
            if (feet.y <= ground) {
                feet.y = ground;
                velocity.y = 0.0f;
                grounded = true;
            } else if (feet.y > ground + 0.05f) {
                grounded = false;
            }

            move_speed = std::sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
            bob_amount = approach(bob_amount, grounded ? std::min(move_speed / 3.6f, 1.6f) : 0.0f, 10.0f, dt);
            bob_phase += dt * (5.0f + move_speed * 1.6f);
            const f32 head_bob = std::sin(bob_phase * 2.0f) * 0.018f * bob_amount * (1.0f - 0.7f * ads);
            forward = {std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw)};
            camera.position = feet + Vec3{0, kEye + head_bob, 0};
            camera.target = camera.position + forward;
            right = math::normalize(math::cross(forward, {0, 1, 0}));
            up = math::cross(right, forward);
        }

        // ---- weapon (active only in Play mode)
        ads = approach(ads, (!studio.editor_mode && aim_held && reload_timer < 0.0f && !sprint) ? 1.0f : 0.0f, 14.0f, dt);
        if (!studio.editor_mode) {
            camera.fov_y = math::radians(math::lerp(68.0f, 52.0f, ads));
        }
        fire_cooldown -= dt;
        flash_timer -= dt;
        const bool reload_pressed = (!studio.editor_mode && input.was_pressed(platform::Key::R)) ||
                                    (script_reload && frame + 50 == max_frames);
        if (script_reload && frame + 51 == max_frames) {
            ammo = 0; // scripted: show the full (empty-magazine) reload
        }
        if ((reload_pressed && ammo < kMagSize && reload_timer < 0.0f) ||
            (fire_held && ammo == 0 && reload_timer < 0.0f && fire_cooldown <= 0.0f)) {
            reload_timer = 0.0f;
            if (rig) {
                reload_duration = rig->start_reload(ammo == 0);
            }
        }
        if (reload_timer >= 0.0f) {
            reload_timer += dt;
            if (reload_timer >= reload_duration) {
                reload_timer = -1.0f;
                ammo = kMagSize;
            }
        }
        const Vec3 eye = camera.position;
        if (fire_held && ammo > 0 && reload_timer < 0.0f && fire_cooldown <= 0.0f && !sprint) {
            fire_cooldown += kFireInterval;
            fire_cooldown = std::max(fire_cooldown, 0.0f);
            --ammo;
            ++shots_fired;
            flash_timer = 0.045f;
            flash_roll = rng.range(0.0f, math::kPi * 2.0f);
            if (rig) {
                rig->fire();
            }
            kick = std::min(kick + 1.0f, 2.5f);
            const f32 climb = math::radians(rng.range(0.55f, 0.85f)) * (1.0f - 0.35f * ads);
            pitch = std::min(pitch + climb, 1.45f);
            recoil_pitch += climb;
            yaw += math::radians(rng.range(-0.25f, 0.3f));

            // Bullet: hitscan with cone spread (tighter when aiming, wider when moving).
            const f32 spread = math::radians(math::lerp(1.6f, 0.12f, ads) + std::min(move_speed, 6.0f) * 0.25f);
            const f32 a = rng.range(0.0f, math::kPi * 2.0f), r = std::sqrt(rng.uniform()) * spread;
            const Vec3 dir = math::normalize(forward + right * (std::cos(a) * r) + up * (std::sin(a) * r));
            Hit hit;
            for (usize i = 0; i < targets.size(); ++i) {
                f32 t;
                Vec3 n;
                if (ray_target(eye, dir, targets[i], t, n) && t < hit.t) {
                    hit = {t, n, Hit::Kind::Target, static_cast<int>(i)};
                }
            }
            for (const village::Collider& c : scene.colliders) {
                f32 t;
                Vec3 n;
                if (ray_obb(eye, dir, c, t, n) && t < hit.t) {
                    hit = {t, n, Hit::Kind::Wall, -1};
                }
            }
            if (f32 t; ray_terrain(eye, dir, std::min(hit.t, 400.0f), t) && t < hit.t) {
                const Vec3 p = eye + dir * t;
                hit = {t, terrain_normal(p.x, p.z), Hit::Kind::Terrain, -1};
            }
            if (hit.kind != Hit::Kind::None) {
                const Vec3 p = eye + dir * hit.t;
                spawn_impact(p, hit.normal, hit.kind == Hit::Kind::Target);
                if (hit.kind == Hit::Kind::Target) {
                    Target& t = targets[static_cast<usize>(hit.target)];
                    t.velocity += 7.0f;
                    ++hits;
                }
            }
        }
        kick = approach(kick, 0.0f, 9.0f, dt);

        // ---- targets: hinged plates that fall when hit; round resets once all are down
        int down = 0;
        for (Target& t : targets) {
            if (t.velocity > 0.0f || t.angle > 0.0f) {
                t.velocity -= 9.0f * std::cos(t.angle) * dt * (t.angle < 0.3f ? 0.3f : 1.0f);
                t.angle += t.velocity * dt;
                if (t.angle >= 1.45f) {
                    t.angle = 1.45f;
                    t.velocity = 0.0f;
                    t.down = true;
                } else if (t.angle <= 0.0f) {
                    t.angle = 0.0f;
                    t.velocity = 0.0f;
                }
            }
            down += t.down ? 1 : 0;
        }
        if (down == static_cast<int>(targets.size())) {
            if (reset_timer < 0.0f) {
                reset_timer = 3.0f;
                NOVA_INFO("Guerrilla", "all {} targets down in {:.1f} s ({} hits / {} shots)", targets.size(),
                          round_time, hits, shots_fired);
            }
            reset_timer -= dt;
            if (reset_timer <= 0.0f) {
                for (Target& t : targets) {
                    t.angle = 0.0f;
                    t.velocity = 0.0f;
                    t.down = false;
                }
                reset_timer = -1.0f;
                round_time = 0.0f;
            }
        } else {
            round_time += dt;
        }

        // ---- particles
        for (Particle& q : particles) {
            q.life += dt;
            q.velocity.y -= kGravity * dt;
            q.position += q.velocity * dt;
            q.spin += dt * 12.0f;
            const f32 g = village::terrain_height(q.position.x, q.position.z);
            if (q.position.y < g + q.size) {
                q.position.y = g + q.size;
                q.velocity = Vec3{q.velocity.x * 0.4f, -q.velocity.y * 0.25f, q.velocity.z * 0.4f};
            }
        }
        std::erase_if(particles, [](const Particle& q) { return q.life >= q.max_life; });

        // ---- shadow map follows the player / editor focus, snapped to shadow texels to maximize PRISM-VSM cache hits!
        f32 target_shadow_radius = studio.clay_scene ? 20.0f : (studio.blank_scene ? 14.0f : 40.0f);
        if (studio.blank_scene && !studio.clay_scene && studio.selected >= 0 && studio.selected < static_cast<int>(studio.entities.size())) {
            const auto& se = studio.entities[static_cast<usize>(studio.selected)];
            if (se.kind == craft::EntityKind::Mesh) {
                target_shadow_radius = 8.0f; // 3.9 mm ultra-crisp studio shadow texels around inspected .glb / props!
            }
        }
        settings.shadow_radius = target_shadow_radius;
        settings.bounce = studio.clay_scene ? 0.92f : std::max(quality->bounce, 0.64f);
        settings.ssao = studio.clay_scene ? 0.95f : quality->ssao;
        settings.ssao_radius = studio.clay_scene ? 0.82f : 0.70f;
        // Disable coarse 1024-texel top-down height-map contact_ao on ultra-smooth Clay GI road (GTAO + 36-tap Bilinear PCSS are cleaner!)
        settings.contact_ao = studio.clay_scene ? 0.0f : 0.70f;
        {
            const Vec3 sun = math::normalize(settings.sun_direction);
            const Vec3 lf = -sun;
            const Vec3 lr = math::normalize(math::cross(lf, {0, 1, 0}));
            const Vec3 lu = math::cross(lr, lf);
            Vec3 focus{0.0f, 0.0f, -10.0f};
            if (studio.editor_mode) {
                if (studio.clay_scene) {
                    focus = {0.0f, 0.0f, -2.0f}; // Covers entire clay courtyard from foreground blocks to distant canopy!
                } else if (studio.blank_scene) {
                    // Tight studio focus around selected object (if movable) or central studio platform for razor-sharp 6mm shadows!
                    if (studio.selected >= 0 && studio.selected < static_cast<int>(studio.entities.size()) &&
                        studio.entities[static_cast<usize>(studio.selected)].kind != craft::EntityKind::Terrain) {
                        const Vec3 wc = studio.entities[static_cast<usize>(studio.selected)].world_center();
                        focus = {std::round(wc.x * 0.5f) * 2.0f, std::round(wc.y * 0.5f) * 2.0f, std::round(wc.z * 0.5f) * 2.0f};
                    } else {
                        focus = {std::round(editor_cam_pos.x * 0.5f) * 2.0f, 0.0f, std::round((editor_cam_pos.z - 6.0f) * 0.5f) * 2.0f};
                    }
                } else {
                    focus = {std::round(editor_cam_pos.x * 0.25f) * 4.0f, 0.0f, std::round((editor_cam_pos.z - 12.0f) * 0.25f) * 4.0f};
                }
            } else {
                focus = feet + Vec3{std::cos(yaw), 0.0f, std::sin(yaw)} * (settings.shadow_radius * 0.45f);
            }
            const f32 texel = 2.0f * settings.shadow_radius / static_cast<f32>(quality->shadow_map);
            const f32 cr = math::dot(focus, lr), cu = math::dot(focus, lu);
            settings.shadow_center = focus + lr * (std::round(cr / texel) * texel - cr) +
                                     lu * (std::round(cu / texel) * texel - cu);
        }

        // ---- assemble the frame (Clay GI Benchmark OR New Scene custom world OR Village + editable NOVA Craft entities)
        items.clear();
        if (studio.clay_scene) {
            studio.push_clay_layers(items);
        } else if (!studio.blank_scene) {
            items.assign(level.begin(), level.end());
        }
        for (const craft::CraftEntity& e : studio.entities) {
            items.push_back(e.to_draw_item());
        }
        if (!studio.blank_scene) {
            for (const Target& t : targets) {
                const Mat4 f = target_frame(t);
                WorldDrawItem plate{plate_id, f};
                plate.material = steel_mat;
                plate.two_sided = true;
                items.push_back(plate);
                WorldDrawItem base{base_id, math::translate(t.base) * math::rotate({0, 1, 0}, t.yaw)};
                base.material = post_mat;
                items.push_back(base);
            }
        }
        for (const Particle& q : particles) {
            const f32 fade = q.spark ? 1.0f - q.life / q.max_life : 1.0f;
            WorldDrawItem chip{chip_id, math::translate(q.position) * math::rotate(q.spin_axis, q.spin) *
                                            math::scale(Vec3{q.size, q.size, q.size * (q.spark ? 4.0f : 1.0f)} *
                                                        std::max(fade, 0.2f))};
            chip.material = q.spark ? spark_mat : dust_mat;
            chip.casts_shadow = !q.spark;
            items.push_back(chip);
        }

        // Viewmodel: built in weapon space, then shrunk about the eye (keeps its screen size but
        // stops the barrel poking through walls) and attached to the camera.
        constexpr f32 kVmScale = 0.4f;
        const Vec3 look_delta{input.mouse_dx(), input.mouse_dy(), 0};
        sway = approach(sway, Vec3{-look_delta.x * 0.0009f, look_delta.y * 0.0009f, 0}, 8.0f, dt);
        const Vec3 hip_offset{0.19f, -0.225f, -0.30f};
        const Vec3 sight_mid = (vm.rear_sight + vm.front_sight) * 0.5f;
        // Eye a few mm above the sight line, ~0.37 m behind the rear sight (cheek on the stock).
        const Vec3 ads_offset{-sight_mid.x, -vm.rear_sight.y - 0.006f, -0.37f - vm.rear_sight.z};
        Vec3 offset = math::lerp(hip_offset, ads_offset, smooth01(ads));
        const f32 sprint_w = sprint && move_speed > 1.0f ? 1.0f : 0.0f;
        sprint_blend = approach(sprint_blend, sprint_w, 8.0f, dt);
        const f32 weapon_bob = (1.0f - 0.85f * ads) * bob_amount;
        offset += Vec3{std::sin(bob_phase) * 0.011f * weapon_bob, -std::abs(std::cos(bob_phase)) * 0.009f * weapon_bob,
                       kick * 0.022f};
        // Reload: tilt the rifle, drop the magazine, seat a new one.
        f32 reload_tilt = 0.0f, mag_drop = 0.0f;
        if (reload_timer >= 0.0f) {
            const f32 r = reload_timer / reload_duration;
            reload_tilt = smooth01(r / 0.15f) * (1.0f - smooth01((r - 0.85f) / 0.15f));
            mag_drop = r < 0.5f ? smooth01((r - 0.15f) / 0.2f) : 1.0f - smooth01((r - 0.55f) / 0.25f);
        }
        const Mat4 cam_world = basis(right, up, -forward, eye);
        const Mat4 gun = cam_world * math::scale({kVmScale, kVmScale, kVmScale}) * math::translate(offset) *
                         math::rotate({0, 1, 0}, sway.x + 0.03f * sprint_blend) *
                         math::rotate({1, 0, 0}, sway.y + kick * 0.035f - 0.45f * sprint_blend - 0.18f * reload_tilt) *
                         math::rotate({0, 0, 1}, 0.55f * reload_tilt + 0.25f * sprint_blend);
        const Mat4 mag = gun * math::translate(vm.magazine_pivot + Vec3{0, -0.32f * mag_drop, 0.06f * mag_drop}) *
                         math::rotate({1, 0, 0}, 0.5f * mag_drop) * math::translate(-vm.magazine_pivot);
        const f32 left_follow = smooth01(mag_drop * 1.6f);
        const Mat4 left = gun * math::translate(Vec3{0.0f, -0.30f * left_follow, 0.28f * left_follow});
        auto push_parts = [&](const std::vector<VmPart>& parts, const Mat4& m) {
            for (const VmPart& p : parts) {
                WorldDrawItem it{p.mesh, m};
                it.material = p.material;
                it.two_sided = p.two_sided;
                it.casts_shadow = false; // too small for the shadow map; still receives shadows
                items.push_back(it);
            }
        };
        bool flash_on = !studio.editor_mode && flash_timer > 0.0f;
        if (!studio.editor_mode) {
            if (rig) {
                rig->update({dt, move_speed, grounded, sprint, ads});
                constexpr f32 kRigScale = 0.4f;
                const Mat4 view_to_world = cam_world * math::scale({kRigScale, kRigScale, kRigScale}) *
                                           math::translate(Vec3{0.05f, -0.055f, 0.03f} * (1.0f - smooth01(ads)) +
                                                           Vec3{0.0f, 0.0f, kick * 0.02f * ads}) *
                                           math::rotate({0, 1, 0}, sway.x) * math::rotate({1, 0, 0}, sway.y);
                rig->push_items(items, view_to_world);
                if (flash_on) {
                    push_parts(vm_flash, view_to_world * rig->flash_to_view(ads, vm.muzzle) *
                                             math::translate(vm.muzzle) * math::rotate({0, 0, 1}, flash_roll) *
                                             math::scale({1.0f, 1.0f, rng.range(0.8f, 1.25f)}) *
                                             math::translate(-vm.muzzle));
                    settings.point_light_position =
                        math::transform_point(view_to_world, rig->muzzle_view(ads) + Vec3{0, 0.08f, 0});
                }
            } else {
                push_parts(vm_weapon, gun);
                push_parts(vm_mag, mag);
                push_parts(vm_right, gun);
                push_parts(vm_left, left);
                if (flash_on) {
                    push_parts(vm_flash, gun * math::translate(vm.muzzle) * math::rotate({0, 0, 1}, flash_roll) *
                                             math::scale({1.0f, 1.0f, rng.range(0.8f, 1.25f)}) *
                                             math::translate(-vm.muzzle));
                    settings.point_light_position = math::transform_point(gun, vm.muzzle + Vec3{0, 0.05f, 0});
                }
            }
        }
        if (flash_on) {
            settings.point_light_color = Vec3{12.0f, 6.5f, 2.4f} * rng.range(0.8f, 1.2f);
        } else {
            settings.point_light_color = {0, 0, 0};
        }
        settings.crosshair_px = (!studio.editor_mode && ads < 0.4f && !headless) ? 7.0f + std::min(move_speed, 6.0f) * 1.5f : 0.0f;
        if (headless && !script_ads && !studio.editor_mode) {
            settings.crosshair_px = 7.0f;
        }

        const platform::WindowSize size = window->size();
        if (!headless && (size.width == 0 || size.height == 0)) {
            continue; // minimized: skip rendering cleanly
        }
        if (!headless && (size.width != device->back_buffer_extent().width ||
                          size.height != device->back_buffer_extent().height)) {
            device->resize(size.width, size.height);
        }

        // Update NOVA Craft Panel UI, 3D Gizmos, Drag & Drop, Window Controls, and live Time-of-Day sun slider!
        const rhi::Extent bb = device->back_buffer_extent();
        const craft::WindowAction win_act =
            studio.update_and_draw(ui_overlay, world, input, camera, bb.width, bb.height, dt,
                                   gpu_stats.summarize().avg_ms, world.stats().shadow_cached, headless,
                                   window->is_fullscreen(), window->is_maximized());
        if (win_act == craft::WindowAction::Minimize) {
            window->minimize();
            continue;
        } else if (win_act == craft::WindowAction::ToggleMaximize) {
            window->toggle_maximize();
        } else if (win_act == craft::WindowAction::ToggleFullscreen) {
            window->toggle_fullscreen();
        } else if (win_act == craft::WindowAction::Close) {
            window->request_close();
            break;
        }
        studio.apply_sky_and_lights(settings, flash_on);

        settings.time = time;
        world.render_frame(camera, settings, items, [&](rhi::CommandList& cmd) {
            ui_overlay.render(cmd);
        });
        if (device->last_gpu_frame_ms() >= 0.0) {
            gpu_stats.record(device->last_gpu_frame_ms());
        }
        if (title_timer.elapsed_ms() >= 250.0) {
            title_timer.reset();
            const auto c = cpu_stats.summarize();
            if (studio.editor_mode) {
                window->set_title(std::format("NOVA PRISM Studio | {} objects | PRISM-VSM {} | {:.0f} fps (GPU {:.1f} ms) | {}",
                                              studio.entities.size(), world.stats().shadow_cached ? "CACHED" : "LIVE",
                                              c.avg_fps, gpu_stats.summarize().avg_ms, device->info().device_name));
            } else {
                window->set_title(std::format("NOVA Guerrilla [PLAY] | AK {}/{}{} | targets {}/{} | {:.1f} s | {:.0f} fps "
                                              "(GPU {:.1f} ms) | {}",
                                              ammo, kMagSize, reload_timer >= 0.0f ? " RELOADING" : "", down,
                                              targets.size(), round_time, c.avg_fps, gpu_stats.summarize().avg_ms,
                                              device->info().device_name));
            }
        }
    }

    device->wait_idle();
    int exit_code = 0;
    if (!screenshot.empty()) {
        std::vector<u8> pixels;
        const rhi::Extent e = device->back_buffer_extent();
        if (headless && device->read_back(pixels) && write_ppm(screenshot, pixels, e.width, e.height)) {
            NOVA_INFO("Guerrilla", "wrote {}", screenshot);
        } else {
            NOVA_ERROR("Guerrilla", "failed to write screenshot {} (requires --headless)", screenshot);
            exit_code = 1;
        }
    }
    const auto c = cpu_stats.summarize();
    const auto g = gpu_stats.summarize();
    NOVA_INFO("Guerrilla", "exit: CPU avg {:.2f} ms p99 {:.2f} ms | GPU avg {:.2f} ms p99 {:.2f} ms | {} shots, {} hits",
              c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms, shots_fired, hits);
    if (device->validation_error_count() > 0) {
        NOVA_ERROR("Guerrilla", "{} Vulkan validation errors", device->validation_error_count());
        exit_code = 1;
    }
    return exit_code;
}
