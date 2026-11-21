#pragma once

// NOVA Craft — Built-In Zero-Restart Live 2K Studio Panel UI & 3D Transform Gizmos
// - Drag & Drop PBR Assets directly onto the 3D terrain/scene (plus OS Windows Explorer .glb / .png / 2x2 atlas drop!)
// - 3D Interactive Move [W] / Rotate [E] / Scale [R] Viewport Gizmos + Wireframe Selection Box
// - Live 120 FPS UV & PBR Texture Editor (Tile, Offset, Rotation, World-Lock UV, NOVA PRISM Steep POM Depth, Normal, Roughness, Tint)
// - Instant ▶ PLAY / ⏹ STOP state snapshotting + Undo (Ctrl+Z) + Save (.nova.json)

#include "nova/core/log.hpp"
#include "nova/math/math.hpp"
#include "nova/platform/input.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/model.hpp"
#include "nova/renderer/ui_overlay.hpp"
#include "nova/renderer/world_renderer.hpp"
#include "clay_test_scene.hpp"
#include "village_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace nova::craft {

using math::Mat4;
using math::Vec2;
using math::Vec3;
using math::Vec4;

enum class GizmoMode : u8 { Move = 0, Rotate = 1, Scale = 2 };
enum class WindowAction : u8 { None = 0, Minimize, ToggleMaximize, ToggleFullscreen, Close };
enum class EntityKind : u8 { Mesh = 0, Plane, Terrain, PointLight };

struct SkyConfig {
    f32 sun_hour = 16.65f;
    f32 sun_intensity = 3.60f;
    Vec3 sun_color{1.0f, 0.86f, 0.68f};
    Vec3 sky_zenith{0.028f, 0.165f, 0.62f};
    Vec3 sky_horizon{0.30f, 0.54f, 0.86f};
    Vec3 ground_color{0.48f, 0.39f, 0.29f};
    f32 ambient = 0.68f;
    f32 fog_density = 0.00016f;
    f32 exposure = 0.70f;
    f32 bloom = 0.14f;
    f32 saturation = 1.12f;
    int preset_id = 0; // 0=Golden, 1=Noon, 2=Sunset, 3=Night
};

struct CraftEntity {
    std::string name;
    EntityKind kind = EntityKind::Mesh;
    renderer::MeshId mesh;
    renderer::MaterialId material;
    Vec3 position{0, 0, 0};
    Vec3 rotation_deg{0, 0, 0}; // pitch (X), yaw (Y), roll (Z)
    Vec3 scale{1, 1, 1};
    Vec3 local_center{0, 0, 0};
    Vec3 half_extents{0.5f, 0.5f, 0.5f};
    Vec2 uv_tile{1.0f, 1.0f};
    Vec2 uv_offset{0.0f, 0.0f};
    f32 uv_rotation_deg = 0.0f;
    f32 pom_depth = 0.024f;
    bool world_lock_uv = true;
    bool two_sided = false;
    bool casts_shadow = true;
    bool is_static_village = false; // village base mesh (already world-positioned; transform defaults to identity)
    bool has_collider = true;
    u32 group_id = 0; // non-zero links multi-primitive imported .glb models so Move/Rotate/Scale transforms all parts together!
    // PointLight properties (used when kind == EntityKind::PointLight)
    Vec3 light_color{1.0f, 0.68f, 0.32f};
    f32 light_intensity = 16.0f;
    f32 light_range = 16.0f;
    // Terrain procedural parameters (used when kind == EntityKind::Terrain)
    f32 terrain_size = 110.0f;
    f32 terrain_amp = 16.0f;
    f32 terrain_freq = 0.032f;
    f32 terrain_seed = 42.0f;
    bool terrain_alpine_shader = true; // true = procedural alpine rock/forest/snow; false = PBR material

    [[nodiscard]] Mat4 model_matrix() const noexcept {
        return math::translate(position)
             * math::rotate({0, 1, 0}, math::radians(rotation_deg.y))
             * math::rotate({1, 0, 0}, math::radians(rotation_deg.x))
             * math::rotate({0, 0, 1}, math::radians(rotation_deg.z))
             * math::scale(scale);
    }

    [[nodiscard]] Vec3 world_center() const noexcept {
        return math::transform_point(model_matrix(), local_center);
    }

    [[nodiscard]] renderer::WorldDrawItem to_draw_item() const noexcept {
        renderer::WorldDrawItem item;
        item.mesh = mesh;
        item.transform = model_matrix();
        item.two_sided = two_sided;
        item.casts_shadow = casts_shadow;
        item.material = (kind == EntityKind::Terrain && terrain_alpine_shader) ? renderer::MaterialId{} : material;
        item.uv_tile = uv_tile;
        item.uv_offset = uv_offset;
        item.uv_rotation_deg = uv_rotation_deg;
        item.pom_depth = pom_depth;
        item.world_lock_uv = world_lock_uv && !is_static_village && kind != EntityKind::Terrain;
        return item;
    }
};

struct AssetPrefab {
    std::string label;
    std::string sublabel;
    renderer::MeshId mesh;
    renderer::MaterialId material;
    rhi::TextureHandle thumb;
    Vec3 default_scale{1, 1, 1};
    Vec3 local_center{0, 0.5f, 0};
    Vec3 half_extents{0.5f, 0.5f, 0.5f};
    f32 y_offset = 0.0f;
    f32 pom_depth = 0.028f;
    bool casts_shadow = true;
    bool has_collider = true;
    Vec4 fallback_tint{0.7f, 0.7f, 0.7f, 1.0f};
};

// Assign clean planar/box UVs and exact tangent frames to procedural prefabs so Normal + Steep POM work out-of-the-box!
inline renderer::MeshData make_pbr_cube(f32 sx = 1.0f, f32 sy = 1.0f, f32 sz = 1.0f) {
    renderer::MeshData m = renderer::make_cube(1.0f);
    for (auto& v : m.vertices) {
        v.position.x *= sx;
        v.position.y = (v.position.y + 0.5f) * sy; // pivot cleanly at bottom y = 0!
        v.position.z *= sz;
        v.color = {1.0f, 1.0f, 1.0f, 0.0f};
        const Vec3 n = v.normal;
        if (std::abs(n.y) > 0.5f) {
            v.uv = {v.position.x / std::max(sx, 0.1f) + 0.5f, v.position.z / std::max(sz, 0.1f) + 0.5f};
            v.tangent = {1.0f, 0.0f, 0.0f, n.y > 0.0f ? -1.0f : 1.0f};
        } else if (std::abs(n.x) > 0.5f) {
            v.uv = {v.position.z / std::max(sz, 0.1f) + 0.5f, 1.0f - v.position.y / std::max(sy, 0.1f)};
            v.tangent = {0.0f, 0.0f, 1.0f, n.x > 0.0f ? -1.0f : 1.0f};
        } else {
            v.uv = {v.position.x / std::max(sx, 0.1f) + 0.5f, 1.0f - v.position.y / std::max(sy, 0.1f)};
            v.tangent = {1.0f, 0.0f, 0.0f, n.z > 0.0f ? 1.0f : -1.0f};
        }
    }
    return m;
}

inline renderer::MeshData make_pbr_cylinder(f32 radius, f32 height, u32 segs = 20) {
    renderer::MeshData m = renderer::make_cylinder(segs, radius, radius * 0.92f, height);
    for (auto& v : m.vertices) {
        v.color = {1.0f, 1.0f, 1.0f, 0.0f};
        const f32 ang = std::atan2(v.position.z, v.position.x);
        v.uv = {(ang / (2.0f * math::kPi) + 0.5f) * 2.0f, 1.0f - v.position.y / std::max(height, 0.1f)};
        Vec3 t{-std::sin(ang), 0.0f, std::cos(ang)};
        v.tangent = {t.x, t.y, t.z, 1.0f};
    }
    return m;
}

inline renderer::MeshData make_pbr_boulder(f32 rx, f32 ry, f32 rz) {
    renderer::MeshData m = renderer::make_icosphere(2);
    for (auto& v : m.vertices) {
        v.position = {v.position.x * rx, (v.position.y + 0.65f) * ry, v.position.z * rz};
        v.color = {1.0f, 1.0f, 1.0f, 0.0f};
        const f32 ang = std::atan2(v.position.z, v.position.x);
        v.uv = {ang / (2.0f * math::kPi) + 0.5f, 0.5f - v.normal.y * 0.5f};
        Vec3 t{-std::sin(ang), 0.0f, std::cos(ang)};
        v.tangent = {t.x, t.y, t.z, 1.0f};
    }
    return m;
}

inline renderer::MeshData make_mirror_puddle_mesh(f32 radius) {
    renderer::MeshData m;
    constexpr int kSegs = 32;
    const Vec4 col = renderer::material_color(renderer::Material::Water, {0.03f, 0.025f, 0.02f});
    m.vertices.push_back({{0, 0.035f, 0}, {0, 1, 0}, col, {0.5f, 0.5f}, {1, 0, 0, 1}});
    for (int i = 0; i <= kSegs; ++i) {
        const f32 a = (static_cast<f32>(i) / kSegs) * 2.0f * math::kPi;
        const f32 w = 0.78f + 0.22f * std::sin(a * 3.0f + 0.5f) + 0.12f * std::cos(a * 5.0f - 0.8f);
        m.vertices.push_back({{std::cos(a) * radius * w, 0.035f, std::sin(a) * radius * 1.25f * w},
                              {0, 1, 0}, col, {0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a)}, {1, 0, 0, 1}});
    }
    for (int i = 1; i <= kSegs; ++i) {
        m.indices.insert(m.indices.end(), {0, static_cast<u32>(i + 1), static_cast<u32>(i)});
    }
    return m;
}

// Subdivided 3D PBR Ground Plane (with exact UVs & Tangents for Normal + Steep POM + World-Lock UV)
inline renderer::MeshData make_pbr_plane(f32 size = 16.0f, int segs = 16) {
    renderer::MeshData m;
    const f32 half = size * 0.5f;
    const f32 uv_rep = size * 0.35f;
    m.vertices.reserve(static_cast<usize>((segs + 1) * (segs + 1)));
    for (int j = 0; j <= segs; ++j) {
        const f32 tz = static_cast<f32>(j) / static_cast<f32>(segs);
        for (int i = 0; i <= segs; ++i) {
            const f32 tx = static_cast<f32>(i) / static_cast<f32>(segs);
            const f32 x = -half + tx * size;
            const f32 z = -half + tz * size;
            m.vertices.push_back({{x, 0.0f, z}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 0.0f},
                                  {tx * uv_rep, tz * uv_rep}, {1.0f, 0.0f, 0.0f, -1.0f}});
        }
    }
    for (int j = 0; j < segs; ++j) {
        for (int i = 0; i < segs; ++i) {
            const u32 a = static_cast<u32>(j * (segs + 1) + i);
            const u32 b = a + 1;
            const u32 c = a + static_cast<u32>(segs + 1);
            const u32 d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, d, a, d, b});
        }
    }
    return m;
}

// Glowing Emissive Light Bulb Orb for draggable 3D Point Lights
inline renderer::MeshData make_light_orb(f32 radius = 0.28f, const Vec3& color = {1.0f, 0.78f, 0.42f}) {
    renderer::MeshData m = renderer::make_icosphere(2);
    const Vec4 col = renderer::material_color(renderer::Material::Emissive, color);
    for (auto& v : m.vertices) {
        v.position = v.position * radius;
        v.color = col;
    }
    return m;
}

// Procedural Terrain height function (smooth valley basin at center + ridged mountains around perimeter!)
inline f32 craft_terrain_height(f32 x, f32 z, f32 amp, f32 freq, f32 seed) noexcept {
    auto h2 = [](int ix, int iz, u32 s) noexcept -> f32 {
        u32 h = static_cast<u32>(ix) * 374761393u + static_cast<u32>(iz) * 668265263u + s * 1442695041u;
        h = (h ^ (h >> 13u)) * 1274126177u;
        return static_cast<f32>((h ^ (h >> 16u)) & 0xFFFFu) / 65535.0f;
    };
    auto vnoise = [&](f32 px, f32 pz, u32 s) noexcept -> f32 {
        const f32 fx = std::floor(px), fz = std::floor(pz);
        const int ix = static_cast<int>(fx), iz = static_cast<int>(fz);
        const f32 tx = px - fx, tz = pz - fz;
        const f32 ux = tx * tx * (3.0f - 2.0f * tx);
        const f32 uz = tz * tz * (3.0f - 2.0f * tz);
        const f32 a = h2(ix, iz, s), b = h2(ix + 1, iz, s), c = h2(ix, iz + 1, s), d = h2(ix + 1, iz + 1, s);
        return math::lerp(math::lerp(a, b, ux), math::lerp(c, d, ux), uz);
    };
    const u32 s = static_cast<u32>(std::max(1.0f, seed * 17.0f));
    f32 sum = 0.0f, a = 0.55f, f = freq;
    for (int oct = 0; oct < 5; ++oct) {
        const f32 n = vnoise(x * f + seed * 3.1f, z * f - seed * 1.9f, s + static_cast<u32>(oct * 29));
        const f32 rid = 1.0f - std::abs(n * 2.0f - 1.0f);
        sum += a * (rid * rid * 0.65f + n * 0.35f);
        f *= 2.05f;
        a *= 0.48f;
    }
    // Flat playable studio clearing near the origin so Planes & imported .glb models sit cleanly flush in the valley!
    const f32 dist = std::hypot(x, z);
    const f32 clearing = std::clamp((dist - 12.0f) / 18.0f, 0.0f, 1.0f);
    const f32 rim_boost = 1.0f + std::clamp((dist - 22.0f) / 38.0f, 0.0f, 1.8f);
    return (sum - 0.18f) * amp * (0.015f + 0.985f * clearing * clearing) * rim_boost;
}

inline renderer::MeshData make_procedural_terrain(f32 size = 110.0f, int segs = 104,
                                                  f32 amp = 16.0f, f32 freq = 0.032f, f32 seed = 42.0f) {
    renderer::MeshData m;
    const f32 half = size * 0.5f;
    const f32 step = size / static_cast<f32>(segs);
    const f32 uv_rep = size * 0.14f;
    m.vertices.reserve(static_cast<usize>((segs + 1) * (segs + 1)));
    for (int j = 0; j <= segs; ++j) {
        const f32 tz = static_cast<f32>(j) / static_cast<f32>(segs);
        const f32 z = -half + tz * size;
        for (int i = 0; i <= segs; ++i) {
            const f32 tx = static_cast<f32>(i) / static_cast<f32>(segs);
            const f32 x = -half + tx * size;
            const f32 y = craft_terrain_height(x, z, amp, freq, seed);
            const f32 e = step * 0.65f;
            const f32 hx0 = craft_terrain_height(x - e, z, amp, freq, seed);
            const f32 hx1 = craft_terrain_height(x + e, z, amp, freq, seed);
            const f32 hz0 = craft_terrain_height(x, z - e, amp, freq, seed);
            const f32 hz1 = craft_terrain_height(x, z + e, amp, freq, seed);
            const Vec3 n = math::normalize(Vec3{hx0 - hx1, 2.0f * e, hz0 - hz1});
            const Vec3 t = math::normalize(Vec3{1.0f, (hx1 - hx0) / (2.0f * e), 0.0f});
            const Vec4 col = renderer::material_color(renderer::Material::Terrain, {1.0f, 1.0f, 1.0f});
            m.vertices.push_back({{x, y, z}, n, col, {tx * uv_rep, tz * uv_rep}, {t.x, t.y, t.z, -1.0f}});
        }
    }
    for (int j = 0; j < segs; ++j) {
        for (int i = 0; i < segs; ++i) {
            const u32 a = static_cast<u32>(j * (segs + 1) + i);
            const u32 b = a + 1;
            const u32 c = a + static_cast<u32>(segs + 1);
            const u32 d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, d, a, d, b});
        }
    }
    return m;
}

class Studio {
public:
    bool editor_mode = true;
    bool ui_visible = true;
    bool blank_scene = false; // true = New Scene (user builds from scratch with Sky, Light, Plane, Terrain!)
    bool clay_scene = false;  // true = Architectural Clay GI Lighting Benchmark Courtyard!
    bool inspect_sky = false; // true = Right Inspector shows Sky & Atmosphere Studio controls
    bool camera_reset_requested = false;
    Vec3 requested_cam_pos{0.0f, 4.2f, 13.5f};
    f32 requested_cam_yaw = math::radians(-90.0f);
    f32 requested_cam_pitch = math::radians(-11.0f);
    GizmoMode gizmo_mode = GizmoMode::Move;
    bool snap_enabled = false;
    SkyConfig sky;
    std::vector<CraftEntity> entities;
    int selected = -1;
    std::string status_toast;
    f32 toast_timer = 0.0f;

    void init(renderer::WorldRenderer& world, const village::VillageMaterials& mats,
              renderer::MeshId wall_mesh, renderer::MeshId dirt_mesh, renderer::MeshId mud_mesh,
              renderer::MeshId forest_mesh, renderer::MeshId wood_mesh, renderer::MeshId bark_mesh) {
        mats_ = mats;
        village_meshes_ = {wall_mesh, dirt_mesh, mud_mesh, forest_mesh, wood_mesh, bark_mesh};
        mat_palette_ = {
            {"WALL",   mats.stone_wall},
            {"WOOD",   mats.wood_props},
            {"BARK",   mats.tree_bark},
            {"DIRT",   mats.dirt_road},
            {"MUD",    mats.mud_patches},
            {"FOREST", mats.forest_floor},
        };

        // Upload clean PBR Drag-and-Drop prefab meshes + Plane + Light Orb
        m_wall_blk_ = world.upload(make_pbr_cube(1.8f, 1.2f, 0.65f));
        m_crate_    = world.upload(make_pbr_cube(0.85f, 0.85f, 0.85f));
        m_pillar_   = world.upload(make_pbr_cylinder(0.24f, 2.2f, 20));
        m_boulder_  = world.upload(make_pbr_boulder(0.65f, 0.45f, 0.55f));
        m_slab_     = world.upload(make_pbr_cube(2.2f, 0.25f, 2.2f));
        m_moss_     = world.upload(make_pbr_boulder(0.85f, 0.35f, 0.75f));
        m_plaster_  = world.upload(make_pbr_cube(2.4f, 2.6f, 2.4f));
        m_puddle_   = world.upload(make_mirror_puddle_mesh(1.1f));
        m_plane_    = world.upload(make_pbr_plane(16.0f, 16));
        m_light_    = world.upload(make_light_orb(0.28f, {1.0f, 0.82f, 0.48f}));

        auto thumb_of = [&](renderer::MaterialId id) -> rhi::TextureHandle {
            const auto* m = world.get_material(id);
            return m ? m->base_color : rhi::TextureHandle{};
        };

        prefabs_ = {
            {"Stone Wall",  "PBR + POM",   m_wall_blk_, mats.stone_wall,   thumb_of(mats.stone_wall),   {1, 1, 1}, {0, 0.60f, 0}, {0.90f, 0.60f, 0.32f}, 0.0f, 0.032f, true,  true,  {0.82f, 0.62f, 0.42f, 1}},
            {"Wood Crate",  "PBR Planks",  m_crate_,    mats.wood_props,   thumb_of(mats.wood_props),   {1, 1, 1}, {0, 0.42f, 0}, {0.42f, 0.42f, 0.42f}, 0.0f, 0.022f, true,  true,  {0.62f, 0.45f, 0.28f, 1}},
            {"Bark Pillar", "3D Log",      m_pillar_,   mats.tree_bark,    thumb_of(mats.tree_bark),    {1, 1, 1}, {0, 1.10f, 0}, {0.26f, 1.10f, 0.26f}, 0.0f, 0.028f, true,  true,  {0.48f, 0.36f, 0.24f, 1}},
            {"Mud Rock",    "Wet PBR",     m_boulder_,  mats.mud_patches,  thumb_of(mats.mud_patches),  {1, 1, 1}, {0, 0.30f, 0}, {0.62f, 0.38f, 0.52f}, 0.0f, 0.026f, true,  true,  {0.35f, 0.28f, 0.22f, 1}},
            {"Dirt Slab",   "Gravel Pad",  m_slab_,     mats.dirt_road,    thumb_of(mats.dirt_road),    {1, 1, 1}, {0, 0.12f, 0}, {1.10f, 0.14f, 1.10f}, 0.0f, 0.024f, true,  true,  {0.58f, 0.46f, 0.32f, 1}},
            {"Moss Stone",  "Forest PBR",  m_moss_,     mats.forest_floor, thumb_of(mats.forest_floor), {1, 1, 1}, {0, 0.22f, 0}, {0.82f, 0.32f, 0.72f}, 0.0f, 0.025f, true,  true,  {0.32f, 0.42f, 0.22f, 1}},
            {"Stone Block", "Modular",     m_plaster_,  mats.stone_wall,   thumb_of(mats.stone_wall),   {1, 1, 1}, {0, 1.30f, 0}, {1.20f, 1.30f, 1.20f}, 0.0f, 0.030f, true,  true,  {0.78f, 0.68f, 0.56f, 1}},
            {"SSR Puddle",  "Live Mirror", m_puddle_,   {},                {},                          {1, 1, 1}, {0, 0.04f, 0}, {1.10f, 0.06f, 1.35f}, 0.0f, 0.0f,   false, false, {0.32f, 0.60f, 0.92f, 1}},
        };

        load_village_scene(world);
    }

    void apply_sky_preset(int preset) noexcept {
        sky.preset_id = preset;
        if (preset == 0) { // Golden Hour Alpine (balanced natural warm sun + rich cobalt sky fill!)
            sky.sun_hour = 16.65f;
            sky.sun_intensity = 3.60f;
            sky.sun_color = {1.0f, 0.86f, 0.68f};
            sky.sky_zenith = {0.028f, 0.165f, 0.62f};
            sky.sky_horizon = {0.30f, 0.54f, 0.86f};
            sky.ground_color = {0.48f, 0.39f, 0.29f};
            sky.ambient = 0.68f;
            sky.fog_density = 0.00016f;
            sky.exposure = 0.70f;
            sky.bloom = 0.14f;
            sky.saturation = 1.12f;
        } else if (preset == 1) { // Crisp Studio Daylight (true-to-life PBR material & color calibration)
            sky.sun_hour = 13.8f;
            sky.sun_intensity = 3.45f;
            sky.sun_color = {1.0f, 0.95f, 0.88f};
            sky.sky_zenith = {0.036f, 0.215f, 0.72f};
            sky.sky_horizon = {0.40f, 0.65f, 0.92f};
            sky.ground_color = {0.46f, 0.41f, 0.34f};
            sky.ambient = 0.72f;
            sky.fog_density = 0.00010f;
            sky.exposure = 0.69f;
            sky.bloom = 0.10f;
            sky.saturation = 1.08f;
        } else if (preset == 2) { // Dramatic Crimson Sunset
            sky.sun_hour = 18.15f;
            sky.sun_intensity = 3.90f;
            sky.sun_color = {1.0f, 0.50f, 0.24f};
            sky.sky_zenith = {0.06f, 0.09f, 0.36f};
            sky.sky_horizon = {0.58f, 0.31f, 0.25f};
            sky.ground_color = {0.34f, 0.22f, 0.15f};
            sky.ambient = 0.56f;
            sky.fog_density = 0.00024f;
            sky.exposure = 0.72f;
            sky.bloom = 0.20f;
            sky.saturation = 1.15f;
        } else { // Moonlit Night (Point Lights shine dramatically!)
            sky.sun_hour = 17.4f;
            sky.sun_intensity = 0.45f;
            sky.sun_color = {0.55f, 0.72f, 1.0f};
            sky.sky_zenith = {0.006f, 0.015f, 0.065f};
            sky.sky_horizon = {0.025f, 0.048f, 0.125f};
            sky.ground_color = {0.04f, 0.05f, 0.08f};
            sky.ambient = 0.42f;
            sky.fog_density = 0.00032f;
            sky.exposure = 0.88f;
            sky.bloom = 0.22f;
            sky.saturation = 1.05f;
        }
    }

    void load_village_scene(renderer::WorldRenderer& world) {
        blank_scene = false;
        clay_scene = false;
        inspect_sky = false;
        apply_sky_preset(0);
        entities.clear();
        auto add_village = [&](std::string n, renderer::MeshId m, renderer::MaterialId mat, Vec3 center, Vec3 half,
                               bool shadow, f32 pom) {
            CraftEntity e;
            e.name = std::move(n);
            e.mesh = m;
            e.material = mat;
            e.local_center = center;
            e.half_extents = half;
            e.casts_shadow = shadow;
            e.pom_depth = pom;
            e.is_static_village = true;
            e.has_collider = false;
            entities.push_back(std::move(e));
        };
        add_village("Village_Stone_Walls", village_meshes_[0], mats_.stone_wall,   {-2.3f, 0.8f, 6.0f},  {0.5f, 0.8f, 6.0f},  true,  0.022f);
        add_village("Village_Dirt_Road",   village_meshes_[1], mats_.dirt_road,    { 0.0f, 0.05f, 0.0f}, {2.4f, 0.1f, 12.0f}, false, 0.020f);
        add_village("Village_Wet_Mud",     village_meshes_[2], mats_.mud_patches,  { 0.0f, 0.05f, 1.3f}, {1.6f, 0.1f, 2.4f},  false, 0.018f);
        add_village("Village_Forest_Bed",  village_meshes_[3], mats_.forest_floor, {-1.7f, 0.05f, 5.8f}, {1.4f, 0.1f, 2.2f},  false, 0.018f);
        add_village("Village_Wood_Props",  village_meshes_[4], mats_.wood_props,   {-3.1f, 1.2f, -1.0f}, {0.8f, 1.2f, 2.5f},  true,  0.016f);
        add_village("Village_Olive_Bark",  village_meshes_[5], mats_.tree_bark,    {-3.2f, 2.2f, 3.5f},  {0.6f, 2.2f, 0.6f},  true,  0.020f);

        CraftEntity crate;
        crate.name = "Supply_Crate_01";
        crate.mesh = m_crate_;
        crate.material = mats_.wood_props;
        crate.position = {1.35f, village::terrain_height(1.35f, 4.2f) + 0.02f, 4.2f};
        crate.rotation_deg = {0.0f, -22.0f, 0.0f};
        crate.scale = {1.0f, 1.0f, 1.0f};
        crate.local_center = {0, 0.425f, 0};
        crate.half_extents = {0.425f, 0.425f, 0.425f};
        crate.pom_depth = 0.024f;
        crate.world_lock_uv = true;
        entities.push_back(crate);
        selected = static_cast<int>(entities.size()) - 1;
        requested_cam_pos = {0.4f, village::terrain_height(0.4f, 9.2f) + 2.05f, 9.2f};
        requested_cam_yaw = math::radians(-90.0f);
        requested_cam_pitch = math::radians(-6.0f);
        camera_reset_requested = true;
        world.invalidate_contact_ao();
        toast("Loaded Village Scene");
    }

    // Spawn a 3D PBR Ground Plane
    int spawn_plane(renderer::WorldRenderer& world, Vec3 pos = {0.0f, 0.0f, 0.0f}) {
        push_undo();
        CraftEntity e;
        e.name = std::format("Plane_{:02d}", static_cast<int>(entities.size()) + 1);
        e.kind = EntityKind::Plane;
        e.mesh = m_plane_;
        e.material = mats_.dirt_road;
        e.position = pos;
        e.scale = {1.0f, 1.0f, 1.0f};
        e.local_center = {0, 0.02f, 0};
        e.half_extents = {8.0f, 0.06f, 8.0f};
        e.pom_depth = 0.022f;
        e.casts_shadow = false;
        e.has_collider = false;
        e.world_lock_uv = true;
        entities.push_back(std::move(e));
        selected = static_cast<int>(entities.size()) - 1;
        inspect_sky = false;
        world.invalidate_contact_ao();
        toast("Added 3D PBR Plane");
        return selected;
    }

    // Spawn a 3D Procedural Sculptable Terrain
    int spawn_terrain(renderer::WorldRenderer& world, Vec3 pos = {0.0f, -0.35f, -12.0f},
                      f32 amp = 16.0f, f32 freq = 0.032f, f32 seed = 42.0f, bool alpine = true) {
        push_undo();
        CraftEntity e;
        e.name = std::format("Terrain_{:02d}", static_cast<int>(entities.size()) + 1);
        e.kind = EntityKind::Terrain;
        e.terrain_size = 120.0f;
        e.terrain_amp = amp;
        e.terrain_freq = freq;
        e.terrain_seed = seed;
        e.terrain_alpine_shader = alpine;
        e.mesh = world.upload(make_procedural_terrain(e.terrain_size, 108, amp, freq, seed));
        e.material = mats_.dirt_road;
        e.position = pos;
        e.scale = {1.0f, 1.0f, 1.0f};
        e.local_center = {0, amp * 0.25f, 0};
        e.half_extents = {e.terrain_size * 0.5f, amp, e.terrain_size * 0.5f};
        e.pom_depth = 0.018f;
        e.casts_shadow = true;
        e.has_collider = false;
        e.world_lock_uv = false;
        entities.push_back(std::move(e));
        selected = static_cast<int>(entities.size()) - 1;
        inspect_sky = false;
        world.invalidate_contact_ao();
        toast("Added 3D Procedural Terrain (120m)");
        return selected;
    }

    // Spawn a Live Draggable 3D Point Light Orb
    int spawn_point_light(renderer::WorldRenderer& world, Vec3 pos = {0.0f, 2.4f, 0.0f}) {
        push_undo();
        CraftEntity e;
        e.name = std::format("Point_Light_{:02d}", static_cast<int>(entities.size()) + 1);
        e.kind = EntityKind::PointLight;
        e.mesh = m_light_;
        e.material = {};
        e.position = pos;
        e.scale = {1.0f, 1.0f, 1.0f};
        e.local_center = {0, 0, 0};
        e.half_extents = {0.35f, 0.35f, 0.35f};
        e.light_color = {1.0f, 0.68f, 0.30f};
        e.light_intensity = 18.0f;
        e.light_range = 18.0f;
        e.casts_shadow = false;
        e.has_collider = false;
        e.pom_depth = 0.0f;
        entities.push_back(std::move(e));
        selected = static_cast<int>(entities.size()) - 1;
        inspect_sky = false;
        world.invalidate_shadows();
        toast("Added 3D Point Light (drag with Gizmo!)");
        return selected;
    }

    // Create a clean New Scene — optionally pre-seeded with Sky + Point Light + Plane + Terrain!
    void create_new_scene(renderer::WorldRenderer& world, bool populate_starter = true) {
        blank_scene = true;
        clay_scene = false;
        inspect_sky = false;
        entities.clear();
        undo_stack_.clear();
        selected = -1;
        // Studio-calibrated natural daylight-golden balance: vivid authentic PBR colors (olive cloth, flags, skin, metal!)
        apply_sky_preset(0);
        sky.sun_hour = 15.65f;
        sky.sun_intensity = 3.55f;
        sky.sun_color = {1.0f, 0.90f, 0.78f};
        sky.sky_zenith = {0.032f, 0.185f, 0.66f};
        sky.sky_horizon = {0.34f, 0.58f, 0.88f};
        sky.ground_color = {0.48f, 0.40f, 0.31f};
        sky.ambient = 0.70f;
        sky.fog_density = 0.00011f;
        sky.exposure = 0.70f;
        sky.bloom = 0.12f;
        sky.saturation = 1.10f;
        if (populate_starter) {
            // 1. Sculpted 3D Procedural Mountain Terrain (with flat playable central clearing)
            spawn_terrain(world, {0.0f, -0.35f, -10.0f}, 18.0f, 0.032f, 42.0f, true);
            // 2. Clean 3D PBR Ground Plane sitting cleanly ON TOP of the sunlit valley floor!
            const f32 valley_y = ground_height_at(0.0f, 0.0f);
            const int pl_idx = spawn_plane(world, {0.0f, valley_y + 0.14f, 0.0f});
            entities[static_cast<usize>(pl_idx)].material = mats_.dirt_road;
            entities[static_cast<usize>(pl_idx)].uv_tile = {1.2f, 1.2f};
            // 3. Draggable Glowing 3D Point Light (balanced warm studio rim light)
            const int lt_idx = spawn_point_light(world, {2.8f, valley_y + 2.4f, -1.5f});
            entities[static_cast<usize>(lt_idx)].light_color = {1.0f, 0.82f, 0.58f};
            entities[static_cast<usize>(lt_idx)].light_intensity = 10.0f;
            entities[static_cast<usize>(lt_idx)].light_range = 16.0f;
            // Select the Point Light so its 3D Move Gizmo + Light controls are immediately live!
            selected = lt_idx;
            toast("New Scene Ready: Studio Sky + Light + Plane + Terrain!");
        } else {
            inspect_sky = true;
            toast("Empty New Scene Ready — click +PLANE / +TERRAIN / +LIGHT!");
        }
        requested_cam_pos = {0.0f, 4.2f, 13.5f};
        requested_cam_yaw = math::radians(-90.0f);
        requested_cam_pitch = math::radians(-11.0f);
        camera_reset_requested = true;
        world.invalidate_contact_ao();
    }

    // Load the Architectural Clay GI Lighting Benchmark Courtyard (pure beveled clay + warm-peach / cool-cobalt GI!)
    void load_clay_gi_scene(renderer::WorldRenderer& world) {
        blank_scene = true;
        clay_scene = true;
        inspect_sky = false;
        entities.clear();
        undo_stack_.clear();
        selected = -1;
        if (clay_opaque_.id == 0) {
            const clay::ClayScene cs = clay::build_clay_scene();
            clay_opaque_ = world.upload(cs.opaque);
            clay_two_sided_ = world.upload(cs.two_sided);
            clay_water_ = world.upload(cs.water);
            clay_colliders_ = cs.colliders;
        }
        // Pure Physical Golden-Hour Palette: pure warm golden sun (R>G>B) + pure atmospheric sky blue (B>G>R)
        // ZERO green-channel magenta dip -> ZERO purple/normal-map tint on shadowed clay walls & bevels!
        sky.preset_id = 0;
        sky.sun_hour = 17.35f;
        sky.sun_intensity = 4.0f;
        sky.sun_color = {1.0f, 0.62f, 0.32f};
        sky.sky_zenith = {0.050f, 0.20f, 0.62f};
        sky.sky_horizon = {0.90f, 0.70f, 0.54f}; // peach sunset haze (art-direction reference)
        sky.ground_color = {0.72f, 0.55f, 0.38f};
        sky.ambient = 0.50f;
        sky.fog_density = 0.00014f;
        sky.exposure = 0.58f;
        sky.bloom = 0.11f;
        sky.saturation = 1.06f;

        requested_cam_pos = {0.14f, 1.48f, 9.85f};
        requested_cam_yaw = math::radians(-91.2f);
        requested_cam_pitch = math::radians(-0.6f);
        camera_reset_requested = true;
        world.invalidate_contact_ao();
        world.invalidate_shadows();
        toast("Loaded Clay GI Lighting Benchmark (F1 Hide UI | F5 Play)");
    }

    void push_clay_layers(std::vector<renderer::WorldDrawItem>& out) const {
        if (!clay_scene) return;
        if (clay_opaque_.id != 0) out.push_back({clay_opaque_, Mat4::identity(), false, true});
        if (clay_two_sided_.id != 0) out.push_back({clay_two_sided_, Mat4::identity(), true, true});
        if (clay_water_.id != 0) out.push_back({clay_water_, Mat4::identity(), false, false});
    }

    [[nodiscard]] const std::vector<village::Collider>& clay_colliders() const noexcept { return clay_colliders_; }

    // Ground height query for Play Mode walking & Editor drag-and-drop snapping!
    [[nodiscard]] f32 ground_height_at(f32 x, f32 z) const noexcept {
        if (clay_scene) {
            return clay::detail::ground_height(x, z);
        }
        if (!blank_scene) {
            return village::terrain_height(x, z);
        }
        bool hit = false;
        f32 best_y = -1e6f;
        for (const auto& e : entities) {
            if (e.kind == EntityKind::Plane) {
                const f32 hx = e.half_extents.x * e.scale.x;
                const f32 hz = e.half_extents.z * e.scale.z;
                if (std::abs(x - e.position.x) <= hx && std::abs(z - e.position.z) <= hz) {
                    best_y = std::max(best_y, e.position.y);
                    hit = true;
                }
            } else if (e.kind == EntityKind::Terrain) {
                const f32 lx = (x - e.position.x) / std::max(e.scale.x, 0.05f);
                const f32 lz = (z - e.position.z) / std::max(e.scale.z, 0.05f);
                const f32 half = e.terrain_size * 0.5f;
                if (std::abs(lx) <= half && std::abs(lz) <= half) {
                    const f32 ty = e.position.y + craft_terrain_height(lx, lz, e.terrain_amp, e.terrain_freq, e.terrain_seed) * e.scale.y;
                    best_y = std::max(best_y, ty);
                    hit = true;
                }
            }
        }
        return hit ? best_y : 0.0f;
    }

    void push_undo() {
        undo_stack_.push_back(entities);
        if (undo_stack_.size() > 32) {
            undo_stack_.erase(undo_stack_.begin());
        }
    }

    void undo(renderer::WorldRenderer& world) {
        if (undo_stack_.empty()) return;
        entities = std::move(undo_stack_.back());
        undo_stack_.pop_back();
        if (selected >= static_cast<int>(entities.size())) {
            selected = static_cast<int>(entities.size()) - 1;
        }
        world.invalidate_contact_ao();
        toast("Undo restored");
    }

    void enter_play_mode() {
        if (!editor_mode) return;
        play_snapshot_ = entities;
        editor_mode = false;
        dragging_prefab_ = -1;
        active_gizmo_axis_ = -1;
        active_slider_id_ = 0;
        toast("PLAY MODE (Press F5 or ESC to return to NOVA Craft)");
    }

    void stop_play_mode(renderer::WorldRenderer& world) {
        if (editor_mode) return;
        if (!play_snapshot_.empty()) {
            entities = play_snapshot_;
            if (selected >= static_cast<int>(entities.size())) {
                selected = static_cast<int>(entities.size()) - 1;
            }
            world.invalidate_contact_ao();
        }
        editor_mode = true;
        toast("Returned to NOVA Craft Editor");
    }

    void toast(std::string msg) {
        status_toast = std::move(msg);
        toast_timer = 3.2f;
    }

    // Apply live Sky & Atmosphere Studio settings + active 3D Point Light entity to WorldSettings!
    void apply_sky_and_lights(renderer::WorldSettings& s, bool flash_active = false) const noexcept {
        const f32 t = std::clamp((sky.sun_hour - 6.0f) / 13.0f, 0.0f, 1.0f);
        const f32 az = math::lerp(-1.25f, 0.64f, t);
        const f32 el = std::sin(t * math::kPi) * 0.92f + 0.14f;
        const f32 golden_t = std::clamp((sky.sun_hour - 15.0f) / 2.2f, 0.0f, 1.0f);
        if (clay_scene) {
            // Exact back-right golden-hour sun slicing cleanly through the courtyard gap onto the left house, doorstep & road!
            const f32 dh = (sky.sun_hour - 17.35f) * 0.24f;
            const Vec3 clay_base = math::normalize(Vec3{0.80f, 0.30f, -0.41f}); // low sun: long raking shadows
            const f32 c = std::cos(dh), sn = std::sin(dh);
            s.sun_direction = math::normalize(Vec3{clay_base.x * c - clay_base.z * sn,
                                                   std::clamp(clay_base.y - dh * 0.14f, 0.10f, 0.85f),
                                                   clay_base.x * sn + clay_base.z * c});
        } else {
            const Vec3 arc_dir = math::normalize(Vec3{std::sin(az) * std::cos(el), std::sin(el), -std::cos(az) * std::cos(el)});
            const Vec3 pass_dir = math::normalize(Vec3{0.58f, 0.24f, -0.78f});
            s.sun_direction = math::normalize(math::lerp(arc_dir, pass_dir, golden_t * 0.85f));
        }
        s.sun_color = sky.sun_color;
        s.sun_intensity = sky.sun_intensity;
        s.sky_zenith = sky.sky_zenith;
        s.sky_horizon = sky.sky_horizon;
        s.ground_color = sky.ground_color;
        s.ambient = sky.ambient;
        s.fog_density = sky.fog_density;
        s.exposure = sky.exposure;
        s.bloom = sky.bloom;
        s.saturation = sky.saturation;

        // Drive live dynamic Point Light from the selected (or first) PointLight entity when muzzle flash isn't active!
        if (!flash_active) {
            const CraftEntity* active_light = nullptr;
            if (selected >= 0 && selected < static_cast<int>(entities.size()) &&
                entities[static_cast<usize>(selected)].kind == EntityKind::PointLight) {
                active_light = &entities[static_cast<usize>(selected)];
            } else {
                for (const auto& e : entities) {
                    if (e.kind == EntityKind::PointLight) {
                        active_light = &e;
                        break;
                    }
                }
            }
            if (active_light) {
                s.point_light_position = active_light->position;
                s.point_light_color = active_light->light_color * active_light->light_intensity;
                s.point_light_range = active_light->light_range;
            } else {
                s.point_light_color = {0, 0, 0};
            }
        }
    }

    // Export current scene entities + UV + sun time as clean .nova.json
    bool save_scene_json(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            toast("Save failed!");
            return false;
        }
        std::fprintf(f, "{\n  \"engine\": \"NOVA PRISM\",\n  \"blank_scene\": %s,\n  \"sun_hour\": %.3f,\n  \"entities\": [\n",
                     blank_scene ? "true" : "false", sky.sun_hour);
        for (usize i = 0; i < entities.size(); ++i) {
            const auto& e = entities[i];
            std::fprintf(f,
                "    {\"name\":\"%s\",\"pos\":[%.3f,%.3f,%.3f],\"rot\":[%.2f,%.2f,%.2f],\"scale\":[%.3f,%.3f,%.3f],"
                "\"uv_tile\":[%.3f,%.3f],\"uv_offset\":[%.3f,%.3f],\"uv_rot\":%.2f,\"pom\":%.4f,\"world_lock\":%s}%s\n",
                e.name.c_str(), e.position.x, e.position.y, e.position.z,
                e.rotation_deg.x, e.rotation_deg.y, e.rotation_deg.z,
                e.scale.x, e.scale.y, e.scale.z,
                e.uv_tile.x, e.uv_tile.y, e.uv_offset.x, e.uv_offset.y,
                e.uv_rotation_deg, e.pom_depth, e.world_lock_uv ? "true" : "false",
                (i + 1 < entities.size()) ? "," : "");
        }
        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
        toast("Saved " + path);
        return true;
    }

    // Import a glTF / GLB 3D asset with intact node hierarchy, clean grounding, exact bounds, and zero POM UV distortion!
    bool import_gltf_model(const std::filesystem::path& fp, renderer::WorldRenderer& world, Vec3 spawn_pos,
                           f32 scale_override = 0.0f) {
        std::string err;
        auto model = renderer::load_gltf(fp, &err);
        if (!model.has_value()) {
            toast("Model import failed: " + err);
            return false;
        }
        const Vec3 bmin = model->bounds_min;
        const Vec3 bmax = model->bounds_max;
        const Vec3 bsize{bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z};
        const Vec3 bcenter{(bmin.x + bmax.x) * 0.5f, bmin.y, (bmin.z + bmax.z) * 0.5f};
        const f32 max_dim = std::max({bsize.x, bsize.y, bsize.z, 1e-4f});
        const f32 auto_scale = scale_override > 0.0f
            ? scale_override
            : ((max_dim > 6.0f || max_dim < 0.18f) ? (1.85f / max_dim) : 1.0f);
        const Mat4 norm_xf = math::scale({auto_scale, auto_scale, auto_scale})
                           * math::translate({-bcenter.x, -bcenter.y, -bcenter.z});

        std::vector<Vec3> prim_mins;
        std::vector<Vec3> prim_maxs;
        prim_mins.reserve(model->primitives.size());
        prim_maxs.reserve(model->primitives.size());
        for (auto& prim : model->primitives) {
            if (prim.mesh.vertices.empty() || prim.mesh.indices.empty()) continue;
            const Mat4 full_xf = norm_xf * prim.transform;
            const Vec3 cx{full_xf.m[0][0], full_xf.m[0][1], full_xf.m[0][2]};
            const Vec3 cy{full_xf.m[1][0], full_xf.m[1][1], full_xf.m[1][2]};
            const Vec3 cz{full_xf.m[2][0], full_xf.m[2][1], full_xf.m[2][2]};
            const f32 det3 = math::dot(cx, math::cross(cy, cz));
            Vec3 pmin{1e9f, 1e9f, 1e9f}, pmax{-1e9f, -1e9f, -1e9f};
            for (auto& v : prim.mesh.vertices) {
                v.position = math::transform_point(full_xf, v.position);
                const Vec3 nrm = math::transform_normal(full_xf, v.normal);
                const f32 nlen = math::length(nrm);
                v.normal = nlen > 1e-8f ? (nrm * (1.0f / nlen)) : Vec3{0.0f, 1.0f, 0.0f};
                const Vec3 tv{v.tangent.x, v.tangent.y, v.tangent.z};
                if (math::dot(tv, tv) > 1e-10f) {
                    Vec3 tt{cx.x * tv.x + cy.x * tv.y + cz.x * tv.z,
                            cx.y * tv.x + cy.y * tv.y + cz.y * tv.z,
                            cx.z * tv.x + cy.z * tv.y + cz.z * tv.z};
                    tt = tt - v.normal * math::dot(v.normal, tt);
                    const f32 tlen = math::length(tt);
                    if (tlen > 1e-8f) {
                        tt = tt * (1.0f / tlen);
                        v.tangent = {tt.x, tt.y, tt.z, det3 < 0.0f ? -v.tangent.w : v.tangent.w};
                    }
                }
                pmin = {std::min(pmin.x, v.position.x), std::min(pmin.y, v.position.y), std::min(pmin.z, v.position.z)};
                pmax = {std::max(pmax.x, v.position.x), std::max(pmax.y, v.position.y), std::max(pmax.z, v.position.z)};
            }
            if (det3 < 0.0f) {
                for (usize k = 0; k + 2 < prim.mesh.indices.size(); k += 3) {
                    std::swap(prim.mesh.indices[k + 1], prim.mesh.indices[k + 2]);
                }
            }
            prim.transform = Mat4::identity();
            prim_mins.push_back(pmin);
            prim_maxs.push_back(pmax);
        }

        const auto items = world.upload_model(*model, Mat4::identity());
        const u32 grp = items.size() > 1 ? next_group_id_++ : 0u;
        push_undo();
        for (usize i = 0; i < items.size(); ++i) {
            CraftEntity e;
            e.name = fp.stem().string() + (items.size() > 1 ? std::format("_{}", i) : "");
            e.mesh = items[i].mesh;
            e.material = items[i].material;
            e.position = spawn_pos;
            e.group_id = grp;
            if (i < prim_mins.size()) {
                const Vec3 mn = prim_mins[i], mx = prim_maxs[i];
                e.local_center = {(mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f, (mn.z + mx.z) * 0.5f};
                e.half_extents = {std::max((mx.x - mn.x) * 0.5f, 0.05f),
                                  std::max((mx.y - mn.y) * 0.5f, 0.05f),
                                  std::max((mx.z - mn.z) * 0.5f, 0.05f)};
            } else {
                e.local_center = {0.0f, bsize.y * auto_scale * 0.5f, 0.0f};
                e.half_extents = {std::max(bsize.x * auto_scale * 0.5f, 0.1f),
                                  std::max(bsize.y * auto_scale * 0.5f, 0.1f),
                                  std::max(bsize.z * auto_scale * 0.5f, 0.1f)};
            }
            // CRITICAL: glTF PBR models use authentic UV maps & AO textures — never run Steep POM on AO!
            e.pom_depth = 0.0f;
            e.world_lock_uv = false;
            e.two_sided = items[i].two_sided;
            e.casts_shadow = true;
            entities.push_back(std::move(e));
        }
        if (!entities.empty()) selected = static_cast<int>(entities.size()) - 1;
        world.invalidate_contact_ao();
        world.invalidate_shadows();
        toast("Imported 3D model (Studio PBR): " + fp.filename().string());
        return true;
    }

    // Handle Windows Explorer Drag & Drop (.glb / .gltf models OR .png / .jpg textures & 2x2 PBR atlases!)
    void handle_os_file_drops(const platform::Input& input, renderer::WorldRenderer& world,
                              const renderer::Camera& cam) {
        for (const std::string& path : input.dropped_files()) {
            std::filesystem::path fp(path);
            std::string ext = fp.extension().string();
            for (char& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

            if (ext == ".glb" || ext == ".gltf") {
                const Vec3 fwd = math::normalize(cam.target - cam.position);
                Vec3 spawn = cam.position + fwd * 4.2f;
                spawn.y = ground_height_at(spawn.x, spawn.z) + 0.01f;
                import_gltf_model(fp, world, spawn);
            } else if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga") {
                renderer::ImageData img;
                if (renderer::load_image_file(path, img)) {
                    // Auto-detect 2x2 PBR Atlas (Top-Left Albedo, Top-Right blue Normal, Bot-Left Roughness, Bot-Right Height)!
                    const bool is_square = (img.width >= 256 && img.width == img.height);
                    bool is_2x2_atlas = false;
                    if (is_square) {
                        const u32 hw = img.width / 2;
                        double b_sum = 0.0, r_sum = 0.0;
                        int samples = 0;
                        for (u32 y = 16; y < hw - 16; y += 16) {
                            for (u32 x = hw + 16; x < img.width - 16; x += 16) {
                                const usize idx = (static_cast<usize>(y) * img.width + x) * 4;
                                r_sum += img.rgba[idx + 0];
                                b_sum += img.rgba[idx + 2];
                                ++samples;
                            }
                        }
                        if (samples > 0 && (b_sum / samples) > 185.0 && (b_sum / samples) > (r_sum / samples) + 55.0) {
                            is_2x2_atlas = true;
                        }
                    }

                    auto crop_quad = [&](u32 x0, u32 y0, u32 w, u32 h) -> renderer::ImageData {
                        renderer::ImageData q;
                        q.width = w;
                        q.height = h;
                        q.rgba.resize(static_cast<usize>(w) * h * 4);
                        for (u32 y = 0; y < h; ++y) {
                            std::memcpy(q.rgba.data() + static_cast<usize>(y) * w * 4,
                                        img.rgba.data() + (static_cast<usize>(y0 + y) * img.width + x0) * 4,
                                        static_cast<usize>(w) * 4);
                        }
                        return q;
                    };

                    renderer::PbrMaterial pbr;
                    if (is_2x2_atlas) {
                        const u32 hw = img.width / 2;
                        const u32 pad = 2;
                        const u32 sz = hw - pad * 2;
                        auto q_alb = crop_quad(pad, pad, sz, sz);
                        auto q_nrm = crop_quad(hw + pad, pad, sz, sz);
                        auto q_rgh = crop_quad(pad, hw + pad, sz, sz);
                        auto q_hgt = crop_quad(hw + pad, hw + pad, sz, sz);
                        // Remap dielectric roughness matte & metallic = 0
                        for (usize i = 0; i < q_rgh.rgba.size(); i += 4) {
                            f32 r = static_cast<f32>(q_rgh.rgba[i + 1]) / 255.0f;
                            if (r < 0.35f) r = 0.94f;
                            const u8 g = static_cast<u8>(std::clamp((0.76f + 0.22f * r) * 255.0f, 0.0f, 255.0f));
                            q_rgh.rgba[i + 0] = g; q_rgh.rgba[i + 1] = g; q_rgh.rgba[i + 2] = 0; q_rgh.rgba[i + 3] = 255;
                        }
                        pbr.base_color = world.upload_texture(q_alb, true);
                        pbr.normal = world.upload_texture(q_nrm, false);
                        pbr.metallic_roughness = world.upload_texture(q_rgh, false);
                        pbr.occlusion = world.upload_texture(q_hgt, false);
                        pbr.metallic = 0.0f;
                        pbr.roughness = 1.0f;
                        pbr.normal_scale = 1.2f;
                        pbr.occlusion_strength = 0.72f;
                    } else {
                        pbr.base_color = world.upload_texture(img, true);
                        pbr.metallic = 0.0f;
                        pbr.roughness = 0.85f;
                    }
                    const renderer::MaterialId new_mat = world.add_material(pbr);
                    if (selected >= 0 && selected < static_cast<int>(entities.size())) {
                        push_undo();
                        entities[static_cast<usize>(selected)].material = new_mat;
                        toast(is_2x2_atlas ? ("Auto-sliced 4-Map PBR Atlas -> " + entities[static_cast<usize>(selected)].name)
                                           : ("Applied Texture -> " + entities[static_cast<usize>(selected)].name));
                    }
                }
            }
        }
    }

    // Update interactive editor UI, 3D gizmos, drag-and-drop & build UiOverlay vertices for this frame
    WindowAction update_and_draw(renderer::UiOverlay& ui, renderer::WorldRenderer& world,
                                 const platform::Input& input, const renderer::Camera& cam,
                                 u32 sw, u32 sh, f32 dt, f64 gpu_ms, bool shadow_cached,
                                 bool headless_capture, bool is_fullscreen, bool is_maximized) {
        WindowAction win_action = WindowAction::None;
        if (toast_timer > 0.0f) toast_timer = std::max(0.0f, toast_timer - dt);
        ui.begin(sw, sh);

        const f32 sc = std::clamp(static_cast<f32>(sh) / 1440.0f, 0.58f, 1.5f);
        const f32 mx = input.mouse_x();
        const f32 my = input.mouse_y();
        const bool lmb_down = input.is_down(platform::MouseButton::Left);
        const bool lmb_press = input.was_pressed(platform::MouseButton::Left);
        const bool lmb_rel = input.was_released(platform::MouseButton::Left);

        if (!lmb_down) {
            active_slider_id_ = 0;
            active_gizmo_axis_ = -1;
        }

        // During PLAY mode: render only a tiny translucent corner pill (omitted in headless captures!)
        if (!editor_mode) {
            if (!headless_capture) {
                const f32 pw = 350.0f * sc, ph = 36.0f * sc;
                const f32 px = (static_cast<f32>(sw) - pw) * 0.5f, py = 14.0f * sc;
                ui.rect_rounded(px, py, pw, ph, 10.0f * sc, {0.06f, 0.09f, 0.13f, 0.72f}, {0.25f, 0.85f, 0.55f, 0.55f}, 1.5f);
                const std::string lbl = "PLAYING  |  F5/ESC : EDITOR  |  F11 : FULLSCREEN";
                const f32 tw = ui.measure_text(lbl, 14.5f * sc);
                ui.text(px + (pw - tw) * 0.5f, py + 9.0f * sc, lbl, 14.5f * sc, {0.88f, 0.98f, 0.92f, 0.95f});
            }
            return win_action;
        }

        if (input.was_pressed(platform::Key::F1)) {
            ui_visible = !ui_visible;
        }
        if (!ui_visible) {
            const std::string hint = "F1 : Show NOVA Craft UI   |   F11 : Fullscreen";
            ui.rect_rounded(16 * sc, 16 * sc, 330 * sc, 34 * sc, 8 * sc, {0.06f, 0.08f, 0.12f, 0.75f});
            ui.text(28 * sc, 24 * sc, hint, 15 * sc, {0.85f, 0.90f, 0.96f, 0.9f});
            return win_action;
        }

        // Hotkeys (only when RMB fly camera is NOT held)
        const bool rmb = input.is_down(platform::MouseButton::Right);
        const bool ctrl = input.is_down(platform::Key::LeftCtrl) || input.is_down(platform::Key::RightCtrl);
        if (!rmb) {
            if (input.was_pressed(platform::Key::W)) gizmo_mode = GizmoMode::Move;
            if (input.was_pressed(platform::Key::E)) gizmo_mode = GizmoMode::Rotate;
            if (input.was_pressed(platform::Key::R)) gizmo_mode = GizmoMode::Scale;
            if (ctrl && input.was_pressed(platform::Key::Z)) undo(world);
            if (ctrl && input.was_pressed(platform::Key::S)) save_scene_json("village_scene.nova.json");
            if (ctrl && input.was_pressed(platform::Key::D) && selected >= 0 && selected < static_cast<int>(entities.size())) {
                if (!entities[static_cast<usize>(selected)].is_static_village) {
                    push_undo();
                    CraftEntity dup = entities[static_cast<usize>(selected)];
                    dup.name += "_copy";
                    dup.position.x += 0.9f;
                    entities.push_back(std::move(dup));
                    selected = static_cast<int>(entities.size()) - 1;
                    world.invalidate_contact_ao();
                    toast("Duplicated object");
                }
            }
            if (input.was_pressed(platform::Key::Delete) && selected >= 0 && selected < static_cast<int>(entities.size())) {
                if (!entities[static_cast<usize>(selected)].is_static_village) {
                    push_undo();
                    entities.erase(entities.begin() + selected);
                    selected = std::min(selected, static_cast<int>(entities.size()) - 1);
                    world.invalidate_contact_ao();
                    toast("Deleted object");
                } else {
                    toast("Village base layer cannot be deleted (edit UV/Material instead)");
                }
            }
        }

        // Layout metrics (clean, spacious, sleek at 2K)
        const f32 top_h   = 58.0f * sc;
        const f32 left_w  = 292.0f * sc;
        const f32 right_w = 360.0f * sc;
        const f32 pad     = 12.0f * sc;
        const f32 W = static_cast<f32>(sw);
        const f32 H = static_cast<f32>(sh);

        const bool over_top   = (my <= top_h + 6 * sc);
        const bool over_left  = (mx <= left_w + pad);
        const bool over_right = (mx >= W - right_w - pad);
        pointer_over_ui_ = over_top || over_left || over_right || (active_slider_id_ != 0) || (dragging_prefab_ >= 0);

        // Camera projection helpers for 3D Gizmos & Raycasting
        const f32 aspect = W / std::max(H, 1.0f);
        const Mat4 view = math::look_at(cam.position, cam.target, cam.up);
        const Mat4 proj = math::perspective(cam.fov_y, aspect, cam.z_near, cam.z_far);
        const Mat4 vp = proj * view;

        auto project_3d = [&](const Vec3& p, Vec2& out_px) -> bool {
            const Vec4 clip = vp * Vec4{p.x, p.y, p.z, 1.0f};
            if (clip.w <= 0.05f) return false;
            const f32 inv_w = 1.0f / clip.w;
            out_px.x = (clip.x * inv_w * 0.5f + 0.5f) * W;
            out_px.y = (clip.y * inv_w * 0.5f + 0.5f) * H;
            return true;
        };

        auto screen_ray = [&](f32 sx, f32 sy, Vec3& ro, Vec3& rd) {
            const Vec3 fwd = math::normalize(cam.target - cam.position);
            const Vec3 right = math::normalize(math::cross(fwd, cam.up));
            const Vec3 up = math::cross(right, fwd);
            const f32 tan_y = std::tan(cam.fov_y * 0.5f);
            const f32 tan_x = tan_y * aspect;
            const f32 ndc_x = (sx / W) * 2.0f - 1.0f;
            const f32 ndc_y = (sy / H) * 2.0f - 1.0f;
            ro = cam.position;
            rd = math::normalize(fwd + right * (ndc_x * tan_x) - up * (ndc_y * tan_y));
        };

        auto raycast_ground = [&](f32 sx, f32 sy) -> Vec3 {
            Vec3 ro, rd;
            screen_ray(sx, sy, ro, rd);
            f32 t = 7.5f;
            for (int step = 0; step < 72; ++step) {
                const f32 ft = 0.8f + static_cast<f32>(step) * 0.55f;
                const Vec3 p = ro + rd * ft;
                const f32 gh = ground_height_at(p.x, p.z) + 0.02f;
                if (p.y <= gh) {
                    t = ft;
                    break;
                }
            }
            Vec3 hit = ro + rd * t;
            hit.y = ground_height_at(hit.x, hit.z) + 0.02f;
            if (snap_enabled) {
                hit.x = std::round(hit.x * 4.0f) * 0.25f;
                hit.z = std::round(hit.z * 4.0f) * 0.25f;
            }
            return hit;
        };

        // ---- 1. 3D SELECTION WIREFRAME & INTERACTIVE MOVE / ROTATE / SCALE GIZMO (drawn behind panels!)
        if (!inspect_sky && selected >= 0 && selected < static_cast<int>(entities.size())) {
            CraftEntity& ent = entities[static_cast<usize>(selected)];
            const Mat4 M = ent.model_matrix();
            const Vec3 center = ent.world_center();

            // Draw sleek cyan-gold 3D wireframe bounding box around selected object (skip huge village/terrain meshes)
            if (!ent.is_static_village && ent.kind != EntityKind::Terrain) {
                const Vec3 c0 = ent.local_center;
                const Vec3 h  = ent.half_extents;
                Vec2 corners[8]{};
                bool vis[8]{};
                for (int i = 0; i < 8; ++i) {
                    const Vec3 lp{
                        c0.x + ((i & 1) ? h.x : -h.x),
                        c0.y + ((i & 2) ? h.y : -h.y),
                        c0.z + ((i & 4) ? h.z : -h.z),
                    };
                    vis[i] = project_3d(math::transform_point(M, lp), corners[i]);
                }
                constexpr int edges[12][2] = {
                    {0,1},{2,3},{4,5},{6,7},
                    {0,2},{1,3},{4,6},{5,7},
                    {0,4},{1,5},{2,6},{3,7}
                };
                for (const auto& e : edges) {
                    if (vis[e[0]] && vis[e[1]]) {
                        ui.line(corners[e[0]].x, corners[e[0]].y, corners[e[1]].x, corners[e[1]].y,
                                1.8f * sc, {0.32f, 0.85f, 1.0f, 0.72f});
                    }
                }
            }

            // Draw & interact with 3D RGB Transform Gizmo (for movable objects)
            Vec2 origin_px{};
            if (!ent.is_static_village && project_3d(center, origin_px)) {
                const f32 cam_dist = std::max(1.2f, math::length(cam.position - center));
                const f32 gizmo_len = cam_dist * 0.16f;
                const Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                const Vec4 cols[3] = {
                    {0.96f, 0.28f, 0.30f, 0.96f}, // X Red
                    {0.26f, 0.88f, 0.44f, 0.96f}, // Y Green
                    {0.28f, 0.60f, 0.98f, 0.96f}, // Z Blue
                };
                const char* axis_lbl[3] = {"X", "Y", "Z"};

                int hovered_axis = -1;
                Vec2 tips[3]{};
                bool tip_vis[3]{};
                for (int a = 0; a < 3; ++a) {
                    tip_vis[a] = project_3d(center + axes[a] * gizmo_len, tips[a]);
                    if (tip_vis[a] && !pointer_over_ui_) {
                        // Point-to-segment distance for crisp gizmo hit-testing
                        const Vec2 ab{tips[a].x - origin_px.x, tips[a].y - origin_px.y};
                        const f32 ab2 = ab.x * ab.x + ab.y * ab.y;
                        const f32 t = ab2 > 1.0f ? std::clamp(((mx - origin_px.x) * ab.x + (my - origin_px.y) * ab.y) / ab2, 0.0f, 1.15f) : 0.0f;
                        const f32 px = origin_px.x + ab.x * t, py = origin_px.y + ab.y * t;
                        if (std::hypot(mx - px, my - py) < 18.0f * sc) {
                            hovered_axis = a;
                        }
                    }
                }

                // Start gizmo drag
                if (lmb_press && hovered_axis >= 0 && !pointer_over_ui_) {
                    push_undo();
                    active_gizmo_axis_ = hovered_axis;
                    gizmo_start_mouse_ = {mx, my};
                    gizmo_start_pos_ = ent.position;
                    gizmo_start_rot_ = ent.rotation_deg;
                    gizmo_start_scale_ = ent.scale;
                }

                // Update active gizmo drag
                if (active_gizmo_axis_ >= 0 && lmb_down) {
                    const int a = active_gizmo_axis_;
                    Vec2 axis_dir{tips[a].x - origin_px.x, tips[a].y - origin_px.y};
                    const f32 alen = std::max(8.0f, std::hypot(axis_dir.x, axis_dir.y));
                    axis_dir = {axis_dir.x / alen, axis_dir.y / alen};
                    const f32 drag_px = (mx - gizmo_start_mouse_.x) * axis_dir.x + (my - gizmo_start_mouse_.y) * axis_dir.y;

                    if (gizmo_mode == GizmoMode::Move) {
                        f32 delta = (drag_px / alen) * gizmo_len;
                        if (snap_enabled) delta = std::round(delta * 4.0f) * 0.25f;
                        ent.position = gizmo_start_pos_ + axes[a] * delta;
                    } else if (gizmo_mode == GizmoMode::Rotate) {
                        f32 deg = ((mx - gizmo_start_mouse_.x) - (my - gizmo_start_mouse_.y)) * 0.35f;
                        if (snap_enabled) deg = std::round(deg / 15.0f) * 15.0f;
                        ent.rotation_deg = gizmo_start_rot_;
                        if (a == 0) ent.rotation_deg.x += deg;
                        if (a == 1) ent.rotation_deg.y += deg;
                        if (a == 2) ent.rotation_deg.z += deg;
                    } else if (gizmo_mode == GizmoMode::Scale) {
                        f32 factor = std::max(0.15f, 1.0f + (drag_px / alen) * 0.85f);
                        if (snap_enabled) factor = std::round(factor * 4.0f) * 0.25f;
                        ent.scale = gizmo_start_scale_;
                        if (a == 0) ent.scale.x = std::clamp(gizmo_start_scale_.x * factor, 0.1f, 12.0f);
                        if (a == 1) ent.scale.y = std::clamp(gizmo_start_scale_.y * factor, 0.1f, 12.0f);
                        if (a == 2) ent.scale.z = std::clamp(gizmo_start_scale_.z * factor, 0.1f, 12.0f);
                    }
                    sync_group_transform(ent);
                    world.invalidate_contact_ao();
                }

                // Render the 3D Gizmo visuals!
                if (gizmo_mode == GizmoMode::Rotate) {
                    const f32 ring_r = 68.0f * sc;
                    for (int a = 0; a < 3; ++a) {
                        const bool hi = (hovered_axis == a || active_gizmo_axis_ == a);
                        const Vec4 c = hi ? Vec4{1.0f, 0.94f, 0.36f, 1.0f} : cols[a];
                        const Vec3 u_ax = (a == 0) ? Vec3{0, 1, 0} : (a == 1 ? Vec3{1, 0, 0} : Vec3{1, 0, 0});
                        const Vec3 v_ax = (a == 0) ? Vec3{0, 0, 1} : (a == 1 ? Vec3{0, 0, 1} : Vec3{0, 1, 0});
                        constexpr int kRingSegs = 32;
                        Vec2 prev{};
                        bool prev_ok = false;
                        for (int s = 0; s <= kRingSegs; ++s) {
                            const f32 ang = (static_cast<f32>(s) / kRingSegs) * 2.0f * math::kPi;
                            const Vec3 wp = center + (u_ax * std::cos(ang) + v_ax * std::sin(ang)) * (gizmo_len * 0.82f);
                            Vec2 pt{};
                            const bool ok = project_3d(wp, pt);
                            if (ok && prev_ok) {
                                ui.line(prev.x, prev.y, pt.x, pt.y, (hi ? 4.5f : 2.8f) * sc, c);
                            }
                            prev = pt;
                            prev_ok = ok;
                        }
                        (void)ring_r;
                    }
                }
                for (int a = 0; a < 3; ++a) {
                    if (!tip_vis[a]) continue;
                    const bool hi = (hovered_axis == a || active_gizmo_axis_ == a);
                    const Vec4 c = hi ? Vec4{1.0f, 0.95f, 0.40f, 1.0f} : cols[a];
                    ui.line(origin_px.x, origin_px.y, tips[a].x, tips[a].y, (hi ? 5.0f : 3.5f) * sc, c);
                    const Vec2 d{tips[a].x - origin_px.x, tips[a].y - origin_px.y};
                    const f32 dl = std::max(1.0f, std::hypot(d.x, d.y));
                    const Vec2 dir{d.x / dl, d.y / dl};
                    const Vec2 perp{-dir.y, dir.x};
                    if (gizmo_mode == GizmoMode::Move) {
                        const Vec2 tip_end{tips[a].x + dir.x * (16.0f * sc), tips[a].y + dir.y * (16.0f * sc)};
                        const Vec2 b1{tips[a].x + perp.x * (7.5f * sc), tips[a].y + perp.y * (7.5f * sc)};
                        const Vec2 b2{tips[a].x - perp.x * (7.5f * sc), tips[a].y - perp.y * (7.5f * sc)};
                        ui.triangle(tip_end.x, tip_end.y, b1.x, b1.y, b2.x, b2.y, c);
                    } else if (gizmo_mode == GizmoMode::Scale) {
                        const f32 bs = 11.0f * sc;
                        ui.rect_rounded(tips[a].x - bs * 0.5f, tips[a].y - bs * 0.5f, bs, bs, 2.5f * sc, c);
                    } else {
                        ui.circle_filled(tips[a].x, tips[a].y, 6.5f * sc, c);
                    }
                    ui.text(tips[a].x + dir.x * 20.0f * sc - 5 * sc, tips[a].y + dir.y * 20.0f * sc - 8 * sc,
                            axis_lbl[a], 15.0f * sc, c);
                }
                ui.circle_filled(origin_px.x, origin_px.y, 7.0f * sc, {0.98f, 0.98f, 1.0f, 0.95f});
                ui.circle_outline(origin_px.x, origin_px.y, 9.5f * sc, 2.0f * sc, {0.10f, 0.14f, 0.22f, 0.9f});
            }
        }

        // ---- 2. VIEWPORT CLICK SELECTION (when clicking empty 3D scene outside UI & gizmo)
        if (lmb_press && !pointer_over_ui_ && active_gizmo_axis_ < 0 && dragging_prefab_ < 0) {
            Vec3 ro, rd;
            screen_ray(mx, my, ro, rd);
            int best_idx = -1;
            f32 best_t = 1e9f;
            for (usize i = 0; i < entities.size(); ++i) {
                const auto& e = entities[i];
                if (e.is_static_village || e.kind == EntityKind::Terrain) continue;
                const Vec3 c = e.world_center();
                const Vec3 he = {e.half_extents.x * e.scale.x + 0.1f, e.half_extents.y * e.scale.y + 0.1f,
                                 e.half_extents.z * e.scale.z + 0.1f};
                const f32 rad = std::min(math::length(he), 4.0f);
                const Vec3 oc = ro - c;
                const f32 b = math::dot(oc, rd);
                const f32 disc = b * b - (math::dot(oc, oc) - rad * rad);
                if (disc >= 0.0f) {
                    const f32 t = -b - std::sqrt(disc);
                    if (t > 0.1f && t < best_t) {
                        best_t = t;
                        best_idx = static_cast<int>(i);
                    }
                }
            }
            if (best_idx >= 0) {
                selected = best_idx;
                inspect_sky = false;
            }
        }

        // ---- 3. DRAG & DROP LIVE VIEWPORT GHOST + SPAWN ON RELEASE!
        if (dragging_prefab_ >= 0 && dragging_prefab_ < static_cast<int>(prefabs_.size())) {
            const auto& pf = prefabs_[static_cast<usize>(dragging_prefab_)];
            if (!over_left && !over_right && !over_top) {
                const Vec3 drop_pt = raycast_ground(mx, my);
                Vec2 pt_px{};
                if (project_3d(drop_pt, pt_px)) {
                    ui.circle_outline(pt_px.x, pt_px.y, 26.0f * sc, 2.5f * sc, {0.20f, 0.92f, 0.72f, 0.95f});
                    ui.circle_filled(pt_px.x, pt_px.y, 5.0f * sc, {0.20f, 0.92f, 0.72f, 1.0f});
                }
                if (lmb_rel) {
                    push_undo();
                    CraftEntity e;
                    e.name = std::format("{}_{:02d}", pf.label, static_cast<int>(entities.size()));
                    e.mesh = pf.mesh;
                    e.material = pf.material;
                    e.position = drop_pt + Vec3{0, pf.y_offset, 0};
                    e.scale = pf.default_scale;
                    e.local_center = pf.local_center;
                    e.half_extents = pf.half_extents;
                    e.pom_depth = pf.pom_depth;
                    e.casts_shadow = pf.casts_shadow;
                    e.has_collider = pf.has_collider;
                    e.world_lock_uv = true;
                    entities.push_back(std::move(e));
                    selected = static_cast<int>(entities.size()) - 1;
                    world.invalidate_contact_ao();
                    toast("Dropped " + pf.label + " into scene!");
                    dragging_prefab_ = -1;
                }
            } else if (lmb_rel) {
                dragging_prefab_ = -1;
            }
        }

        // UI Color Palette (Sleek Obsidian Glass + Cyan/Emerald/Amber Studio Accents)
        const Vec4 col_panel  {0.065f, 0.082f, 0.115f, 0.88f};
        const Vec4 col_card   {0.105f, 0.130f, 0.180f, 0.92f};
        const Vec4 col_card_hi{0.155f, 0.205f, 0.285f, 0.96f};
        const Vec4 col_border {0.42f,  0.54f,  0.72f,  0.26f};
        const Vec4 col_accent {0.18f,  0.78f,  0.96f,  0.96f};
        const Vec4 col_green  {0.14f,  0.76f,  0.46f,  0.96f};
        const Vec4 col_amber  {0.96f,  0.68f,  0.22f,  0.96f};
        const Vec4 col_text   {0.92f,  0.95f,  0.98f,  0.98f};
        const Vec4 col_muted  {0.62f,  0.70f,  0.80f,  0.92f};

        auto btn = [&](f32 x, f32 y, f32 w, f32 h, std::string_view label, bool active, const Vec4& accent,
                       f32 fsz = 15.0f) -> bool {
            const bool hov = (mx >= x && mx <= x + w && my >= y && my <= y + h);
            const Vec4 bg = active ? accent : (hov ? col_card_hi : col_card);
            const Vec4 bd = active ? Vec4{1, 1, 1, 0.45f} : (hov ? Vec4{accent.x, accent.y, accent.z, 0.7f} : col_border);
            ui.rect_rounded(x, y, w, h, 8.0f * sc, bg, bd, 1.4f * sc);
            const f32 tw = ui.measure_text(label, fsz * sc);
            const Vec4 tc = active ? Vec4{0.04f, 0.06f, 0.09f, 1.0f} : col_text;
            ui.text(x + (w - tw) * 0.5f, y + (h - fsz * sc) * 0.5f - 1.0f * sc, label, fsz * sc, tc);
            return hov && lmb_press;
        };

        auto slider = [&](u32 id, f32 x, f32 y, f32 w, f32 h, std::string_view label, f32& val,
                          f32 vmin, f32 vmax, const Vec4& bar_col, const char* fmt = "%.2f") -> bool {
            const bool hov = (mx >= x && mx <= x + w && my >= y && my <= y + h);
            if (hov && lmb_press) {
                active_slider_id_ = id;
            }
            bool changed = false;
            if (active_slider_id_ == id && lmb_down) {
                const f32 t = std::clamp((mx - x) / std::max(w, 1.0f), 0.0f, 1.0f);
                const f32 nv = vmin + (vmax - vmin) * t;
                if (std::abs(nv - val) > 1e-5f) {
                    val = nv;
                    changed = true;
                }
            }
            const f32 frac = std::clamp((val - vmin) / std::max(vmax - vmin, 1e-5f), 0.0f, 1.0f);
            ui.rect_rounded(x, y, w, h, 6.0f * sc, {0.045f, 0.058f, 0.085f, 0.94f},
                            (hov || active_slider_id_ == id) ? Vec4{bar_col.x, bar_col.y, bar_col.z, 0.8f} : col_border, 1.2f * sc);
            if (frac > 0.01f) {
                ui.rect_rounded(x + 2 * sc, y + 2 * sc, std::max(4 * sc, (w - 4 * sc) * frac), h - 4 * sc,
                                4.5f * sc, {bar_col.x, bar_col.y, bar_col.z, 0.42f});
            }
            ui.text(x + 9 * sc, y + (h - 14.5f * sc) * 0.5f - 1 * sc, label, 14.5f * sc, col_muted);
            char vbuf[32]{};
            std::snprintf(vbuf, sizeof(vbuf), fmt, val);
            const f32 vw = ui.measure_text(vbuf, 14.5f * sc);
            ui.text(x + w - vw - 9 * sc, y + (h - 14.5f * sc) * 0.5f - 1 * sc, vbuf, 14.5f * sc, col_text);
            return changed;
        };

        // ============================================================ TOP TOOLBAR
        ui.rect(0, 0, W, top_h, {0.050f, 0.065f, 0.095f, 0.92f});
        ui.rect(0, top_h - 1.5f * sc, W, 1.5f * sc, col_border);

        f32 tx = 14.0f * sc;
        const f32 by = 10.0f * sc, bh = top_h - 20.0f * sc;

        // Brand badge
        ui.rect_rounded(tx, by, 146 * sc, bh, 8 * sc, {0.10f, 0.16f, 0.24f, 0.95f}, {0.18f, 0.78f, 0.96f, 0.65f}, 1.5f * sc);
        ui.text(tx + 14 * sc, by + 9 * sc, "NOVA PRISM", 17.0f * sc, col_accent);
        tx += 158 * sc;

        // ▶ PLAY button (instant zero-restart live FPS gameplay!)
        if (btn(tx, by, 108 * sc, bh, ">  PLAY", true, col_green, 15.5f)) {
            enter_play_mode();
        }
        tx += 118 * sc;

        // Scene Switcher: NEW SCENE | CLAY GI (Pure Lighting Benchmark) | VILLAGE
        if (btn(tx, by, 86 * sc, bh, "+ NEW", blank_scene && !clay_scene, {0.58f, 0.36f, 0.96f, 0.96f}, 13.5f)) {
            create_new_scene(world, true);
        }
        tx += 92 * sc;
        if (btn(tx, by, 86 * sc, bh, "CLAY GI", clay_scene, {0.96f, 0.62f, 0.34f, 0.96f}, 13.5f)) {
            load_clay_gi_scene(world);
        }
        tx += 92 * sc;
        if (btn(tx, by, 84 * sc, bh, "VILLAGE", !blank_scene, col_amber, 13.5f)) {
            load_village_scene(world);
        }
        tx += 92 * sc;

        // Transform mode pills: [W] MOVE | [E] ROTATE | [R] SCALE
        if (btn(tx, by, 90 * sc, bh, "[W] MOVE", gizmo_mode == GizmoMode::Move, col_accent, 14.0f)) gizmo_mode = GizmoMode::Move;
        tx += 96 * sc;
        if (btn(tx, by, 98 * sc, bh, "[E] ROT", gizmo_mode == GizmoMode::Rotate, col_accent, 14.0f)) gizmo_mode = GizmoMode::Rotate;
        tx += 104 * sc;
        if (btn(tx, by, 94 * sc, bh, "[R] SCALE", gizmo_mode == GizmoMode::Scale, col_accent, 14.0f)) gizmo_mode = GizmoMode::Scale;
        tx += 104 * sc;

        if (btn(tx, by, 66 * sc, bh, "SNAP", snap_enabled, col_amber, 13.5f)) snap_enabled = !snap_enabled;
        tx += 72 * sc;
        if (btn(tx, by, 62 * sc, bh, "DUP", false, col_accent, 13.5f) && selected >= 0 &&
            selected < static_cast<int>(entities.size()) && !entities[static_cast<usize>(selected)].is_static_village) {
            push_undo();
            CraftEntity dup = entities[static_cast<usize>(selected)];
            dup.name += "_copy";
            dup.position.x += 0.9f;
            entities.push_back(std::move(dup));
            selected = static_cast<int>(entities.size()) - 1;
            world.invalidate_contact_ao();
            toast("Duplicated object");
        }
        tx += 68 * sc;
        if (btn(tx, by, 60 * sc, bh, "DEL", false, {0.92f, 0.30f, 0.30f, 1}, 13.5f) && selected >= 0 &&
            selected < static_cast<int>(entities.size()) && !entities[static_cast<usize>(selected)].is_static_village) {
            push_undo();
            entities.erase(entities.begin() + selected);
            selected = std::min(selected, static_cast<int>(entities.size()) - 1);
            world.invalidate_contact_ao();
            toast("Deleted object");
        }
        tx += 66 * sc;
        if (btn(tx, by, 66 * sc, bh, "SAVE", false, col_green, 13.5f)) {
            save_scene_json(blank_scene ? "custom_scene.nova.json" : "village_scene.nova.json");
        }
        tx += 76 * sc;

        // Live Sun Time-of-Day slider right in the top bar!
        slider(1001, tx, by, 175 * sc, bh, "SUN HR", sky.sun_hour, 6.5f, 18.5f, col_amber, "%.1fh");

        // Top-right Window Controls: [FS Fullscreen]  [— Minimize]  [□ Maximize/Restore]  [× Close]
        const f32 win_btn_w = 40.0f * sc;
        const f32 fs_btn_w  = 56.0f * sc;
        const f32 win_gap   = 6.0f * sc;
        f32 wx = W - 14.0f * sc;

        // 1. Close button [×]
        wx -= win_btn_w;
        {
            const bool hov = (mx >= wx && mx <= wx + win_btn_w && my >= by && my <= by + bh);
            ui.rect_rounded(wx, by, win_btn_w, bh, 7 * sc,
                            hov ? Vec4{0.86f, 0.18f, 0.22f, 0.98f} : Vec4{0.14f, 0.12f, 0.16f, 0.92f},
                            hov ? Vec4{1.0f, 0.45f, 0.48f, 0.85f} : col_border, 1.3f * sc);
            const f32 cx = wx + win_btn_w * 0.5f, cy_b = by + bh * 0.5f, arm = 6.5f * sc;
            const Vec4 ic = hov ? Vec4{1, 1, 1, 1} : Vec4{0.92f, 0.82f, 0.84f, 0.95f};
            ui.line(cx - arm, cy_b - arm, cx + arm, cy_b + arm, 2.2f * sc, ic);
            ui.line(cx + arm, cy_b - arm, cx - arm, cy_b + arm, 2.2f * sc, ic);
            if (hov && lmb_press) win_action = WindowAction::Close;
        }
        wx -= (win_btn_w + win_gap);

        // 2. Maximize / Restore button [□]
        {
            const bool hov = (mx >= wx && mx <= wx + win_btn_w && my >= by && my <= by + bh);
            ui.rect_rounded(wx, by, win_btn_w, bh, 7 * sc, hov ? col_card_hi : col_card,
                            hov ? col_accent : col_border, 1.3f * sc);
            const f32 cx = wx + win_btn_w * 0.5f, cy_b = by + bh * 0.5f, sz = 12.0f * sc;
            const Vec4 ic = hov ? Vec4{1, 1, 1, 1} : col_text;
            if (is_maximized) {
                ui.rect_outline(cx - sz * 0.5f + 2.5f * sc, cy_b - sz * 0.5f - 2.0f * sc, sz - 2 * sc, sz - 2 * sc, 1.7f * sc, col_muted);
                ui.rect_outline(cx - sz * 0.5f - 1.5f * sc, cy_b - sz * 0.5f + 2.0f * sc, sz - 2 * sc, sz - 2 * sc, 1.9f * sc, ic);
            } else {
                ui.rect_outline(cx - sz * 0.5f, cy_b - sz * 0.5f, sz, sz, 1.9f * sc, ic);
            }
            if (hov && lmb_press) win_action = WindowAction::ToggleMaximize;
        }
        wx -= (win_btn_w + win_gap);

        // 3. Minimize button [—]
        {
            const bool hov = (mx >= wx && mx <= wx + win_btn_w && my >= by && my <= by + bh);
            ui.rect_rounded(wx, by, win_btn_w, bh, 7 * sc, hov ? col_card_hi : col_card,
                            hov ? col_accent : col_border, 1.3f * sc);
            const f32 cx = wx + win_btn_w * 0.5f, cy_b = by + bh * 0.5f + 3.0f * sc, arm = 6.5f * sc;
            ui.line(cx - arm, cy_b, cx + arm, cy_b, 2.2f * sc, hov ? Vec4{1, 1, 1, 1} : col_text);
            if (hov && lmb_press) win_action = WindowAction::Minimize;
        }
        wx -= (fs_btn_w + win_gap + 4.0f * sc);

        // 4. Fullscreen Choice toggle [FS] (also F11!)
        if (btn(wx, by, fs_btn_w, bh, is_fullscreen ? "WIN" : "FULL", is_fullscreen, col_accent, 13.5f)) {
            win_action = WindowAction::ToggleFullscreen;
        }

        // Live PRISM-VSM & GPU ms badge neatly to the left of Window Controls
        const f64 ms = gpu_ms > 0.0 ? gpu_ms : 10.6;
        const int fps = static_cast<int>(std::round(1000.0 / std::max(ms, 0.1)));
        const std::string perf = std::format("PRISM-VSM: {}  |  {:.1f} ms ({} FPS)",
                                             shadow_cached ? "CACHED" : "LIVE", ms, fps);
        const f32 pw = ui.measure_text(perf, 14.0f * sc) + 24 * sc;
        const f32 px_badge = wx - pw - 12 * sc;
        if (px_badge > tx + 205 * sc) {
            ui.rect_rounded(px_badge, by, pw, bh, 8 * sc, {0.08f, 0.12f, 0.17f, 0.94f},
                            shadow_cached ? Vec4{0.18f, 0.84f, 0.54f, 0.55f} : Vec4{0.96f, 0.68f, 0.22f, 0.65f}, 1.4f * sc);
            ui.text(px_badge + 12 * sc, by + 10 * sc, perf, 14.0f * sc, shadow_cached ? Vec4{0.42f, 0.95f, 0.68f, 1} : col_amber);
        }

        // ============================================================ LEFT PANEL: CREATE + ASSETS + SCENE LIST
        const f32 lx = pad, ly = top_h + pad;
        const f32 lh = H - top_h - pad * 2.0f;
        ui.rect_rounded(lx, ly, left_w, lh, 12.0f * sc, col_panel, col_border, 1.5f * sc);

        f32 cy = ly + 12 * sc;
        ui.text(lx + 14 * sc, cy, "CREATE WORLD", 14.5f * sc, col_amber);
        cy += 22 * sc;
        // 4 Dedicated 1-Click Creator Pills: + PLANE | + TERRAIN | + LIGHT | SKY & SUN
        const f32 cw2 = (left_w - 28 * sc - 8 * sc) * 0.5f;
        const f32 ch2 = 30 * sc;
        if (btn(lx + 14 * sc,            cy, cw2, ch2, "+ PLANE",   false, col_accent, 13.5f)) {
            const Vec3 fwd = math::normalize(cam.target - cam.position);
            Vec3 p = cam.position + Vec3{fwd.x, 0, fwd.z} * 6.0f;
            p.y = ground_height_at(p.x, p.z) + 0.02f;
            spawn_plane(world, p);
        }
        if (btn(lx + 14 * sc + cw2 + 8*sc, cy, cw2, ch2, "+ TERRAIN", false, col_green, 13.5f)) {
            spawn_terrain(world, {0.0f, -0.35f, -10.0f}, 18.0f, 0.032f, 42.0f + static_cast<f32>(entities.size()) * 7.0f, true);
        }
        cy += ch2 + 7 * sc;
        if (btn(lx + 14 * sc,            cy, cw2, ch2, "+ LIGHT",   false, col_amber, 13.5f)) {
            const Vec3 fwd = math::normalize(cam.target - cam.position);
            Vec3 p = cam.position + fwd * 5.0f;
            p.y = std::max(p.y, ground_height_at(p.x, p.z) + 1.8f);
            spawn_point_light(world, p);
        }
        if (btn(lx + 14 * sc + cw2 + 8*sc, cy, cw2, ch2, "SKY & SUN", inspect_sky, {0.35f, 0.65f, 1.0f, 1.0f}, 13.5f)) {
            inspect_sky = !inspect_sky;
        }
        cy += ch2 + 12 * sc;
        ui.rect(lx + 14 * sc, cy, left_w - 28 * sc, 1.2f * sc, col_border);
        cy += 10 * sc;

        ui.text(lx + 14 * sc, cy, "ASSETS  (DRAG & DROP)", 14.5f * sc, col_accent);
        cy += 22 * sc;

        // 2-column PBR asset card grid with REAL GPU textures!
        const f32 card_gap = 8 * sc;
        const f32 card_w = (left_w - 28 * sc - card_gap) * 0.5f;
        const f32 card_h = 98 * sc;
        for (usize i = 0; i < prefabs_.size(); ++i) {
            const int col_i = static_cast<int>(i % 2);
            const int row_i = static_cast<int>(i / 2);
            const f32 cx = lx + 14 * sc + static_cast<f32>(col_i) * (card_w + card_gap);
            const f32 ry = cy + static_cast<f32>(row_i) * (card_h + card_gap);
            const bool hov = (mx >= cx && mx <= cx + card_w && my >= ry && my <= ry + card_h);
            const bool dragging_this = (dragging_prefab_ == static_cast<int>(i));

            ui.rect_rounded(cx, ry, card_w, card_h, 8 * sc, hov ? col_card_hi : col_card,
                            (hov || dragging_this) ? col_accent : col_border, 1.4f * sc);
            const f32 th_h = 56 * sc;
            if (prefabs_[i].thumb.valid()) {
                ui.textured_rect(cx + 5 * sc, ry + 5 * sc, card_w - 10 * sc, th_h, prefabs_[i].thumb);
            } else {
                ui.rect_rounded(cx + 5 * sc, ry + 5 * sc, card_w - 10 * sc, th_h, 6 * sc, prefabs_[i].fallback_tint);
            }
            ui.text(cx + 7 * sc, ry + th_h + 8 * sc, prefabs_[i].label, 13.5f * sc, col_text);
            ui.text(cx + 7 * sc, ry + th_h + 23 * sc, prefabs_[i].sublabel, 11.0f * sc, col_muted);

            if (hov && lmb_press) {
                dragging_prefab_ = static_cast<int>(i);
            }
        }
        cy += static_cast<f32>((prefabs_.size() + 1) / 2) * (card_h + card_gap) + 10 * sc;

        // Scene Hierarchy Outliner (Sky & Sun + all Terrain / Plane / Light / PBR entities!)
        ui.rect(lx + 14 * sc, cy, left_w - 28 * sc, 1.2f * sc, col_border);
        cy += 10 * sc;
        ui.text(lx + 14 * sc, cy, "SCENE OBJECTS", 14.0f * sc, col_accent);
        cy += 22 * sc;

        const f32 row_h = 28 * sc;
        // Dedicated top outliner entry for Sky & Sun Atmosphere
        {
            const f32 rx = lx + 12 * sc, rw = left_w - 24 * sc;
            const bool hov = (mx >= rx && mx <= rx + rw && my >= cy && my <= cy + row_h);
            ui.rect_rounded(rx, cy, rw, row_h - 3 * sc, 6 * sc,
                            inspect_sky ? Vec4{0.25f, 0.62f, 1.0f, 0.30f} : (hov ? col_card_hi : Vec4{0.08f, 0.11f, 0.16f, 0.72f}),
                            inspect_sky ? Vec4{0.35f, 0.72f, 1.0f, 0.9f} : Vec4{0, 0, 0, 0}, 1.3f * sc);
            ui.circle_filled(rx + 12 * sc, cy + (row_h - 3 * sc) * 0.5f, 4.0f * sc, {0.35f, 0.76f, 1.0f, 1.0f});
            ui.text(rx + 24 * sc, cy + 4 * sc, "Sky_&_Sun_Atmosphere", 13.0f * sc, inspect_sky ? col_text : col_muted);
            if (hov && lmb_press) {
                inspect_sky = true;
            }
            cy += row_h;
        }
        for (usize i = 0; i < entities.size() && cy + row_h < ly + lh - 10 * sc; ++i) {
            const bool is_sel = (!inspect_sky && static_cast<int>(i) == selected);
            const f32 rx = lx + 12 * sc, rw = left_w - 24 * sc;
            const bool hov = (mx >= rx && mx <= rx + rw && my >= cy && my <= cy + row_h);
            ui.rect_rounded(rx, cy, rw, row_h - 3 * sc, 6 * sc,
                            is_sel ? Vec4{0.18f, 0.78f, 0.96f, 0.28f} : (hov ? col_card_hi : Vec4{0.08f, 0.10f, 0.14f, 0.65f}),
                            is_sel ? col_accent : Vec4{0, 0, 0, 0}, 1.3f * sc);
            Vec4 dot = col_green;
            if (entities[i].kind == EntityKind::PointLight) dot = {1.0f, 0.85f, 0.25f, 1.0f};
            else if (entities[i].kind == EntityKind::Terrain) dot = {0.35f, 0.88f, 0.55f, 1.0f};
            else if (entities[i].kind == EntityKind::Plane) dot = col_accent;
            else if (entities[i].is_static_village) dot = col_amber;
            ui.circle_filled(rx + 12 * sc, cy + (row_h - 3 * sc) * 0.5f, 4.0f * sc, dot);
            ui.text(rx + 24 * sc, cy + 4 * sc, entities[i].name, 13.0f * sc, is_sel ? col_text : col_muted);
            if (hov && lmb_press) {
                selected = static_cast<int>(i);
                inspect_sky = false;
            }
            cy += row_h;
        }

        // ============================================================ RIGHT PANEL: CONTEXT-AWARE INSPECTOR
        const f32 rx = W - right_w - pad;
        const f32 ry_top = top_h + pad;
        ui.rect_rounded(rx, ry_top, right_w, lh, 12.0f * sc, col_panel, col_border, 1.5f * sc);

        f32 iy = ry_top + 14 * sc;
        const f32 ix = rx + 14 * sc;
        const f32 iw = right_w - 28 * sc;
        const f32 s3_gap = 6 * sc;
        const f32 s3_w = (iw - s3_gap * 2.0f) / 3.0f;
        const f32 sh_row = 28 * sc;
        const Vec4 cx_col{0.95f, 0.30f, 0.32f, 1}, cy_col{0.26f, 0.86f, 0.44f, 1}, cz_col{0.28f, 0.60f, 0.98f, 1};

        ui.text(ix, iy, inspect_sky ? "SKY & LIGHTING STUDIO" : "INSPECTOR", 15.5f * sc, col_accent);
        if (inspect_sky) {
            iy += 28 * sc;
            ui.text(ix, iy, "ATMOSPHERE PRESETS (1-CLICK)", 13.0f * sc, col_amber);
            iy += 22 * sc;
            const f32 p4_w = (iw - 18 * sc) / 4.0f;
            if (btn(ix,                    iy, p4_w, 30 * sc, "GOLDEN", sky.preset_id == 0, col_amber, 12.5f)) apply_sky_preset(0);
            if (btn(ix + (p4_w + 6 * sc),  iy, p4_w, 30 * sc, "NOON",   sky.preset_id == 1, col_accent, 12.5f)) apply_sky_preset(1);
            if (btn(ix + (p4_w + 6 * sc)*2,iy, p4_w, 30 * sc, "SUNSET", sky.preset_id == 2, {0.96f, 0.44f, 0.24f, 1}, 12.5f)) apply_sky_preset(2);
            if (btn(ix + (p4_w + 6 * sc)*3,iy, p4_w, 30 * sc, "NIGHT",  sky.preset_id == 3, {0.42f, 0.56f, 0.98f, 1}, 12.5f)) apply_sky_preset(3);
            iy += 42 * sc;

            ui.text(ix, iy, "SUN & DIRECTIONAL LIGHT", 13.0f * sc, col_amber);
            iy += 22 * sc;
            slider(5001, ix, iy, iw, sh_row, "Sun Time (Hour)", sky.sun_hour, 6.2f, 18.8f, col_amber, "%.2f");
            iy += sh_row + 7 * sc;
            slider(5002, ix, iy, iw, sh_row, "Sun Intensity",   sky.sun_intensity, 0.1f, 6.0f, col_amber, "%.2f");
            iy += sh_row + 7 * sc;
            slider(5003, ix,                    iy, s3_w, sh_row, "SR", sky.sun_color.x, 0.1f, 1.2f, cx_col, "%.2f");
            slider(5004, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "SG", sky.sun_color.y, 0.1f, 1.2f, cy_col, "%.2f");
            slider(5005, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "SB", sky.sun_color.z, 0.1f, 1.2f, cz_col, "%.2f");
            iy += sh_row + 14 * sc;

            ui.text(ix, iy, "SKY ZENITH & HORIZON", 13.0f * sc, col_accent);
            iy += 22 * sc;
            slider(5011, ix,                    iy, s3_w, sh_row, "ZR", sky.sky_zenith.x, 0.0f, 0.8f, cx_col, "%.2f");
            slider(5012, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "ZG", sky.sky_zenith.y, 0.0f, 0.8f, cy_col, "%.2f");
            slider(5013, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "ZB", sky.sky_zenith.z, 0.0f, 1.0f, cz_col, "%.2f");
            iy += sh_row + 7 * sc;
            slider(5014, ix,                    iy, s3_w, sh_row, "HR", sky.sky_horizon.x, 0.0f, 1.0f, cx_col, "%.2f");
            slider(5015, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "HG", sky.sky_horizon.y, 0.0f, 1.0f, cy_col, "%.2f");
            slider(5016, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "HB", sky.sky_horizon.z, 0.0f, 1.0f, cz_col, "%.2f");
            iy += sh_row + 14 * sc;

            ui.text(ix, iy, "ATMOSPHERE & CAMERA", 13.0f * sc, col_green);
            iy += 22 * sc;
            slider(5021, ix, iy, iw, sh_row, "Ambient Fill",  sky.ambient,     0.1f, 1.4f, col_green, "%.2f");
            iy += sh_row + 7 * sc;
            slider(5022, ix, iy, iw, sh_row, "Fog Density",   sky.fog_density, 0.0f, 0.0020f, col_green, "%.5f");
            iy += sh_row + 7 * sc;
            slider(5023, ix, iy, iw, sh_row, "Exposure",      sky.exposure,    0.3f, 1.4f, col_accent, "%.2f");
            iy += sh_row + 7 * sc;
            slider(5024, ix, iy, iw, sh_row, "Bloom Glow",    sky.bloom,       0.0f, 0.5f, col_amber, "%.2f");
            iy += sh_row + 7 * sc;
            slider(5025, ix, iy, iw, sh_row, "Saturation",    sky.saturation,  0.5f, 1.6f, col_accent, "%.2f");
        } else if (selected >= 0 && selected < static_cast<int>(entities.size())) {
            CraftEntity& ent = entities[static_cast<usize>(selected)];
            // Selected object title pill
            iy += 26 * sc;
            ui.rect_rounded(ix, iy, iw, 34 * sc, 7 * sc, col_card, col_border, 1.2f * sc);
            ui.text(ix + 10 * sc, iy + 8 * sc, ent.name, 15.0f * sc, col_text);
            iy += 44 * sc;

            // ---- TRANSFORM SECTION (Move / Rotate / Scale)
            ui.text(ix, iy, "TRANSFORM  (MOVE / ROTATE / SCALE)", 13.5f * sc, col_amber);
            iy += 22 * sc;
            bool xform_changed = false;

            xform_changed |= slider(2001, ix,                    iy, s3_w, sh_row, "PX", ent.position.x, -45.0f, 45.0f, cx_col, "%.2f");
            xform_changed |= slider(2002, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "PY", ent.position.y, -5.0f,  25.0f, cy_col, "%.2f");
            xform_changed |= slider(2003, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "PZ", ent.position.z, -45.0f, 35.0f, cz_col, "%.2f");
            iy += sh_row + 7 * sc;

            if (ent.kind != EntityKind::PointLight) {
                xform_changed |= slider(2011, ix,                    iy, s3_w, sh_row, "RX", ent.rotation_deg.x, -180.0f, 180.0f, cx_col, "%.0f");
                xform_changed |= slider(2012, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "RY", ent.rotation_deg.y, -180.0f, 180.0f, cy_col, "%.0f");
                xform_changed |= slider(2013, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "RZ", ent.rotation_deg.z, -180.0f, 180.0f, cz_col, "%.0f");
                iy += sh_row + 7 * sc;

                xform_changed |= slider(2021, ix,                    iy, s3_w, sh_row, "SX", ent.scale.x, 0.15f, 8.0f, cx_col, "%.2f");
                xform_changed |= slider(2022, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "SY", ent.scale.y, 0.15f, 8.0f, cy_col, "%.2f");
                xform_changed |= slider(2023, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "SZ", ent.scale.z, 0.15f, 8.0f, cz_col, "%.2f");
                iy += sh_row + 12 * sc;
            } else {
                iy += 6 * sc;
            }
            if (xform_changed) {
                sync_group_transform(ent);
                world.invalidate_contact_ao();
            }

            if (ent.kind == EntityKind::PointLight) {
                // ---- LIVE 3D POINT LIGHT INSPECTOR
                ui.rect(ix, iy, iw, 1.2f * sc, col_border);
                iy += 12 * sc;
                ui.text(ix, iy, "DYNAMIC POINT LIGHT", 13.5f * sc, col_amber);
                iy += 24 * sc;
                slider(6001, ix, iy, iw, sh_row, "Intensity (Lumens)", ent.light_intensity, 0.0f, 45.0f, col_amber, "%.1f");
                iy += sh_row + 7 * sc;
                slider(6002, ix, iy, iw, sh_row, "Light Range (m)",    ent.light_range,     2.0f, 45.0f, col_accent, "%.1f");
                iy += sh_row + 7 * sc;
                slider(6003, ix,                    iy, s3_w, sh_row, "LR", ent.light_color.x, 0.0f, 1.0f, cx_col, "%.2f");
                slider(6004, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "LG", ent.light_color.y, 0.0f, 1.0f, cy_col, "%.2f");
                slider(6005, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "LB", ent.light_color.z, 0.0f, 1.0f, cz_col, "%.2f");
                iy += sh_row + 12 * sc;

                ui.text(ix, iy, "LIGHT COLOR PRESETS", 12.5f * sc, col_muted);
                iy += 20 * sc;
                const f32 p4_w = (iw - 18 * sc) / 4.0f;
                if (btn(ix,                    iy, p4_w, 28 * sc, "WARM",  false, col_amber, 12.0f)) ent.light_color = {1.0f, 0.70f, 0.34f};
                if (btn(ix + (p4_w + 6 * sc),  iy, p4_w, 28 * sc, "TORCH", false, {1.0f, 0.42f, 0.14f, 1}, 12.0f)) ent.light_color = {1.0f, 0.38f, 0.12f};
                if (btn(ix + (p4_w + 6 * sc)*2,iy, p4_w, 28 * sc, "CYAN",  false, col_accent, 12.0f)) ent.light_color = {0.22f, 0.84f, 1.0f};
                if (btn(ix + (p4_w + 6 * sc)*3,iy, p4_w, 28 * sc, "WHITE", false, {0.9f, 0.92f, 0.98f, 1}, 12.0f)) ent.light_color = {1.0f, 0.98f, 0.95f};
                iy += 38 * sc;
                if (btn(ix, iy, iw, 30 * sc, "Try Night Sky (See Light Glow!)", sky.preset_id == 3, {0.42f, 0.56f, 0.98f, 1}, 13.0f)) {
                    apply_sky_preset(sky.preset_id == 3 ? 0 : 3);
                }
            } else {
                // ---- TERRAIN SCULPT CONTROLS (when a Terrain entity is selected!)
                if (ent.kind == EntityKind::Terrain) {
                    ui.rect(ix, iy, iw, 1.2f * sc, col_border);
                    iy += 12 * sc;
                    ui.text(ix, iy, "PROCEDURAL TERRAIN SCULPT", 13.5f * sc, col_green);
                    iy += 22 * sc;
                    bool rebuild = false;
                    const f32 p4_w = (iw - 18 * sc) / 4.0f;
                    if (btn(ix,                    iy, p4_w, 28 * sc, "ALPINE", false, col_green, 12.0f)) { ent.terrain_amp = 22.0f; ent.terrain_freq = 0.035f; rebuild = true; }
                    if (btn(ix + (p4_w + 6 * sc),  iy, p4_w, 28 * sc, "HILLS",  false, col_accent, 12.0f)) { ent.terrain_amp = 10.0f; ent.terrain_freq = 0.024f; rebuild = true; }
                    if (btn(ix + (p4_w + 6 * sc)*2,iy, p4_w, 28 * sc, "CANYON", false, col_amber, 12.0f)) { ent.terrain_amp = 28.0f; ent.terrain_freq = 0.048f; rebuild = true; }
                    if (btn(ix + (p4_w + 6 * sc)*3,iy, p4_w, 28 * sc, "FLAT",   false, col_muted, 12.0f)) { ent.terrain_amp = 1.2f;  ent.terrain_freq = 0.020f; rebuild = true; }
                    iy += 36 * sc;
                    rebuild |= slider(7001, ix, iy, iw, sh_row, "Mountain Height", ent.terrain_amp,  0.5f, 36.0f, col_green, "%.1fm");
                    iy += sh_row + 7 * sc;
                    rebuild |= slider(7002, ix, iy, iw, sh_row, "Ridge Frequency", ent.terrain_freq, 0.012f, 0.070f, col_green, "%.3f");
                    iy += sh_row + 7 * sc;
                    rebuild |= slider(7003, ix, iy, iw, sh_row, "Terrain Seed",    ent.terrain_seed, 1.0f, 200.0f, col_accent, "%.0f");
                    iy += sh_row + 8 * sc;
                    if (btn(ix, iy, iw, 28 * sc, ent.terrain_alpine_shader ? "Shader: ALPINE BIOME (Click -> PBR)" : "Shader: PBR MATERIAL (Click -> Alpine)",
                            ent.terrain_alpine_shader, col_green, 12.5f)) {
                        ent.terrain_alpine_shader = !ent.terrain_alpine_shader;
                    }
                    iy += 36 * sc;
                    if (rebuild) {
                        ent.mesh = world.upload(make_procedural_terrain(ent.terrain_size, 108, ent.terrain_amp, ent.terrain_freq, ent.terrain_seed));
                        world.invalidate_contact_ao();
                    }
                }

                // ---- LIVE UV & TEXTURE EDITOR
                ui.rect(ix, iy, iw, 1.2f * sc, col_border);
                iy += 10 * sc;
                ui.text(ix, iy, "UV & TEXTURE EDITOR  (LIVE 120 FPS)", 13.0f * sc, col_accent);
                iy += 22 * sc;

                const f32 s2_w = (iw - 8 * sc) * 0.5f;
                slider(3001, ix,              iy, s2_w, sh_row, "Tile U", ent.uv_tile.x,   0.15f, 8.0f, col_accent, "%.2f");
                slider(3002, ix + s2_w + 8*sc,iy, s2_w, sh_row, "Tile V", ent.uv_tile.y,   0.15f, 8.0f, col_accent, "%.2f");
                iy += sh_row + 6 * sc;
                slider(3003, ix,              iy, s2_w, sh_row, "Off U",  ent.uv_offset.x, -4.0f, 4.0f, col_accent, "%.2f");
                slider(3004, ix + s2_w + 8*sc,iy, s2_w, sh_row, "Off V",  ent.uv_offset.y, -4.0f, 4.0f, col_accent, "%.2f");
                iy += sh_row + 6 * sc;
                slider(3005, ix, iy, iw, sh_row, "UV Rotation (deg)", ent.uv_rotation_deg, -180.0f, 180.0f, col_accent, "%.1f");
                iy += sh_row + 7 * sc;

                if (btn(ix, iy, iw, 28 * sc, ent.world_lock_uv ? "[x] World-Lock UV (No Stretch)" : "[ ] World-Lock UV (Auto-Tile)",
                        ent.world_lock_uv, col_green, 13.0f)) {
                    ent.world_lock_uv = !ent.world_lock_uv;
                }
                iy += 36 * sc;

                // ---- NOVA PRISM PBR MATERIAL & MICRO-GEO CONTROLS
                if (iy + 160 * sc < ry_top + lh) {
                    ui.rect(ix, iy, iw, 1.2f * sc, col_border);
                    iy += 10 * sc;
                    ui.text(ix, iy, "NOVA PRISM  |  PBR & STEEP POM", 13.0f * sc, col_green);
                    iy += 22 * sc;

                    slider(4001, ix, iy, iw, sh_row, "PRISM Steep POM Depth", ent.pom_depth, 0.0f, 0.075f, col_green, "%.3f");
                    iy += sh_row + 7 * sc;

                    if (const auto* mat_ptr = world.get_material(ent.material)) {
                        renderer::PbrMaterial mat = *mat_ptr;
                        bool m_changed = false;

                        if (ent.kind != EntityKind::Terrain) {
                            const f32 pv_h = 54 * sc;
                            const f32 pv_w = (iw - 8 * sc) * 0.5f;
                            ui.rect_rounded(ix, iy, pv_w, pv_h, 6 * sc, col_card, col_border, 1.2f * sc);
                            ui.textured_rect(ix + 4 * sc, iy + 4 * sc, pv_w - 8 * sc, pv_h - 8 * sc, mat.base_color);
                            ui.rect_rounded(ix + pv_w + 8 * sc, iy, pv_w, pv_h, 6 * sc, col_card, col_border, 1.2f * sc);
                            ui.textured_rect(ix + pv_w + 12 * sc, iy + 4 * sc, pv_w - 8 * sc, pv_h - 8 * sc, mat.normal);
                            iy += pv_h + 8 * sc;
                        }

                        m_changed |= slider(4002, ix, iy, iw, sh_row, "Normal Strength", mat.normal_scale,       0.0f, 2.5f, col_accent, "%.2f");
                        iy += sh_row + 6 * sc;
                        m_changed |= slider(4003, ix, iy, iw, sh_row, "Roughness Mult",  mat.roughness,          0.05f, 1.25f, col_accent, "%.2f");
                        iy += sh_row + 6 * sc;
                        if (iy + sh_row < ry_top + lh - 80 * sc) {
                            m_changed |= slider(4004, ix, iy, iw, sh_row, "Occlusion / AO",  mat.occlusion_strength, 0.0f, 1.0f, col_accent, "%.2f");
                            iy += sh_row + 6 * sc;
                        }
                        if (iy + sh_row < ry_top + lh - 72 * sc) {
                            m_changed |= slider(4005, ix,                    iy, s3_w, sh_row, "R", mat.base_color_factor.x, 0.2f, 1.4f, cx_col, "%.2f");
                            m_changed |= slider(4006, ix + (s3_w + s3_gap),  iy, s3_w, sh_row, "G", mat.base_color_factor.y, 0.2f, 1.4f, cy_col, "%.2f");
                            m_changed |= slider(4007, ix + (s3_w + s3_gap)*2,iy, s3_w, sh_row, "B", mat.base_color_factor.z, 0.2f, 1.4f, cz_col, "%.2f");
                            iy += sh_row + 10 * sc;
                        }
                        if (m_changed) {
                            world.update_material(ent.material, mat);
                        }
                    }

                    // Quick 6-PBR Material Swatches (1-click paint!)
                    if (iy + 76 * sc < ry_top + lh) {
                        ui.text(ix, iy, "QUICK PAINT PBR MATERIAL", 12.0f * sc, col_muted);
                        iy += 18 * sc;
                        const f32 sw_w = (iw - 12 * sc) / 3.0f;
                        const f32 sw_h = 26 * sc;
                        for (usize i = 0; i < mat_palette_.size(); ++i) {
                            const f32 sx = ix + static_cast<f32>(i % 3) * (sw_w + 6 * sc);
                            const f32 sy = iy + static_cast<f32>(i / 3) * (sw_h + 5 * sc);
                            if (btn(sx, sy, sw_w, sw_h, mat_palette_[i].first, ent.material.id == mat_palette_[i].second.id, col_amber, 12.0f)) {
                                push_undo();
                                ent.material = mat_palette_[i].second;
                                if (ent.kind == EntityKind::Terrain) ent.terrain_alpine_shader = false;
                                toast("Painted " + ent.name + " -> " + mat_palette_[i].first);
                            }
                        }
                    }
                }
            }
        } else {
            iy += 32 * sc;
            ui.text(ix, iy, "Click +PLANE, +TERRAIN, +LIGHT or SKY", 14.0f * sc, col_muted);
        }

        // ---- 4. FLOATING DRAG CARD PREVIEW UNDER CURSOR
        if (dragging_prefab_ >= 0 && dragging_prefab_ < static_cast<int>(prefabs_.size())) {
            const auto& pf = prefabs_[static_cast<usize>(dragging_prefab_)];
            const f32 gw = 110 * sc, gh = 86 * sc;
            ui.rect_rounded(mx - gw * 0.5f, my - gh * 0.5f, gw, gh, 8 * sc, {0.08f, 0.12f, 0.18f, 0.90f}, col_accent, 2.0f * sc);
            if (pf.thumb.valid()) {
                ui.textured_rect(mx - gw * 0.5f + 6 * sc, my - gh * 0.5f + 6 * sc, gw - 12 * sc, gh - 30 * sc, pf.thumb);
            }
            ui.text(mx - gw * 0.5f + 8 * sc, my + gh * 0.5f - 20 * sc, pf.label, 12.5f * sc, col_text);
        }

        // ---- 5. BOTTOM HINT BAR / STATUS TOAST
        const std::string bar_msg = toast_timer > 0.0f
            ? status_toast
            : "RMB+WASD Fly  |  Click Select  |  W Move / E Rotate / R Scale  |  Drag Assets into 3D  |  F5 PLAY  |  F1 Hide UI";
        const f32 bw = ui.measure_text(bar_msg, 14.0f * sc) + 28 * sc;
        const f32 bx = (W - bw) * 0.5f, by_bot = H - 44 * sc;
        ui.rect_rounded(bx, by_bot, bw, 32 * sc, 9 * sc, {0.055f, 0.075f, 0.105f, 0.86f},
                        toast_timer > 0.0f ? col_green : col_border, 1.3f * sc);
        ui.text(bx + 14 * sc, by_bot + 7 * sc, bar_msg, 14.0f * sc, toast_timer > 0.0f ? Vec4{0.45f, 0.96f, 0.68f, 1} : col_muted);
        return win_action;
    }

    [[nodiscard]] bool pointer_over_ui() const noexcept { return pointer_over_ui_; }

private:
    village::VillageMaterials mats_{};
    std::array<renderer::MeshId, 6> village_meshes_{};
    renderer::MeshId m_wall_blk_{};
    renderer::MeshId m_crate_{};
    renderer::MeshId m_pillar_{};
    renderer::MeshId m_boulder_{};
    renderer::MeshId m_slab_{};
    renderer::MeshId m_moss_{};
    renderer::MeshId m_plaster_{};
    renderer::MeshId m_puddle_{};
    renderer::MeshId m_plane_{};
    renderer::MeshId m_light_{};
    renderer::MeshId clay_opaque_{};
    renderer::MeshId clay_two_sided_{};
    renderer::MeshId clay_water_{};
    std::vector<village::Collider> clay_colliders_;
    std::vector<std::pair<std::string, renderer::MaterialId>> mat_palette_;
    std::vector<AssetPrefab> prefabs_;
    std::vector<std::vector<CraftEntity>> undo_stack_;
    std::vector<CraftEntity> play_snapshot_;
    bool pointer_over_ui_ = false;
    int dragging_prefab_ = -1;
    u32 active_slider_id_ = 0;
    int active_gizmo_axis_ = -1;
    void sync_group_transform(const CraftEntity& src) noexcept {
        if (src.group_id == 0) return;
        for (auto& e : entities) {
            if (e.group_id == src.group_id) {
                e.position = src.position;
                e.rotation_deg = src.rotation_deg;
                e.scale = src.scale;
            }
        }
    }

    u32 next_group_id_ = 1;
    Vec2 gizmo_start_mouse_{0, 0};
    Vec3 gizmo_start_pos_{0, 0, 0};
    Vec3 gizmo_start_rot_{0, 0, 0};
    Vec3 gizmo_start_scale_{1, 1, 1};
};

} // namespace nova::craft
