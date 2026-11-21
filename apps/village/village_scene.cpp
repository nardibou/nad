// Procedural mountain village: terrain, houses, dry-stone walls, olive trees, props and colliders.

#include "village_scene.hpp"

#include "nova/core/log.hpp"
#include "nova/renderer/png_image.hpp"
#include "nova/renderer/world_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace nova::village {

using math::Mat4;
using math::Vec3;
using math::Vec4;
using renderer::Material;
using renderer::MeshData;

namespace {

// ------------------------------------------------------------------ deterministic randomness / noise

struct Rng {
    u32 state;
    explicit Rng(u32 seed) : state(seed * 747796405u + 2891336453u) {}
    u32 next() {
        state = state * 747796405u + 2891336453u;
        u32 w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (w >> 22u) ^ w;
    }
    f32 uniform() { return static_cast<f32>(next() >> 8) / 16777216.0f; }
    f32 range(f32 lo, f32 hi) { return lo + (hi - lo) * uniform(); }
};

f32 hash2(i32 x, i32 y) {
    u32 h = static_cast<u32>(x) * 374761393u + static_cast<u32>(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return static_cast<f32>((h ^ (h >> 16)) & 0xFFFFFF) / 16777215.0f;
}

f32 value_noise(f32 x, f32 y) {
    const f32 fx = std::floor(x), fy = std::floor(y);
    const i32 ix = static_cast<i32>(fx), iy = static_cast<i32>(fy);
    const f32 tx = x - fx, ty = y - fy;
    const f32 ux = tx * tx * (3 - 2 * tx), uy = ty * ty * (3 - 2 * ty);
    const f32 a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
    return math::lerp(math::lerp(a, b, ux), math::lerp(c, d, ux), uy);
}

f32 fbm(f32 x, f32 y, int octaves) {
    f32 sum = 0, amp = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * value_noise(x, y);
        x = x * 2.03f + 13.7f;
        y = y * 2.03f + 7.3f;
        amp *= 0.5f;
    }
    return sum;
}

f32 ridged(f32 x, f32 y, int octaves) {
    f32 sum = 0, amp = 0.5f, weight = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        f32 n = 1.0f - std::abs(value_noise(x, y) * 2.0f - 1.0f);
        n *= n * weight;
        weight = std::clamp(n * 1.6f, 0.0f, 1.0f);
        sum += n * amp;
        x = x * 2.1f + 5.1f;
        y = y * 2.1f + 1.7f;
        amp *= 0.5f;
    }
    return sum;
}

f32 smoothstep(f32 e0, f32 e1, f32 x) {
    const f32 t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}

// ------------------------------------------------------------------ terrain

constexpr Vec3 kTowerHill{-30.0f, 0.0f, -150.0f};

f32 terrain_height_at(f32 x, f32 z) {
    const f32 r = std::sqrt(x * x + z * z);
    const f32 hills = (fbm(x * 0.012f, z * 0.012f, 4) - 0.35f) * 18.0f * smoothstep(32.0f, 95.0f, r);

    // Sculpted Djurdjura / Atlas alpine massif with crisp ridged peaks leaving 50% open blue sky above!
    const f32 crag_base = ridged(x * 0.0042f + 3.1f, z * 0.0042f - 1.4f, 6);
    const f32 crag_fine = ridged(x * 0.0135f - 2.2f, z * 0.0135f + 5.7f, 4);
    f32 mountains = (crag_base * 68.0f + crag_fine * 26.0f) * smoothstep(145.0f, 420.0f, r);

    // Hero alpine peaks framing the exact composition (left snow peak, center massif, right sun-pass ridge):
    auto peak = [&](f32 px, f32 pz, f32 h, f32 sx, f32 sz) {
        const f32 ddx = (x - px) / sx;
        const f32 ddz = (z - pz) / sz;
        const f32 env = std::exp(-(ddx * ddx + ddz * ddz));
        return h * env * (0.62f + 0.45f * crag_base + 0.25f * crag_fine);
    };
    mountains += peak(-225.0f, -260.0f, 155.0f, 105.0f, 105.0f); // Left towering snow-dusted peak behind olive tree
    mountains += peak(  16.0f, -380.0f, 125.0f,  95.0f,  95.0f); // Center Djurdjura alpine massif down the street
    mountains += peak( 335.0f, -280.0f, 165.0f, 105.0f, 105.0f); // Far-right jagged snow peak framing the sun
    mountains += peak( 165.0f, -380.0f,  85.0f,  85.0f,  85.0f); // Mid-right ridge flanking the sun pass

    // Carve a crisp golden-hour alpine V-pass right along the sun vector (+0.58, -0.78) so the sun beams through!
    if (x > 50.0f && z < -60.0f) {
        const f32 along = (x * 0.58f - z * 0.78f);
        const f32 across = std::abs(x * 0.78f + z * 0.58f + 12.0f);
        const f32 pass = std::exp(-(across * across) / (2.0f * 52.0f * 52.0f)) * smoothstep(90.0f, 220.0f, along);
        mountains *= (1.0f - 0.62f * pass);
    }

    const f32 dx = x - kTowerHill.x, dz = z - kTowerHill.z;
    const f32 hill = 22.0f * std::exp(-(dx * dx + dz * dz) / (2.0f * 40.0f * 40.0f));
    const f32 ground = (fbm(x * 0.3f, z * 0.3f, 2) - 0.5f) * 0.05f; // keeps the streets almost flat
    return std::max(hills, 0.0f) + mountains + hill + ground;
}

MeshData build_terrain() {
    constexpr int kCells = 340;
    constexpr f32 kSize = 1350.0f;
    constexpr f32 kStep = kSize / kCells;
    const Vec3 origin{-kSize * 0.5f, 0.0f, -kSize * 0.72f}; // extends further towards the mountains
    MeshData mesh;
    mesh.vertices.reserve((kCells + 1) * (kCells + 1));
    for (int j = 0; j <= kCells; ++j) {
        for (int i = 0; i <= kCells; ++i) {
            const f32 x = origin.x + static_cast<f32>(i) * kStep, z = origin.z + static_cast<f32>(j) * kStep;
            const f32 e = kStep * 0.5f;
            const Vec3 n = math::normalize(Vec3{terrain_height(x - e, z) - terrain_height(x + e, z), 2.0f * e,
                                                terrain_height(x, z - e) - terrain_height(x, z + e)});
            mesh.vertices.push_back(
                {{x, terrain_height(x, z), z}, n, renderer::material_color(Material::Terrain, {1, 1, 1})});
        }
    }
    for (int j = 0; j < kCells; ++j) {
        for (int i = 0; i < kCells; ++i) {
            const u32 a = j * (kCells + 1) + i, b = a + 1, c = a + (kCells + 1), d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, c, d, a, d, b}); // CCW seen from above
        }
    }
    return mesh;
}

// ------------------------------------------------------------------ village builder

class Builder {
public:
    MeshData opaque;       // buildings, plaster, stone plinths, metal
    MeshData foliage;      // tree crowns
    MeshData two_sided;    // grass, flag cloth, draped rug
    MeshData water;        // puddles (no shadows)
    MeshData stone_walls;  // 3D displaced PBR stone walls
    MeshData dirt_road;    // 3D displaced PBR dirt & gravel street
    MeshData mud_patches;  // 3D displaced PBR wet mud & bootprints around puddles
    MeshData forest_floor; // 3D displaced PBR twigs, leaves & moss under trees
    MeshData wood_props;   // PBR weathered wooden planks (doors, lintels, beams, ladder)
    MeshData tree_bark;    // PBR gnarled mossy tree bark (trunks & branches)
    std::vector<Collider> colliders;

    Builder() : cube_(renderer::make_cube(1.0f)), sphere_(renderer::make_icosphere(1)),
                cylinder_(renderer::make_cylinder(14, 1, 1, 1)) {}

    static Mat4 place(const Vec3& pos, const Vec3& size, f32 yaw = 0, f32 tilt = 0) {
        return math::translate(pos) * math::rotate({0, 1, 0}, yaw) * math::rotate({1, 0, 0}, tilt) * math::scale(size);
    }

    // Append a box to a PBR mesh with world-scaled planar UVs & tangents for crisp per-pixel normal/roughness mapping.
    void append_pbr_box(MeshData& dst, const Mat4& xform, const Vec3& size, f32 uv_scale = 0.85f) {
        const usize start_v = dst.vertices.size();
        renderer::append(dst, cube_, xform, {1.0f, 1.0f, 1.0f, 0.0f});
        for (usize i = start_v; i < dst.vertices.size(); ++i) {
            auto& v = dst.vertices[i];
            const Vec3 an{std::abs(v.normal.x), std::abs(v.normal.y), std::abs(v.normal.z)};
            if (an.x >= an.y && an.x >= an.z) {
                v.uv = {v.position.z * uv_scale, v.position.y * uv_scale};
                v.tangent = {0.0f, 0.0f, v.normal.x > 0 ? -1.0f : 1.0f, 1.0f};
            } else if (an.y >= an.x && an.y >= an.z) {
                v.uv = {v.position.x * uv_scale, v.position.z * uv_scale};
                v.tangent = {1.0f, 0.0f, 0.0f, 1.0f};
            } else {
                v.uv = {v.position.x * uv_scale, v.position.y * uv_scale};
                v.tangent = {v.normal.z > 0 ? 1.0f : -1.0f, 0.0f, 0.0f, 1.0f};
            }
            (void)size;
        }
    }

    void box(const Vec3& center, const Vec3& size, Material m, const Vec3& albedo, f32 yaw = 0) {
        if (m == Material::Wood && albedo.x > 0.15f) {
            append_pbr_box(wood_props, place(center, size, yaw), size);
            return;
        }
        renderer::append(opaque, cube_, place(center, size, yaw), renderer::material_color(m, albedo));
    }

    // Box in a house's local frame (x across the facade, +z out of the front face).
    void local_box(const Mat4& frame, const Vec3& center, const Vec3& size, Material m, const Vec3& albedo) {
        if (m == Material::Wood && albedo.x > 0.15f) {
            append_pbr_box(wood_props, frame * math::translate(center) * math::scale(size), size);
            return;
        }
        renderer::append(opaque, cube_, frame * math::translate(center) * math::scale(size),
                         renderer::material_color(m, albedo));
    }

    // Tilt the shading normals towards the corners so light rolls across each uneven stone.
    void stone(MeshData& dst, const Vec3& center, const Vec3& radii, f32 yaw, const Vec3& albedo, Material m) {
        std::array<Vec3, 8> corner_offset;
        for (Vec3& o : corner_offset) {
            o = {rng_.range(-0.5f, 0.5f), rng_.range(-0.45f, 0.45f), rng_.range(-0.5f, 0.5f)};
        }
        MeshData s = cube_;
        for (auto& v : s.vertices) {
            const int idx = (v.position.x > 0 ? 1 : 0) | (v.position.y > 0 ? 2 : 0) | (v.position.z > 0 ? 4 : 0);
            // Pull corners inwards so stones read as rounded-off slabs rather than boxes.
            v.position = v.position * 1.4f + corner_offset[static_cast<usize>(idx)] * 0.8f;
        }
        for (usize f = 0; f < s.indices.size(); f += 6) {
            auto& v0 = s.vertices[s.indices[f]];
            const Vec3 n = math::normalize(math::cross(s.vertices[s.indices[f + 1]].position - v0.position,
                                                       s.vertices[s.indices[f + 2]].position - v0.position));
            for (usize k = 0; k < 6; ++k) {
                auto& vertex = s.vertices[s.indices[f + k]];
                vertex.normal = math::normalize(n + math::normalize(vertex.position) * 0.55f);
            }
        }
        renderer::append(dst, s, place(center, radii, yaw), renderer::material_color(m, albedo));
    }

    void dry_stone_wall(Vec3 a, Vec3 b, f32 height, f32 thickness) {
        const Vec3 d = b - a;
        const f32 len = math::length(d);
        const f32 yaw = std::atan2(d.x, d.z);
        const Vec3 dir = d * (1.0f / len);
        colliders.push_back({(a + b) * 0.5f + Vec3{0, height * 0.5f, 0}, {thickness * 0.5f, height * 0.5f + 0.1f, len * 0.5f}, yaw});
        // Core fills the gaps between stones.
        box((a + b) * 0.5f + Vec3{0, height * 0.45f, 0}, {thickness * 0.8f, height * 0.9f, len}, Material::Stone,
            {0.30f, 0.27f, 0.23f}, yaw);
        for (f32 y = 0.1f; y < height;) {
            const f32 row_h = rng_.range(0.17f, 0.3f);
            for (f32 t = rng_.range(0.0f, 0.3f); t < len;) {
                const f32 sl = rng_.range(0.25f, 0.55f);
                const f32 tone = rng_.range(0.7f, 1.2f);
                const Vec3 c = a + dir * (t + sl * 0.5f) + Vec3{0, y + row_h * 0.5f, 0};
                for (int side = -1; side <= 1; side += 2) {
                    const Vec3 sideways{dir.z * static_cast<f32>(side), 0, -dir.x * static_cast<f32>(side)};
                    stone(opaque, c + sideways * (thickness * 0.33f),
                          {thickness * rng_.range(0.14f, 0.22f), row_h * 0.58f, sl * 0.56f},
                          yaw + rng_.range(-0.3f, 0.3f), Vec3{0.42f, 0.36f, 0.29f} * tone, Material::Stone);
                }
                t += sl;
            }
            y += row_h;
        }
        // Capstones.
        for (f32 t = 0.2f; t < len; t += rng_.range(0.45f, 0.65f)) {
            stone(opaque, a + dir * t + Vec3{0, height + 0.08f, 0}, {thickness * 0.5f, 0.16f, 0.3f},
                  yaw + rng_.range(-0.4f, 0.4f), Vec3{0.6f, 0.55f, 0.47f} * rng_.range(0.8f, 1.1f), Material::Stone);
        }
        grass_along(a, b, 1.0f);
    }

    struct HouseSpec {
        Vec3 position; // centre of the footprint at ground level
        f32 width;     // along the facade
        f32 depth;
        f32 height;
        f32 yaw;       // 0 => facade faces +Z
        Vec3 plaster{0.78f, 0.68f, 0.55f};
        f32 door_offset = 0.0f;
        bool lantern = false;
    };

    Mat4 house(const HouseSpec& h) {
        const Mat4 frame = math::translate(h.position) * math::rotate({0, 1, 0}, h.yaw);
        colliders.push_back({h.position + Vec3{0, h.height * 0.5f, 0}, {h.width * 0.5f, h.height * 0.5f, h.depth * 0.5f}, h.yaw});
        const Vec3 wood{0.36f, 0.25f, 0.16f};
        local_box(frame, {0, h.height * 0.5f, 0}, {h.width, h.height, h.depth}, Material::Plaster, h.plaster);
        // Stone plinth + rustic staggered corner stone quoins (authentic Kabyle / Atlas masonry!)
        local_box(frame, {0, 0.28f, 0}, {h.width + 0.14f, 0.56f, h.depth + 0.14f}, Material::Stone,
                  {0.44f, 0.39f, 0.32f});
        for (f32 sx : {-1.0f, 1.0f}) {
            for (f32 sz : {-1.0f, 1.0f}) {
                for (f32 qy = 0.72f; qy < h.height - 0.35f; qy += 0.44f) {
                    const bool alt = (static_cast<int>(qy * 3.0f) & 1) != 0;
                    const f32 qw = alt ? 0.46f : 0.30f;
                    const f32 qd = alt ? 0.30f : 0.46f;
                    const f32 tone = rng_.range(0.84f, 1.12f);
                    local_box(frame, {sx * (h.width * 0.5f - qw * 0.38f), qy, sz * (h.depth * 0.5f - qd * 0.38f)},
                              {qw, 0.34f, qd}, Material::Stone, Vec3{0.48f, 0.41f, 0.33f} * tone);
                }
            }
        }
        // Flat earthen roof with weathered timber trim lip and protruding beam ends.
        local_box(frame, {0, h.height + 0.12f, 0}, {h.width + 0.52f, 0.24f, h.depth + 0.52f}, Material::Dirt,
                  {0.42f, 0.34f, 0.25f});
        local_box(frame, {0, h.height - 0.02f, 0}, {h.width + 0.36f, 0.09f, h.depth + 0.36f}, Material::Wood,
                  {0.26f, 0.18f, 0.12f});
        for (f32 x = -h.width * 0.5f + 0.35f; x < h.width * 0.5f; x += 0.65f) {
            local_box(frame, {x, h.height - 0.12f, h.depth * 0.5f + 0.2f}, {0.14f, 0.14f, 0.5f}, Material::Wood, wood);
        }
        // Door with lintel.
        const f32 door_h = std::min(2.3f, h.height - 0.6f);
        local_box(frame, {h.door_offset, door_h * 0.5f, h.depth * 0.5f + 0.02f}, {1.15f, door_h, 0.1f},
                  Material::Wood, {0.32f, 0.22f, 0.14f});
        local_box(frame, {h.door_offset, door_h + 0.1f, h.depth * 0.5f + 0.06f}, {1.6f, 0.18f, 0.2f}, Material::Wood,
                  wood);
        // Small dark windows.
        for (f32 x : {-h.width * 0.3f, h.width * 0.3f}) {
            if (std::abs(x - h.door_offset) > 1.4f) {
                local_box(frame, {x, std::min(2.6f, h.height - 1.0f), h.depth * 0.5f + 0.01f}, {0.6f, 0.7f, 0.06f},
                          Material::Wood, {0.08f, 0.06f, 0.05f});
            }
        }
        if (h.lantern) {
            const Vec3 l{h.door_offset + 1.3f, 2.5f, h.depth * 0.5f};
            local_box(frame, l + Vec3{0, 0.25f, 0.25f}, {0.05f, 0.05f, 0.5f}, Material::Metal, {0.15f, 0.13f, 0.1f});
            local_box(frame, l + Vec3{0, 0.0f, 0.45f}, {0.22f, 0.3f, 0.22f}, Material::Emissive, {1.0f, 0.62f, 0.3f});
            local_box(frame, l + Vec3{0, 0.18f, 0.45f}, {0.26f, 0.06f, 0.26f}, Material::Metal, {0.12f, 0.1f, 0.08f});
            local_box(frame, l + Vec3{0, -0.18f, 0.45f}, {0.26f, 0.06f, 0.26f}, Material::Metal, {0.12f, 0.1f, 0.08f});
        }
        grass_along(math::transform_point(frame, {-h.width * 0.5f, 0, h.depth * 0.5f + 0.15f}),
                    math::transform_point(frame, {h.width * 0.5f, 0, h.depth * 0.5f + 0.15f}), 0.6f);
        return frame;
    }

    void olive_tree(const Vec3& base, const Vec3& crown_center, f32 crown_radius) {
        const Vec3 bark{0.28f, 0.22f, 0.17f};
        // Gnarled trunk: bent segments from the base to the crown.
        Vec3 p = base;
        const int segments = 7;
        for (int i = 0; i < segments; ++i) {
            const f32 t = static_cast<f32>(i + 1) / segments;
            Vec3 q = math::lerp(base, crown_center - Vec3{0, crown_radius * 0.38f, 0}, t);
            q += Vec3{rng_.range(-0.22f, 0.22f), 0, rng_.range(-0.22f, 0.22f)} * (1.0f - t);
            segment(p, q, 0.27f * (1.0f - t * 0.56f), bark);
            p = q;
        }

        // Radiating boughs tucked naturally inside the leafy canopy (no protruding bare sticks!)
        const bool hero = crown_radius >= 2.0f && math::length(base) < 35.0f;
        const int bough_count = hero ? 8 : 4;
        std::vector<Vec3> clump_centers;
        clump_centers.reserve(static_cast<usize>(bough_count * 4));
        for (int i = 0; i < bough_count; ++i) {
            const f32 az = (static_cast<f32>(i) / static_cast<f32>(bough_count)) * 2.0f * math::kPi + rng_.range(-0.32f, 0.32f);
            const f32 el = rng_.range(0.05f, 0.58f);
            const f32 reach = crown_radius * rng_.range(0.45f, 0.72f);
            const Vec3 mid = crown_center + Vec3{std::cos(az) * reach * 0.55f, el * crown_radius * 0.25f - 0.18f, std::sin(az) * reach * 0.55f};
            const Vec3 tip = crown_center + Vec3{std::cos(az) * reach, el * crown_radius * 0.45f, std::sin(az) * reach};
            segment(p, mid, 0.088f, bark);
            segment(mid, tip, 0.040f, bark);
            clump_centers.push_back(mid + Vec3{0, 0.25f, 0});
            clump_centers.push_back(tip + Vec3{0, 0.20f, 0});
            if (hero) {
                clump_centers.push_back(tip + Vec3{std::cos(az) * 0.55f, 0.35f, std::sin(az) * 0.55f});
            }
        }
        clump_centers.push_back(crown_center + Vec3{0, crown_radius * 0.35f, 0});

        // Lush, dense 3D Mediterranean olive foliage: thousands of overlapping lanceolate leaves clustered in billowing domes
        // Added to two_sided (Material::Foliage) so leaves render richly from both sides and cast real dappled leaf shadows.
        auto add_leaf_blade = [&](const Vec3& root, const Vec3& dir, const Vec3& side_hint, f32 len, f32 wid, const Vec3& col) {
            const Vec3 d = math::normalize(dir);
            Vec3 s = math::cross(d, side_hint);
            const f32 sl = math::length(s);
            s = (sl > 1e-4f) ? (s * (1.0f / sl)) : Vec3{1.0f, 0.0f, 0.0f};
            const Vec3 n = math::normalize(math::cross(s, d));
            const Vec3 mid_l = root + d * (len * 0.48f) - s * (wid * 0.5f) + n * (len * 0.07f);
            const Vec3 mid_r = root + d * (len * 0.48f) + s * (wid * 0.5f) + n * (len * 0.07f);
            const Vec3 tip   = root + d * len - n * (len * 0.06f);
            const u32 base_v = static_cast<u32>(two_sided.vertices.size());
            const Vec4 vc = renderer::material_color(Material::Foliage, col);
            two_sided.vertices.push_back({root,  n, vc});
            two_sided.vertices.push_back({mid_l, n, vc});
            two_sided.vertices.push_back({tip,   n, vc});
            two_sided.vertices.push_back({mid_r, n, vc});
            two_sided.indices.insert(two_sided.indices.end(), {base_v, base_v + 1, base_v + 2, base_v, base_v + 2, base_v + 3});
        };

        const int sprig_count = hero ? 1150 : 130;
        for (int i = 0; i < sprig_count; ++i) {
            const Vec3 anchor = clump_centers[static_cast<usize>(i) % clump_centers.size()];
            Vec3 off{rng_.range(-1.0f, 1.0f), rng_.range(-0.68f, 0.78f), rng_.range(-1.0f, 1.0f)};
            const f32 ol = math::length(off);
            if (ol > 1.0f) off = off * (1.0f / ol);
            const f32 clump_r = crown_radius * (hero ? rng_.range(0.18f, 0.54f) : rng_.range(0.25f, 0.65f));
            const Vec3 c = anchor + off * clump_r;

            // Deep lush Mediterranean olive-green core -> vibrant warm golden-sunlit crown exterior
            const f32 out_t = std::clamp((c.y - (crown_center.y - crown_radius * 0.45f)) / (crown_radius * 1.1f) * 0.65f
                                         + (c.x - (crown_center.x - crown_radius * 0.5f)) / (crown_radius * 1.4f) * 0.35f, 0.0f, 1.0f);
            const Vec3 dark_leaf{0.07f, 0.13f, 0.03f};
            const Vec3 sunlit_leaf{0.26f, 0.36f, 0.08f};
            const Vec3 leaf_col = math::lerp(dark_leaf, sunlit_leaf, out_t) * rng_.range(0.75f, 1.25f);

            const int blades = hero ? 3 : 2;
            const f32 scale = hero ? rng_.range(0.26f, 0.44f) : rng_.range(0.42f, 0.68f);
            for (int k = 0; k < blades; ++k) {
                const Vec3 ldir{rng_.range(-1.0f, 1.0f), rng_.range(-0.55f, 0.80f), rng_.range(-1.0f, 1.0f)};
                const Vec3 sh{rng_.range(-0.5f, 0.5f), 1.0f, rng_.range(-0.5f, 0.5f)};
                add_leaf_blade(c, ldir, sh, scale, scale * 0.46f, leaf_col * rng_.range(0.88f, 1.12f));
            }
        }
    }

    void segment(const Vec3& a, const Vec3& b, f32 radius, const Vec3& /*albedo*/) {
        const Vec3 d = b - a;
        const f32 len = math::length(d);
        const Vec3 dir = d * (1.0f / len);
        const Vec3 axis = math::cross({0, 1, 0}, dir);
        const f32 angle = std::acos(std::clamp(dir.y, -1.0f, 1.0f));
        const Mat4 rot = math::length(axis) > 1e-4f ? math::rotate(axis, angle) : Mat4::identity();
        const usize start_v = tree_bark.vertices.size();
        renderer::append(tree_bark, cylinder_, math::translate(a) * rot * math::scale({radius, len * 1.05f, radius}),
                         {1.0f, 1.0f, 1.0f, 0.0f});
        // Assign cylindrical bark UVs and tangents for crisp 1K bark_albedo/bark_normal/bark_roughness sampling
        for (usize i = start_v; i < tree_bark.vertices.size(); ++i) {
            auto& v = tree_bark.vertices[i];
            const Vec3 rel = v.position - a;
            const f32 h = math::dot(rel, dir);
            const f32 theta = std::atan2(v.normal.z, v.normal.x);
            v.uv = {(theta / (2.0f * math::kPi) + 0.5f) * 2.0f, h * 0.75f};
            const Vec3 t = math::cross(dir, v.normal);
            const f32 tl = math::length(t);
            const Vec3 tn = tl > 1e-4f ? t * (1.0f / tl) : Vec3{1.0f, 0.0f, 0.0f};
            v.tangent = {tn.x, tn.y, tn.z, 1.0f};
        }
    }

    void grass_along(const Vec3& a, const Vec3& b, f32 density) {
        const f32 len = math::length(b - a);
        const int count = static_cast<int>(len * 2.2f * density);
        for (int i = 0; i < count; ++i) {
            if (rng_.uniform() < 0.35f) {
                continue;
            }
            const Vec3 p = math::lerp(a, b, rng_.uniform()) + Vec3{rng_.range(-0.3f, 0.3f), 0, rng_.range(-0.3f, 0.3f)};
            grass_tuft(p, rng_.range(0.25f, 0.6f));
        }
    }

    void grass_tuft(const Vec3& p, f32 height) {
        const Vec3 base_col = Vec3{0.18f, 0.30f, 0.07f} * rng_.range(0.75f, 1.15f);
        const Vec3 tip_col  = Vec3{0.38f, 0.46f, 0.13f} * rng_.range(0.80f, 1.20f);
        for (int i = 0; i < 10; ++i) {
            const f32 a = rng_.range(0, 6.283f);
            const f32 bh = height * rng_.range(0.55f, 1.05f);
            const Vec3 lean{std::cos(a) * rng_.range(0.06f, 0.24f), 0, std::sin(a) * rng_.range(0.06f, 0.24f)};
            const Vec3 side{-std::sin(a) * 0.017f, 0, std::cos(a) * 0.017f};
            const Vec3 mid = p + lean * 0.35f + Vec3{0, bh * 0.58f, 0};
            const Vec3 tip = p + lean + Vec3{0, bh, 0};
            const Vec3 n = math::normalize(math::cross(side, tip - p));
            const u32 base_v = static_cast<u32>(two_sided.vertices.size());
            const Vec4 c_root = renderer::material_color(Material::Grass, base_col);
            const Vec4 c_mid  = renderer::material_color(Material::Grass, math::lerp(base_col, tip_col, 0.55f));
            const Vec4 c_tip  = renderer::material_color(Material::Grass, tip_col);
            two_sided.vertices.push_back({p - side, n, c_root});
            two_sided.vertices.push_back({p + side, n, c_root});
            two_sided.vertices.push_back({mid - side * 0.55f, n, c_mid});
            two_sided.vertices.push_back({mid + side * 0.55f, n, c_mid});
            two_sided.vertices.push_back({tip, n, c_tip});
            two_sided.indices.insert(two_sided.indices.end(), {
                base_v, base_v + 1, base_v + 3,
                base_v, base_v + 3, base_v + 2,
                base_v + 2, base_v + 3, base_v + 4
            });
        }
    }

    void flag(const Vec3& pole_top, f32 width, f32 height) {
        constexpr int kSeg = 12;
        const u32 base = static_cast<u32>(two_sided.vertices.size());
        for (int j = 0; j <= 1; ++j) {
            for (int i = 0; i <= kSeg; ++i) {
                const f32 u = static_cast<f32>(i) / kSeg;
                const f32 wave = std::sin(u * 5.0f) * 0.12f * u; // resting ripple; the shader adds wind
                // rg = flag uv (u also scales wind sway in the vertex shader).
                two_sided.vertices.push_back({pole_top + Vec3{u * width, -static_cast<f32>(j) * height, wave},
                                              {0, 0, 1},
                                              {u, static_cast<f32>(j), 0.0f, static_cast<f32>(Material::Flag)}});
            }
        }
        for (int i = 0; i < kSeg; ++i) {
            const u32 a = base + i, b = a + 1, c = a + kSeg + 1, d = c + 1;
            two_sided.indices.insert(two_sided.indices.end(), {a, c, d, a, d, b});
        }
    }

    void puddle(const Vec3& center, f32 rx, f32 rz = 0.0f) {
        if (rz <= 0.0f) rz = rx * 1.45f;
        // Dead-flat organic mirror rainwater surface (normal strictly {0,1,0}) so SSR reflects houses, trees & sky crisply.
        constexpr int kSegs = 40;
        const f32 water_y = terrain_height_at(center.x, center.z) + 0.022f;
        const u32 base = static_cast<u32>(water.vertices.size());
        const Vec4 col = renderer::material_color(Material::Water, {0.03f, 0.025f, 0.02f});
        water.vertices.push_back({{center.x, water_y, center.z}, {0.0f, 1.0f, 0.0f}, col});
        for (int i = 0; i <= kSegs; ++i) {
            const f32 ang = (static_cast<f32>(i) / static_cast<f32>(kSegs)) * 2.0f * math::kPi;
            const f32 ca = std::cos(ang), sa = std::sin(ang);
            const f32 n = fbm(center.x * 0.7f + ca * 1.6f + 3.1f, center.z * 0.7f + sa * 1.6f - 1.9f, 3);
            const f32 rad = 0.68f + 0.52f * n;
            water.vertices.push_back({{center.x + ca * rx * rad, water_y, center.z + sa * rz * rad}, {0.0f, 1.0f, 0.0f}, col});
        }
        for (int i = 1; i <= kSegs; ++i) {
            water.indices.insert(water.indices.end(), {base, base + static_cast<u32>(i + 1), base + static_cast<u32>(i)});
        }
    }

    void pbr_displaced_stone_walls() {
        const image::PbrPngSet pbr = image::ensure_pbr_png_assets();

        // Pre-filter the heightmap with a 2D wrap-around separable Gaussian blur into a 256x256 macro-relief grid.
        // This preserves smooth, rounded 3D boulder silhouettes without jagged triangle cliffs or stretched UVs,
        // while wall_normal.png and wall_displacement.png deliver razor-sharp per-pixel crevices & rock pores!
        constexpr int kHSize = 256;
        std::vector<f32> h_src(static_cast<usize>(kHSize * kHSize), 0.5f);
        for (int y = 0; y < kHSize; ++y) {
            const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(kHSize);
            for (int x = 0; x < kHSize; ++x) {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(kHSize);
                h_src[static_cast<usize>(y * kHSize + x)] = pbr.wall_displacement.sample_height(u, v);
            }
        }
        constexpr int kRadius = 6;
        constexpr f32 kSigma = 2.6f;
        std::array<f32, 2 * kRadius + 1> kernel{};
        f32 ksum = 0.0f;
        for (int k = -kRadius; k <= kRadius; ++k) {
            const f32 w = std::exp(-0.5f * static_cast<f32>(k * k) / (kSigma * kSigma));
            kernel[static_cast<usize>(k + kRadius)] = w;
            ksum += w;
        }
        for (f32& w : kernel) w /= ksum;

        std::vector<f32> h_tmp(static_cast<usize>(kHSize * kHSize), 0.5f);
        std::vector<f32> h_smooth(static_cast<usize>(kHSize * kHSize), 0.5f);
        auto wrap_i = [](int i) noexcept { return ((i % kHSize) + kHSize) % kHSize; };
        for (int y = 0; y < kHSize; ++y) {
            for (int x = 0; x < kHSize; ++x) {
                f32 acc = 0.0f;
                for (int k = -kRadius; k <= kRadius; ++k) {
                    acc += h_src[static_cast<usize>(y * kHSize + wrap_i(x + k))] * kernel[static_cast<usize>(k + kRadius)];
                }
                h_tmp[static_cast<usize>(y * kHSize + x)] = acc;
            }
        }
        for (int y = 0; y < kHSize; ++y) {
            for (int x = 0; x < kHSize; ++x) {
                f32 acc = 0.0f;
                for (int k = -kRadius; k <= kRadius; ++k) {
                    acc += h_tmp[static_cast<usize>(wrap_i(y + k) * kHSize + x)] * kernel[static_cast<usize>(k + kRadius)];
                }
                h_smooth[static_cast<usize>(y * kHSize + x)] = acc;
            }
        }

        auto sample_smooth_h = [&](f32 u, f32 v) noexcept -> f32 {
            u = u - std::floor(u);
            v = v - std::floor(v);
            const f32 fx = u * static_cast<f32>(kHSize) - 0.5f;
            const f32 fy = v * static_cast<f32>(kHSize) - 0.5f;
            const int x0 = static_cast<int>(std::floor(fx));
            const int y0 = static_cast<int>(std::floor(fy));
            const f32 tx = fx - static_cast<f32>(x0);
            const f32 ty = fy - static_cast<f32>(y0);
            const f32 a = h_smooth[static_cast<usize>(wrap_i(y0) * kHSize + wrap_i(x0))];
            const f32 b = h_smooth[static_cast<usize>(wrap_i(y0) * kHSize + wrap_i(x0 + 1))];
            const f32 c = h_smooth[static_cast<usize>(wrap_i(y0 + 1) * kHSize + wrap_i(x0))];
            const f32 d = h_smooth[static_cast<usize>(wrap_i(y0 + 1) * kHSize + wrap_i(x0 + 1))];
            return (a * (1.0f - tx) + b * tx) * (1.0f - ty) + (c * (1.0f - tx) + d * tx) * ty;
        };

        // High-density smooth mesh (~1.5 cm vertical, ~2.0-2.9 cm horizontal) eliminates close-up triangle faceting
        constexpr int kWallNz = 640;
        constexpr int kInnerEnd = 104;
        constexpr int kTopEnd = 132;
        constexpr int kWallNy = 148;
        constexpr f32 kInnerX = 2.28f;
        constexpr f32 kThick = 0.65f;
        constexpr f32 kMaxZ = 15.0f;

        // Register colliders for player movement & bullet raycasts in Guerrilla
        colliders.push_back({{-2.60f, 0.80f, 8.8f}, {0.42f, 0.85f, 6.2f}, 0.0f});
        colliders.push_back({{ 2.60f, 0.72f, 5.7f}, {0.42f, 0.78f, 9.3f}, 0.0f});

        const usize per_wall_verts = static_cast<usize>(kWallNz * kWallNy);
        stone_walls.vertices.resize(per_wall_verts * 2);
        stone_walls.indices.clear();
        stone_walls.indices.reserve(static_cast<usize>(2 * (kWallNz - 1) * (kWallNy - 1) * 6));

        for (int side = 0; side < 2; ++side) {
            const u32 base = static_cast<u32>(static_cast<usize>(side) * per_wall_verts);
            for (int iz = 0; iz < kWallNz - 1; ++iz) {
                for (int iy = 0; iy < kWallNy - 1; ++iy) {
                    const u32 a = base + static_cast<u32>(iz * kWallNy + iy);
                    const u32 b = a + 1;
                    const u32 c = a + static_cast<u32>(kWallNy);
                    const u32 d = c + 1;
                    if (side == 0) {
                        stone_walls.indices.insert(stone_walls.indices.end(), {a, b, d, a, d, c});
                    } else {
                        stone_walls.indices.insert(stone_walls.indices.end(), {a, d, b, a, c, d});
                    }
                }
            }

            const f32 sign_x = (side == 0) ? -1.0f : 1.0f;
            const f32 normal_x = -sign_x;
            const f32 base_x = sign_x * kInnerX;
            const f32 side_min_z = (side == 0) ? 2.6f : -3.6f;
            const f32 side_height = (side == 0) ? 1.54f : 1.38f;
            const usize v_offset = static_cast<usize>(side) * per_wall_verts;

            // Pass 1: Displace vertices cleanly using the smoothed macro-relief heightfield
            std::vector<Vec3> base_normals(per_wall_verts);
            for (int iz = 0; iz < kWallNz; ++iz) {
                const f32 z = side_min_z + (kMaxZ - side_min_z) * (static_cast<f32>(iz) / static_cast<f32>(kWallNz - 1));
                const f32 u = (11.0f - z) * 0.68f;
                const f32 far_cap = std::clamp((z - side_min_z) * 4.0f, 0.0f, 1.0f);
                const f32 local_thick = kThick * far_cap;

                for (int iy = 0; iy < kWallNy; ++iy) {
                    f32 profile_x = base_x;
                    f32 profile_y = 0.0f;
                    Vec3 base_n{normal_x, 0.0f, 0.0f};
                    f32 v = 0.0f;
                    f32 disp_amp = 0.060f;

                    if (iy <= kInnerEnd) {
                        const f32 t = static_cast<f32>(iy) / static_cast<f32>(kInnerEnd);
                        profile_x = base_x;
                        profile_y = side_height * t;
                        base_n = {normal_x, 0.0f, 0.0f};
                        v = 1.0f - t * 0.90f;
                        disp_amp = 0.060f * (0.85f + 0.15f * far_cap);
                    } else if (iy <= kTopEnd) {
                        const f32 t = static_cast<f32>(iy - kInnerEnd) / static_cast<f32>(kTopEnd - kInnerEnd);
                        const f32 ang = t * math::kPi;
                        profile_x = base_x - normal_x * (local_thick * t);
                        profile_y = side_height + std::sin(ang) * 0.045f;
                        base_n = math::normalize(Vec3{normal_x * std::cos(ang), std::sin(ang) + 0.45f, 0.0f});
                        v = 0.10f - t * 0.38f;
                        disp_amp = 0.045f * far_cap;
                    } else {
                        const f32 t = static_cast<f32>(iy - kTopEnd) / static_cast<f32>(kWallNy - 1 - kTopEnd);
                        profile_x = base_x - normal_x * local_thick;
                        profile_y = side_height * (1.0f - t);
                        base_n = {-normal_x, 0.0f, 0.0f};
                        v = -0.28f - t * 0.85f;
                        disp_amp = 0.025f * far_cap;
                    }

                    const f32 disp = sample_smooth_h(u, v);
                    const Vec3 pos = Vec3{profile_x, profile_y, z} + base_n * (disp * disp_amp);
                    // Gentle macro cavity modulation; sharp crevice occlusion is supplied per-pixel by wall_displacement.png
                    const f32 cavity = std::clamp(0.68f + 0.34f * disp, 0.62f, 1.02f);

                    const usize local_idx = static_cast<usize>(iz * kWallNy + iy);
                    base_normals[local_idx] = base_n;
                    stone_walls.vertices[v_offset + local_idx] = {
                        pos,
                        base_n,
                        {cavity, cavity, cavity, 0.0f},
                        {u, v},
                        {0.0f, 0.0f, -1.0f, 1.0f},
                    };
                }
            }

            // Pass 2: Compute exact smooth geometric normals and exact UV-aligned tangents/handedness from neighbors
            auto vert_pos = [&](int iz, int iy) -> const Vec3& {
                iz = std::clamp(iz, 0, kWallNz - 1);
                iy = std::clamp(iy, 0, kWallNy - 1);
                return stone_walls.vertices[v_offset + static_cast<usize>(iz * kWallNy + iy)].position;
            };
            for (int iz = 0; iz < kWallNz; ++iz) {
                for (int iy = 0; iy < kWallNy; ++iy) {
                    const usize local_idx = static_cast<usize>(iz * kWallNy + iy);
                    const Vec3 dP_dz = vert_pos(iz + 1, iy) - vert_pos(iz - 1, iy);
                    const Vec3 dP_dy = vert_pos(iz, iy + 1) - vert_pos(iz, iy - 1);
                    Vec3 wn = math::cross(dP_dz, dP_dy);
                    const f32 nlen = math::length(wn);
                    wn = (nlen > 1e-6f) ? (wn * (1.0f / nlen)) : base_normals[local_idx];
                    if (math::dot(wn, base_normals[local_idx]) < 0.0f) {
                        wn = wn * -1.0f;
                    }
                    // +u goes along -z, so tangent T = normalize(-dP_dz) orthogonalized to wn
                    Vec3 t_raw = dP_dz * -1.0f;
                    t_raw = t_raw - wn * math::dot(wn, t_raw);
                    const f32 tlen = math::length(t_raw);
                    const Vec3 t_vec = (tlen > 1e-6f) ? (t_raw * (1.0f / tlen)) : Vec3{0.0f, 0.0f, -1.0f};
                    // OpenGL normal map has +G pointing toward -v (which is +iy on all 3 faces since v decreases with iy)
                    const f32 handedness = (math::dot(math::cross(wn, t_vec), dP_dy) >= 0.0f) ? 1.0f : -1.0f;

                    auto& vtx = stone_walls.vertices[v_offset + local_idx];
                    vtx.normal = wn;
                    vtx.tangent = {t_vec.x, t_vec.y, t_vec.z, handedness};
                }
            }
        }
        grass_along({-2.08f, 0, 12.5f}, {-2.08f, 0, 2.6f}, 1.8f);
        grass_along({ 2.08f, 0, 12.5f}, { 2.08f, 0, -3.6f}, 1.8f);
    }

    // Build 3D displaced PBR Dirt & Gravel Village Street (dirt_*.png), Wet Mud Puddle Banks (mud_*.png),
    // and Forest Floor Patches under the olive trees (forest_*.png).
    void pbr_ground_surfaces() {
        auto load_tex = [](const std::string& fname) {
            image::PngImage img;
            const std::array<std::string, 3> dirs = {
                "assets/textures",
                "C:/Users/ipiou/.gemini/antigravity/scratch/NOVA_guerrilla_c519e86/NOVA/assets/textures",
                "../../../../assets/textures",
            };
            for (const auto& d : dirs) {
                if (image::load_png(d + "/" + fname, img) && img.valid()) return img;
            }
            return img;
        };

        const image::PngImage dirt_h = load_tex("dirt_height.png");
        const image::PngImage mud_h  = load_tex("mud_height.png");
        const image::PngImage for_h  = load_tex("forest_height.png");

        // 1. Subdivided 3D Displaced Dirt & Pebbles Village Street (x in [-2.65..2.65], z in [-38..15])
        {
            constexpr int kNx = 60;
            constexpr int kNz = 220;
            constexpr f32 kMinX = -2.65f, kMaxX = 2.65f;
            constexpr f32 kMinZ = -38.0f, kMaxZ = 15.0f;
            dirt_road.vertices.reserve(static_cast<usize>(kNx * kNz));
            for (int j = 0; j < kNz; ++j) {
                const f32 tz = static_cast<f32>(j) / static_cast<f32>(kNz - 1);
                const f32 z = kMinZ + (kMaxZ - kMinZ) * tz;
                for (int i = 0; i < kNx; ++i) {
                    const f32 tx = static_cast<f32>(i) / static_cast<f32>(kNx - 1);
                    const f32 x = kMinX + (kMaxX - kMinX) * tx;
                    const f32 u = x * 0.42f;
                    const f32 v = z * 0.42f;
                    const f32 edge = std::clamp((2.65f - std::abs(x)) * 2.5f, 0.0f, 1.0f);
                    const f32 disp = dirt_h.valid() ? (dirt_h.sample_height(u, v) - 0.45f) * 0.048f : 0.0f;
                    // Carve shallow natural rainwater basins so mirror puddles emerge organically through pebbles & wet mud!
                    auto basin = [&](f32 bx, f32 bz, f32 brx, f32 brz) {
                        const f32 dx = (x - bx) / brx;
                        const f32 dz = (z - bz) / brz;
                        const f32 wob = (fbm(x * 1.25f + 2.3f, z * 1.25f - 1.7f, 3) - 0.5f) * 0.45f;
                        const f32 rr = std::hypot(dx, dz) + wob;
                        return smoothstep(1.05f, 0.35f, rr);
                    };
                    const f32 dip = std::max({basin(-0.20f, 1.35f, 1.15f, 2.35f),
                                              basin( 0.45f, -6.20f, 1.05f, 1.85f),
                                              basin(-0.65f, -13.0f, 1.10f, 1.90f)});
                    const f32 y = terrain_height_at(x, z) + (0.024f + disp - 0.036f * dip) * edge - 0.015f * (1.0f - edge);
                    dirt_road.vertices.push_back({{x, y, z}, {0.0f, 1.0f, 0.0f}, {1, 1, 1, 0}, {u, v}, {1, 0, 0, 1}});
                }
            }
            for (int j = 0; j < kNz - 1; ++j) {
                for (int i = 0; i < kNx - 1; ++i) {
                    const u32 a = static_cast<u32>(j * kNx + i);
                    const u32 b = a + 1;
                    const u32 c = a + static_cast<u32>(kNx);
                    const u32 d = c + 1;
                    dirt_road.indices.insert(dirt_road.indices.end(), {a, c, d, a, d, b});
                }
            }
        }

        // Helper for organic noise-edged 3D displaced ground patches (wet mud around puddles, forest floor under trees)
        auto add_patch = [&](MeshData& dst, const image::PngImage& hmap, Vec3 center, f32 rx, f32 rz, f32 lift, f32 uv_scale) {
            constexpr int kN = 40;
            const u32 base = static_cast<u32>(dst.vertices.size());
            for (int j = 0; j < kN; ++j) {
                const f32 fy = (static_cast<f32>(j) / static_cast<f32>(kN - 1)) * 2.0f - 1.0f;
                for (int i = 0; i < kN; ++i) {
                    const f32 fx = (static_cast<f32>(i) / static_cast<f32>(kN - 1)) * 2.0f - 1.0f;
                    const f32 x = center.x + fx * rx;
                    const f32 z = center.z + fy * rz;
                    const f32 wobble = (fbm(x * 1.35f + 4.1f, z * 1.35f - 2.3f, 3) - 0.5f) * 0.55f;
                    const f32 r = std::hypot(fx, fy) + wobble;
                    const f32 mask = std::clamp((0.88f - r) * 2.8f, 0.0f, 1.0f);
                    const f32 u = x * uv_scale;
                    const f32 v = z * uv_scale;
                    const f32 disp = hmap.valid() ? (hmap.sample_height(u, v) - 0.45f) * 0.036f : 0.0f;
                    const f32 y = terrain_height_at(x, z) + (lift + disp) * mask - 0.035f * (1.0f - mask);
                    dst.vertices.push_back({{x, y, z}, {0.0f, 1.0f, 0.0f}, {1, 1, 1, 0}, {u, v}, {1, 0, 0, 1}});
                }
            }
            for (int j = 0; j < kN - 1; ++j) {
                for (int i = 0; i < kN - 1; ++i) {
                    const u32 a = base + static_cast<u32>(j * kN + i);
                    const u32 b = a + 1;
                    const u32 c = a + static_cast<u32>(kN);
                    const u32 d = c + 1;
                    dst.indices.insert(dst.indices.end(), {a, c, d, a, d, b});
                }
            }
        };

        // 2. Organic Wet Mud & Bootprints Patches nestled around the rainwater puddles (mud_*.png)
        add_patch(mud_patches, mud_h, {-0.20f, 0.0f,  1.35f}, 1.85f, 3.15f, 0.012f, 0.56f);
        add_patch(mud_patches, mud_h, { 0.45f, 0.0f, -6.20f}, 1.80f, 2.50f, 0.014f, 0.52f);
        add_patch(mud_patches, mud_h, {-0.65f, 0.0f, -13.0f}, 2.00f, 2.75f, 0.014f, 0.52f);

        // 3. Organic Forest Floor (Twigs, Fallen Leaves & Moss) under the olive trees (forest_*.png)
        add_patch(forest_floor, for_h, {-1.72f, 0.0f,  5.8f}, 1.55f, 2.60f, 0.030f, 0.60f);
        add_patch(forest_floor, for_h, {-1.85f, 0.0f, -10.8f}, 1.45f, 2.25f, 0.030f, 0.60f);
    }

    void draped_rug(f32 z_min, f32 z_max) {
        constexpr int kNz = 24;
        constexpr int kNy = 22;
        const u32 base = static_cast<u32>(two_sided.vertices.size());
        const Vec4 col = renderer::material_color(Material::Cloth, {0.45f, 0.10f, 0.07f});

        for (int iz = 0; iz <= kNz; ++iz) {
            const f32 u = static_cast<f32>(iz) / static_cast<f32>(kNz);
            const f32 z = z_min + (z_max - z_min) * u;
            const f32 fold = std::sin(u * math::kPi * 7.0f) * 0.016f;
            for (int iy = 0; iy <= kNy; ++iy) {
                const f32 v = static_cast<f32>(iy) / static_cast<f32>(kNy);
                Vec3 pos{};
                Vec3 n{1.0f, 0.0f, 0.0f};
                if (v < 0.24f) {
                    const f32 t = v / 0.24f;
                    pos = {-2.72f + 0.54f * t, 1.58f + std::sin(t * math::kPi) * 0.05f, z};
                    n = math::normalize(Vec3{t * 0.6f, 1.0f, 0.0f});
                } else {
                    const f32 t = (v - 0.24f) / 0.76f;
                    pos = {-2.18f + fold * (0.35f + 0.65f * t), 1.56f - t * 1.14f, z};
                    n = {1.0f, 0.05f, 0.0f};
                }
                two_sided.vertices.push_back({pos, n, col});
            }
        }
        for (int iz = 0; iz < kNz; ++iz) {
            for (int iy = 0; iy < kNy; ++iy) {
                const u32 a = base + static_cast<u32>(iz * (kNy + 1) + iy);
                const u32 b = a + 1;
                const u32 c = a + static_cast<u32>(kNy + 1);
                const u32 d = c + 1;
                two_sided.indices.insert(two_sided.indices.end(), {a, b, d, a, d, c});
            }
        }
    }

    void scatter_rocks(const Vec3& a, const Vec3& b, int count) {
        for (int i = 0; i < count; ++i) {
            const Vec3 p = math::lerp(a, b, rng_.uniform()) + Vec3{rng_.range(-0.45f, 0.45f), 0, rng_.range(-0.45f, 0.45f)};
            const f32 s = rng_.range(0.05f, 0.16f);
            stone(opaque, p + Vec3{0, s * 0.25f, 0}, {s, s * 0.55f, s * 1.15f}, rng_.range(0, 6.28f),
                  Vec3{0.42f, 0.34f, 0.25f} * rng_.range(0.8f, 1.1f), Material::Stone);
        }
    }

    MeshData& cylinder_target() { return opaque; }
    const MeshData& cylinder() const { return cylinder_; }
    Rng& rng() { return rng_; }

private:
    MeshData cube_;
    MeshData sphere_;
    MeshData cylinder_;
    Rng rng_{1954};
};


void build_village(Builder& b) {
    constexpr f32 kHalfPi = math::kPi * 0.5f;
    const Vec3 warm_plaster{0.46f, 0.37f, 0.26f};
    const Vec3 pale_plaster{0.52f, 0.43f, 0.32f};

    // Left & Right foreground walls: 3D displaced PBR stone walls using the user's textures
    b.pbr_displaced_stone_walls();
    b.pbr_ground_surfaces();
    b.draped_rug(4.5f, 6.1f); // woven rug draped over the Left PBR stone wall
    b.house({{-6.2f, 0, -2.0f}, 13.5f, 6.0f, 4.2f, kHalfPi, warm_plaster, 1.6f, true});
    b.house({{-6.6f, 0, -16.5f}, 9.0f, 6.0f, 4.0f, kHalfPi, pale_plaster, -1.0f});
    b.house({{-7.0f, 0, -27.5f}, 8.0f, 7.0f, 3.8f, kHalfPi, warm_plaster, 0.5f});
    b.dry_stone_wall({-3.2f, 0, -9.7f}, {-3.2f, 0, -12.0f}, 1.2f, 0.6f);

    // Right side (facades face -X).
    const Mat4 house_c = b.house({{7.6f, 0, -9.0f}, 9.0f, 7.0f, 3.8f, -kHalfPi, pale_plaster, 2.0f});
    b.house({{8.2f, 0, -22.0f}, 8.0f, 7.5f, 4.4f, -kHalfPi, warm_plaster, -1.2f, true});
    b.house({{9.0f, 0, -33.0f}, 7.0f, 7.0f, 3.6f, -kHalfPi, pale_plaster});

    // Ladder leaning on house C.
    for (f32 side : {-0.25f, 0.25f}) {
        b.box({3.75f, 1.6f, -6.0f + side}, {0.07f, 3.4f, 0.07f}, Material::Wood, {0.4f, 0.3f, 0.2f});
    }
    for (f32 y = 0.4f; y < 3.2f; y += 0.4f) {
        b.box({3.75f, y, -6.0f}, {0.05f, 0.05f, 0.5f}, Material::Wood, {0.38f, 0.28f, 0.18f});
    }
    // Woven baskets.
    for (const Vec3& p : {Vec3{3.7f, 0, -3.2f}, Vec3{3.6f, 0, -2.4f}}) {
        renderer::append(b.cylinder_target(), b.cylinder(), math::translate(p) * math::scale({0.38f, 0.55f, 0.38f}),
                         renderer::material_color(Material::Wood, {0.55f, 0.43f, 0.26f}));
    }
    // Flag pole on the roof of house C.
    const Vec3 pole_base = math::transform_point(house_c, {2.5f, 3.9f, -1.0f});
    renderer::append(b.cylinder_target(), b.cylinder(),
                     math::translate(pole_base) * math::scale({0.05f, 6.0f, 0.05f}),
                     renderer::material_color(Material::Metal, {0.35f, 0.33f, 0.3f}));
    b.flag(pole_base + Vec3{0.05f, 5.9f, 0}, 2.1f, 1.4f);

    // End of the street (facade faces +Z, towards the camera).
    b.house({{0.0f, 0, -40.0f}, 10.0f, 6.0f, 4.6f, 0.0f, warm_plaster, 0.0f, true});
    b.house({{-12.0f, 0, -42.0f}, 8.0f, 6.0f, 5.0f, 0.3f, pale_plaster});
    b.house({{13.0f, 0, -44.0f}, 9.0f, 6.0f, 4.0f, -0.2f, warm_plaster});
    // Market awning.
    b.box({3.6f, 2.5f, -29.0f}, {2.6f, 0.04f, 3.0f}, Material::Cloth, {0.82f, 0.76f, 0.62f});

    // Olive trees (gnarled trunks + individual 3D leafy sprigs casting dappled shadows on the sunlit plaster wall).
    b.olive_tree({-3.85f, 0, 3.8f}, {-2.15f, 4.45f, 2.9f}, 2.65f);
    b.olive_tree({-3.9f, 0, -11.0f}, {-3.4f, 4.0f, -11.2f}, 1.9f);
    b.olive_tree({11.0f, 0, -14.0f}, {11.0f, 4.4f, -14.0f}, 2.2f);
    b.olive_tree({-15.0f, 0, -20.0f}, {-15.0f, 4.8f, -20.0f}, 2.6f);

    // Street dressing: hero foreground rainwater puddle with SSR mirror reflections + secondary street puddles.
    b.puddle({-0.20f, 0, 1.35f}, 1.05f, 2.20f);
    b.puddle({ 0.45f, 0, -6.20f}, 0.95f, 1.75f);
    b.puddle({-0.65f, 0, -13.0f}, 1.00f, 1.80f);
    b.scatter_rocks({-2.2f, 0, 9.0f}, {-2.2f, 0, -20.0f}, 28);
    b.scatter_rocks({2.2f, 0, 9.0f}, {2.2f, 0, -20.0f}, 24);
    b.grass_along({-2.10f, 0, 11.0f}, {-2.10f, 0, -25.0f}, 1.35f);
    b.grass_along({ 2.15f, 0, 11.0f}, { 2.15f, 0, -25.0f}, 1.35f);

    // Watchtower on the hill and scattered houses on the slopes.
    const Vec3 tower{kTowerHill.x, terrain_height(kTowerHill.x, kTowerHill.z), kTowerHill.z};
    b.colliders.push_back({tower + Vec3{0, 3.5f, 0}, {2.0f, 4.5f, 2.0f}, 0.0f});
    renderer::append(b.cylinder_target(), b.cylinder(),
                     math::translate(tower - Vec3{0, 1, 0}) * math::scale({2.2f, 9.0f, 2.2f}),
                     renderer::material_color(Material::Stone, {0.62f, 0.55f, 0.45f}));
    renderer::append(b.cylinder_target(), renderer::make_cylinder(10, 3.0f, 0.2f, 3.0f),
                     math::translate(tower + Vec3{0, 8.0f, 0}),
                     renderer::material_color(Material::Wood, {0.35f, 0.24f, 0.16f}));
    Rng& rng = b.rng();
    for (int i = 0; i < 26; ++i) {
        const f32 a = rng.range(-2.6f, -0.5f); // mostly in front of the camera
        const f32 r = rng.range(48.0f, 120.0f);
        const f32 x = std::cos(a) * r, z = std::sin(a) * r;
        const f32 w = rng.range(5.0f, 9.0f);
        const f32 base_h = terrain_height(x, z);
        b.house({{x, base_h - 2.2f, z}, w, rng.range(5.0f, 7.0f), rng.range(4.8f, 6.4f),
                 rng.range(0, 6.28f), rng.uniform() < 0.5f ? warm_plaster : pale_plaster, 0.0f});
        if (rng.uniform() < 0.6f) {
            const f32 tx = x + rng.range(-8, 8), tz = z + rng.range(-8, 8);
            const f32 ty = terrain_height(tx, tz);
            b.olive_tree({tx, ty - 0.5f, tz}, {tx, ty + 3.6f, tz}, rng.range(1.6f, 2.4f));
        }
    }
}

} // namespace

f32 terrain_height(f32 x, f32 z) {
    return terrain_height_at(x, z);
}

Scene build_scene() {
    Builder b;
    build_village(b);
    Scene scene;
    scene.terrain = build_terrain();
    scene.opaque = std::move(b.opaque);
    scene.foliage = std::move(b.foliage);
    scene.two_sided = std::move(b.two_sided);
    scene.water = std::move(b.water);
    scene.stone_walls = std::move(b.stone_walls);
    scene.dirt_road = std::move(b.dirt_road);
    scene.mud_patches = std::move(b.mud_patches);
    scene.forest_floor = std::move(b.forest_floor);
    scene.wood_props = std::move(b.wood_props);
    scene.tree_bark = std::move(b.tree_bark);
    scene.colliders = std::move(b.colliders);
    return scene;
}

VillageMaterials upload_village_materials(renderer::WorldRenderer& world) {
    auto load_png_raw = [&](const std::string& fname) -> image::PngImage {
        image::PngImage img;
        const std::array<std::string, 3> dirs = {
            "assets/textures",
            "C:/Users/ipiou/.gemini/antigravity/scratch/NOVA_guerrilla_c519e86/NOVA/assets/textures",
            "../../../../assets/textures",
        };
        for (const auto& d : dirs) {
            if (image::load_png(d + "/" + fname, img) && img.valid()) {
                return img;
            }
        }
        return {};
    };

    auto upload_png = [&](const std::string& fname, bool srgb) -> rhi::TextureHandle {
        const image::PngImage img = load_png_raw(fname);
        return img.valid() ? world.upload_texture({img.width, img.height, img.rgba}, srgb) : rhi::TextureHandle{};
    };

    // Remap roughness PNG into physically matte dielectric range [min_r .. max_r] and zero metallic channel B,
    // so dark mortar cracks between dry stones NEVER act like glossy chrome/glass mirrors!
    auto upload_roughness = [&](const std::string& fname, f32 min_r, f32 max_r, bool invert_dark_crevices) -> rhi::TextureHandle {
        image::PngImage img = load_png_raw(fname);
        if (!img.valid()) return {};
        for (usize i = 0; i < img.rgba.size(); i += 4) {
            f32 r = static_cast<f32>(img.rgba[i + 1]) / 255.0f;
            if (invert_dark_crevices && r < 0.35f) {
                // Deep dark crevices in raw roughness maps are actually rough dry mortar, not glossy mirror!
                r = 0.96f;
            }
            const f32 remapped = std::clamp(min_r + (max_r - min_r) * r, min_r, max_r);
            const u8 g = static_cast<u8>(std::clamp(remapped * 255.0f + 0.5f, 0.0f, 255.0f));
            img.rgba[i + 0] = g;
            img.rgba[i + 1] = g;   // G = roughness
            img.rgba[i + 2] = 0u;  // B = metallic (0 for dielectric stone/dirt/wood/bark)
            img.rgba[i + 3] = 255u;
        }
        return world.upload_texture({img.width, img.height, img.rgba}, false);
    };

    auto make_mat = [&](const std::string& prefix, const std::string& ao_file, f32 min_r, f32 max_r,
                        bool invert_crevices, f32 nrm_scale, f32 ao_str,
                        math::Vec4 tint = {1.0f, 1.0f, 1.0f, 1.0f}) {
        renderer::PbrMaterial mat;
        mat.base_color = upload_png(prefix + "_albedo.png", true);
        mat.normal = upload_png(prefix + "_normal.png", false);
        mat.metallic_roughness = upload_roughness(prefix + "_roughness.png", min_r, max_r, invert_crevices);
        mat.occlusion = upload_png(ao_file, false);
        mat.base_color_factor = tint;
        mat.metallic = 0.0f;
        mat.roughness = 1.0f;
        mat.normal_scale = nrm_scale;
        mat.occlusion_strength = ao_str;
        return world.add_material(mat);
    };

    VillageMaterials mats;
    mats.stone_wall   = make_mat("wall",   "wall_displacement.png", 0.78f, 0.98f, true,  1.20f, 0.78f, {0.82f, 0.89f, 0.96f, 1.0f});
    mats.dirt_road    = make_mat("dirt",   "dirt_height.png",       0.80f, 0.98f, true,  1.15f, 0.45f);
    mats.mud_patches  = make_mat("mud",    "mud_ao.png",            0.22f, 0.72f, false, 1.25f, 0.65f);
    mats.forest_floor = make_mat("forest", "forest_height.png",     0.65f, 0.95f, false, 1.20f, 0.55f);
    mats.wood_props   = make_mat("wood",   "wood_ao.png",           0.76f, 0.96f, true,  1.15f, 0.60f);
    mats.tree_bark    = make_mat("bark",   "bark_ao.png",           0.82f, 0.98f, true,  1.30f, 0.65f);
    NOVA_INFO("PNG", "uploaded 6 complete PBR PNG material sets (Wall, Dirt, Mud, Forest, Wood, Bark) to GPU");
    return mats;
}

} // namespace nova::village
