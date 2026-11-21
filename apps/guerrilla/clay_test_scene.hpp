#pragma once

// NOVA PRISM Architectural Clay GI Lighting Benchmark Scene
// Pure matte architectural clay geometry (zero noisy textures) designed to test direct golden-hour
// lighting, multi-bounce warm-floor vs cool-sky GI, 36-sample Bilinear PCSS shadows, beveled edge
// specular highlights, faceted low-poly foliage shadows, and crystal-clear SSR mirror puddles.

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/world_renderer.hpp"
#include "village_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace nova::clay {

using math::Mat4;
using math::Vec2;
using math::Vec3;
using math::Vec4;
using renderer::Material;
using renderer::MeshData;
using renderer::Vertex;

struct ClayScene {
    MeshData opaque;
    MeshData two_sided;
    MeshData water;
    std::vector<village::Collider> colliders;
};

namespace detail {

inline u32 hash_u32(u32 x) noexcept {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

inline f32 rand01(u32 seed) noexcept {
    return static_cast<f32>(hash_u32(seed) & 0xFFFFu) / 65535.0f;
}

inline Vec4 clay_col(f32 lum = 0.75f, f32 warm = 0.0f) noexcept {
    // Sun-baked sand clay (warm greybox look of the art-direction reference).
    return renderer::material_color(Material::Clay, {lum * 1.0f + warm * 0.015f, lum * 0.86f, lum * 0.70f - warm * 0.01f});
}

// Smooth organic puddle basin depression query (returns depth <= 0.0 in meters)
struct PuddleDef {
    f32 cx, cz;
    f32 rx, rz;
    f32 rot;
    f32 phase;
};

inline constexpr std::array<PuddleDef, 3> kPuddles = {{
    { 0.52f,  3.55f, 1.02f, 0.62f, -0.22f, 0.4f}, // Foreground hero puddle (center-right)
    { 0.92f, -0.55f, 0.92f, 0.44f,  0.14f, 2.1f}, // Midground puddle
    { 0.38f, -5.15f, 1.05f, 0.36f, -0.10f, 4.3f}, // Distant lane puddle
}};

inline f32 puddle_sdf(const PuddleDef& p, f32 x, f32 z) noexcept {
    const f32 dx = x - p.cx, dz = z - p.cz;
    const f32 c = std::cos(p.rot), s = std::sin(p.rot);
    const f32 lx = (dx * c - dz * s) / p.rx;
    const f32 lz = (dx * s + dz * c) / p.rz;
    const f32 ang = std::atan2(lz, lx);
    const f32 wobble = 1.0f + 0.14f * std::sin(ang * 2.0f + p.phase)
                            + 0.10f * std::cos(ang * 3.0f - p.phase * 1.4f)
                            + 0.05f * std::sin(ang * 5.0f + 0.8f);
    return std::hypot(lx, lz) / wobble; // < 1.0 inside puddle
}

inline f32 ground_height(f32 x, f32 z) noexcept {
    // Gentle wind-swept architectural clay sand undulation catching low raking golden sunlight
    f32 h = 0.022f * std::sin(x * 0.85f + z * 0.45f)
          + 0.014f * std::cos(z * 1.15f - x * 0.55f)
          + 0.008f * std::sin(x * 2.1f - z * 1.7f);
    // Keep doorway step area & far horizon super tidy
    const f32 lane_fade = std::clamp(1.0f - std::max(0.0f, -z - 14.0f) / 14.0f, 0.15f, 1.0f);
    h *= lane_fade;
    // Smooth shallow basin depressions for the 3 liquid-mirror puddles
    for (const auto& p : kPuddles) {
        const f32 d = puddle_sdf(p, x, z);
        if (d < 1.24f) {
            const f32 t = std::clamp((1.24f - d) / 0.34f, 0.0f, 1.0f);
            const f32 sm = t * t * (3.0f - 2.0f * t);
            h = math::lerp(h, -0.048f, sm);
        }
    }
    return h;
}

// Watertight high-precision 8x8 rounded-bevel architectural clay block with silky circular-arc G1 analytic normals!
inline void add_beveled_block(MeshData& dst, Vec3 center, Vec3 size, f32 bevel, const Vec4& col, f32 yaw_rad = 0.0f) {
    const f32 r = std::min({bevel, size.x * 0.44f, size.y * 0.44f, size.z * 0.44f});
    const Vec3 core{std::max(0.0f, size.x * 0.5f - r),
                    std::max(0.0f, size.y * 0.5f - r),
                    std::max(0.0f, size.z * 0.5f - r)};
    constexpr int kN = 8;
    constexpr f32 kSign[kN] = {-1.0f, -1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f, 1.0f};
    constexpr f32 kArc[kN]  = {-1.0f, -0.6681786f, -0.3249197f, 0.0f, 0.0f, 0.3249197f, 0.6681786f, 1.0f};
    const f32 cy = std::cos(yaw_rad), sy = std::sin(yaw_rad);
    auto rot_y = [&](Vec3 v) noexcept -> Vec3 {
        return {cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z};
    };

    struct FaceDef { Vec3 u, v, n; };
    const FaceDef faces[6] = {
        {{ 1, 0, 0}, {0,  1, 0}, {0, 0,  1}}, // +Z
        {{-1, 0, 0}, {0,  1, 0}, {0, 0, -1}}, // -Z
        {{ 0, 0,-1}, {0,  1, 0}, {1, 0,  0}}, // +X
        {{ 0, 0, 1}, {0,  1, 0}, {-1,0,  0}}, // -X
        {{ 1, 0, 0}, {0,  0,-1}, {0, 1,  0}}, // +Y
        {{ 1, 0, 0}, {0,  0, 1}, {0,-1,  0}}, // -Y
    };

    for (const auto& f : faces) {
        const u32 base = static_cast<u32>(dst.vertices.size());
        for (int j = 0; j < kN; ++j) {
            for (int i = 0; i < kN; ++i) {
                const Vec3 c = f.u * (kSign[i] * std::abs(math::dot(f.u, core)))
                             + f.v * (kSign[j] * std::abs(math::dot(f.v, core)))
                             + f.n * std::abs(math::dot(f.n, core));
                const Vec3 dir = math::normalize(f.u * kArc[i] + f.v * kArc[j] + f.n);
                const Vec3 pos = center + rot_y(c + dir * r);
                const Vec3 nrm = rot_y(dir);
                dst.vertices.push_back({pos, nrm, col, {0, 0}, {0, 0, 0, 0}});
            }
        }
        for (int j = 0; j < kN - 1; ++j) {
            for (int i = 0; i < kN - 1; ++i) {
                const u32 a = base + static_cast<u32>(j * kN + i);
                const u32 b = a + 1;
                const u32 c = a + kN;
                const u32 d = c + 1;
                dst.indices.insert(dst.indices.end(), {a, b, d, a, d, c});
            }
        }
    }
}

// Flat-shaded faceted low-poly polyhedron (sculptural clay canopy cluster or angular rock!)
inline void add_faceted_polyhedron(MeshData& dst, Vec3 center, Vec3 radii, u32 seed, const Vec4& col, u32 subdiv = 1) {
    MeshData ico = renderer::make_icosphere(subdiv);
    // Perturb shared vertex positions slightly by direction hash so facets have organic sculpted variety
    for (auto& v : ico.vertices) {
        const int hx = static_cast<int>(std::round(v.position.x * 100.0f));
        const int hy = static_cast<int>(std::round(v.position.y * 100.0f));
        const int hz = static_cast<int>(std::round(v.position.z * 100.0f));
        const u32 h = static_cast<u32>(hx * 73856093 ^ hy * 19349663 ^ hz * 83492791) ^ seed;
        const f32 jitter = 0.86f + 0.28f * rand01(h);
        v.position = {v.position.x * radii.x * jitter, v.position.y * radii.y * jitter, v.position.z * radii.z * jitter};
    }
    // Emit flat-shaded triangles with crisp face normals for authentic faceted low-poly clay lighting!
    for (usize i = 0; i + 2 < ico.indices.size(); i += 3) {
        const Vec3 p0 = center + ico.vertices[ico.indices[i + 0]].position;
        const Vec3 p1 = center + ico.vertices[ico.indices[i + 1]].position;
        const Vec3 p2 = center + ico.vertices[ico.indices[i + 2]].position;
        const Vec3 n = math::normalize(math::cross(p1 - p0, p2 - p0));
        const u32 base = static_cast<u32>(dst.vertices.size());
        dst.vertices.push_back({p0, n, col, {0, 0}, {0, 0, 0, 0}});
        dst.vertices.push_back({p1, n, col, {0, 0}, {0, 0, 0, 0}});
        dst.vertices.push_back({p2, n, col, {0, 0}, {0, 0, 0, 0}});
        dst.indices.insert(dst.indices.end(), {base, base + 1, base + 2});
    }
}

// Smooth tapered cylinder segment connecting arbitrary 3D points p0 -> p1 (tree trunks, branches, poles, ladder rails!)
inline void add_tube_segment(MeshData& dst, Vec3 p0, Vec3 p1, f32 r0, f32 r1, int segs, const Vec4& col) {
    const Vec3 axis = p1 - p0;
    const f32 len = math::length(axis);
    if (len < 1e-4f) return;
    const Vec3 up = axis * (1.0f / len);
    const Vec3 ref = std::abs(up.y) < 0.92f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const Vec3 rt = math::normalize(math::cross(up, ref));
    const Vec3 fw = math::cross(rt, up);
    const u32 base = static_cast<u32>(dst.vertices.size());
    for (int i = 0; i <= segs; ++i) {
        const f32 a = (static_cast<f32>(i) / static_cast<f32>(segs)) * 2.0f * math::kPi;
        const Vec3 dir = rt * std::cos(a) + fw * std::sin(a);
        dst.vertices.push_back({p0 + dir * r0, dir, col, {0, 0}, {0, 0, 0, 0}});
        dst.vertices.push_back({p1 + dir * r1, dir, col, {0, 0}, {0, 0, 0, 0}});
    }
    for (int i = 0; i < segs; ++i) {
        const u32 a = base + static_cast<u32>(i * 2);
        const u32 b = a + 1;
        const u32 c = a + 2;
        const u32 d = a + 3;
        dst.indices.insert(dst.indices.end(), {a, c, d, a, d, b});
    }
}

// Stacked staggered running-bond beveled clay block wall along Z (from z0 to z1 at x = cx)
inline void add_beveled_wall_z(MeshData& dst, f32 cx, f32 z0, f32 z1, int courses, f32 block_d, f32 block_h,
                               f32 block_l, f32 bevel, u32 seed, f32 y_base = -0.02f) {
    const f32 gap = 0.018f;
    for (int row = 0; row < courses; ++row) {
        const f32 y = y_base + (static_cast<f32>(row) + 0.5f) * (block_h + gap);
        f32 z = z0 - ((row % 2 == 1) ? block_l * 0.45f : 0.0f);
        int bi = 0;
        while (z < z1) {
            const u32 h = hash_u32(seed + static_cast<u32>(row * 97 + bi * 31));
            const f32 len = block_l * (0.86f + 0.28f * rand01(h));
            const f32 za = std::max(z, z0);
            const f32 zb = std::min(z + len - gap, z1);
            if (zb - za > 0.16f) {
                const f32 jitter_x = (rand01(h ^ 0xA5u) - 0.5f) * 0.024f;
                const f32 lum = 0.74f + (rand01(h ^ 0x5Au) - 0.5f) * 0.03f;
                add_beveled_block(dst, {cx + jitter_x, y, (za + zb) * 0.5f},
                                  {block_d, block_h, zb - za}, bevel, clay_col(lum));
            }
            z += len;
            ++bi;
        }
    }
}

// Stacked staggered running-bond beveled clay block wall along X (from x0 to x1 at z = cz)
inline void add_beveled_wall_x(MeshData& dst, f32 cz, f32 x0, f32 x1, int courses, f32 block_d, f32 block_h,
                               f32 block_l, f32 bevel, u32 seed, f32 y_base = -0.02f) {
    const f32 gap = 0.012f;
    for (int row = 0; row < courses; ++row) {
        const f32 y = y_base + (static_cast<f32>(row) + 0.5f) * (block_h + gap);
        f32 x = x0 - ((row % 2 == 1) ? block_l * 0.45f : 0.0f);
        int bi = 0;
        while (x < x1) {
            const u32 h = hash_u32(seed + static_cast<u32>(row * 113 + bi * 37));
            const f32 len = block_l * (0.88f + 0.24f * rand01(h));
            const f32 xa = std::max(x, x0);
            const f32 xb = std::min(x + len - gap, x1);
            if (xb - xa > 0.18f) {
                const f32 jitter_z = (rand01(h ^ 0xC3u) - 0.5f) * 0.020f;
                const f32 lum = 0.74f + (rand01(h ^ 0x3Cu) - 0.5f) * 0.03f;
                add_beveled_block(dst, {(xa + xb) * 0.5f, y, cz + jitter_z},
                                  {xb - xa, block_h, block_d}, bevel, clay_col(lum));
            }
            x += len;
            ++bi;
        }
    }
}

// Clean clay house with thin roof parapet/cornice overhang and optional deeply recessed doorway!
inline void add_clay_house(MeshData& dst, Vec3 center, Vec3 size, const Vec4& col,
                           int door_face = 0, f32 door_off = 0.0f, f32 door_w = 1.15f, f32 door_h = 2.25f,
                           bool add_step = false, bool add_chimney = false) {
    // door_face: 0=none, 1=+X, 2=-X, 3=+Z
    const f32 hx = size.x * 0.5f, hy = size.y * 0.5f, hz = size.z * 0.5f;
    const f32 y0 = center.y;
    const f32 rec = 0.34f; // deep architectural doorway recess!
    auto box = [&](Vec3 c, Vec3 s, const Vec4& cc) {
        renderer::append(dst, renderer::make_cube(1.0f), math::translate(c) * math::scale(s), cc);
    };

    if (door_face == 0) {
        box({center.x, y0 + hy, center.z}, size, col);
    } else if (door_face == 1) { // Recessed doorway on +X facade (Left Foreground House!)
        const f32 dz = center.z + door_off;
        const f32 z_min = center.z - hz, z_max = center.z + hz;
        const f32 d0 = dz - door_w * 0.5f, d1 = dz + door_w * 0.5f;
        // Solid rear body behind recess
        box({center.x - rec * 0.5f, y0 + hy, center.z}, {size.x - rec, size.y, size.z}, col);
        // Front pillars left & right of doorway + lintel header above doorway
        const f32 fx = center.x + hx - rec * 0.5f;
        box({fx, y0 + hy, (z_min + d0) * 0.5f}, {rec, size.y, d0 - z_min}, col);
        box({fx, y0 + hy, (d1 + z_max) * 0.5f}, {rec, size.y, z_max - d1}, col);
        box({fx, y0 + (door_h + size.y) * 0.5f, dz}, {rec, size.y - door_h, door_w}, col);
        // Recessed double-panel clay door + crisp vertical & horizontal seam grooves
        const f32 door_x = center.x + hx - rec + 0.04f;
        box({door_x, y0 + door_h * 0.5f, dz - door_w * 0.255f}, {0.06f, door_h - 0.04f, door_w * 0.48f}, clay_col(0.73f));
        box({door_x, y0 + door_h * 0.5f, dz + door_w * 0.255f}, {0.06f, door_h - 0.04f, door_w * 0.48f}, clay_col(0.74f));
        box({door_x + 0.025f, y0 + door_h * 0.36f, dz}, {0.025f, 0.03f, door_w * 0.96f}, clay_col(0.66f));
        if (add_step) {
            box({center.x + hx + 0.24f, y0 + 0.11f, dz}, {0.56f, 0.22f, door_w + 0.34f}, clay_col(0.76f));
        }
    } else if (door_face == 2) { // Recessed opening on -X facade (Right Midground House!)
        const f32 dz = center.z + door_off;
        const f32 z_min = center.z - hz, z_max = center.z + hz;
        const f32 d0 = dz - door_w * 0.5f, d1 = dz + door_w * 0.5f;
        box({center.x + rec * 0.5f, y0 + hy, center.z}, {size.x - rec, size.y, size.z}, col);
        const f32 fx = center.x - hx + rec * 0.5f;
        box({fx, y0 + hy, (z_min + d0) * 0.5f}, {rec, size.y, d0 - z_min}, col);
        box({fx, y0 + hy, (d1 + z_max) * 0.5f}, {rec, size.y, z_max - d1}, col);
        box({fx, y0 + (door_h + size.y) * 0.5f, dz}, {rec, size.y - door_h, door_w}, col);
        box({center.x - hx + rec - 0.03f, y0 + door_h * 0.5f, dz}, {0.05f, door_h, door_w}, clay_col(0.58f));
    } else if (door_face == 3) { // Recessed doorway on +Z facade (Mid-Center House!)
        const f32 dx = center.x + door_off;
        const f32 x_min = center.x - hx, x_max = center.x + hx;
        const f32 d0 = dx - door_w * 0.5f, d1 = dx + door_w * 0.5f;
        box({center.x, y0 + hy, center.z - rec * 0.5f}, {size.x, size.y, size.z - rec}, col);
        const f32 fz = center.z + hz - rec * 0.5f;
        box({(x_min + d0) * 0.5f, y0 + hy, fz}, {d0 - x_min, size.y, rec}, col);
        box({(d1 + x_max) * 0.5f, y0 + hy, fz}, {x_max - d1, size.y, rec}, col);
        box({dx, y0 + (door_h + size.y) * 0.5f, fz}, {door_w, size.y - door_h, rec}, col);
        box({dx, y0 + door_h * 0.5f, center.z + hz - rec + 0.03f}, {door_w - 0.04f, door_h, 0.05f}, clay_col(0.68f));
    }

    // Crisp overhanging thin clay roof cornice slab
    box({center.x, y0 + size.y + 0.06f, center.z}, {size.x + 0.28f, 0.12f, size.z + 0.28f}, clay_col(0.77f));
    if (add_chimney) {
        box({center.x + 0.45f, y0 + size.y + 0.42f, center.z + 0.55f}, {0.55f, 0.62f, 0.55f}, clay_col(0.75f));
    }
}

// Low-poly angled clay grass tuft cluster tucked against walls & stones
inline void add_clay_grass_tuft(MeshData& dst, Vec3 base_pos, u32 seed, f32 scale = 1.0f) {
    const Vec4 gc = clay_col(0.68f, 0.4f);
    for (int i = 0; i < 11; ++i) {
        const u32 h = hash_u32(seed + static_cast<u32>(i * 41));
        const f32 ang = rand01(h) * 2.0f * math::kPi;
        const f32 dist = rand01(h ^ 0x11u) * 0.14f * scale;
        const f32 bh = (0.18f + 0.24f * rand01(h ^ 0x22u)) * scale;
        const f32 bw = (0.018f + 0.014f * rand01(h ^ 0x33u)) * scale;
        const f32 lean = (0.04f + 0.10f * rand01(h ^ 0x44u)) * scale;
        const Vec3 p0 = base_pos + Vec3{std::cos(ang) * dist, -0.01f, std::sin(ang) * dist};
        const Vec3 side{-std::sin(ang) * bw, 0.0f, std::cos(ang) * bw};
        const Vec3 tip = p0 + Vec3{std::cos(ang) * lean, bh, std::sin(ang) * lean};
        const Vec3 a = p0 - side, b = p0 + side;
        const Vec3 n = math::normalize(math::cross(b - a, tip - a));
        const u32 idx = static_cast<u32>(dst.vertices.size());
        dst.vertices.push_back({a, n, gc, {0, 0}, {0, 0, 0, 0}});
        dst.vertices.push_back({b, n, gc, {0, 0}, {0, 0, 0, 0}});
        dst.vertices.push_back({tip, n, gc, {0, 0}, {0, 0, 0, 0}});
        dst.indices.insert(dst.indices.end(), {idx, idx + 1, idx + 2});
    }
}

} // namespace detail

inline ClayScene build_clay_scene() {
    using namespace detail;
    ClayScene s;
    const Vec4 wall_col = clay_col(0.76f);
    const Vec4 prop_col = clay_col(0.74f);
    const Vec4 road_col = clay_col(0.73f, 0.2f);

    // ---------------------------------------------------------------- 1. Gently undulating clay road + recessed puddle basins
    {
        constexpr int kNx = 140, kNz = 150;
        constexpr f32 x0 = -36.0f, x1 = 36.0f, z0 = -38.0f, z1 = 18.0f;
        const f32 dx = (x1 - x0) / kNx;
        const f32 dz = (z1 - z0) / kNz;
        const u32 base = static_cast<u32>(s.opaque.vertices.size());
        for (int j = 0; j <= kNz; ++j) {
            const f32 z = z0 + static_cast<f32>(j) * dz;
            for (int i = 0; i <= kNx; ++i) {
                const f32 x = x0 + static_cast<f32>(i) * dx;
                const f32 y = ground_height(x, z);
                const f32 e = 0.14f;
                const Vec3 n = math::normalize(Vec3{ground_height(x - e, z) - ground_height(x + e, z), 2.0f * e,
                                                    ground_height(x, z - e) - ground_height(x, z + e)});
                s.opaque.vertices.push_back({{x, y, z}, n, road_col, {0, 0}, {0, 0, 0, 0}});
            }
        }
        for (int j = 0; j < kNz; ++j) {
            for (int i = 0; i < kNx; ++i) {
                const u32 a = base + static_cast<u32>(j * (kNx + 1) + i);
                const u32 b = a + 1;
                const u32 c = a + static_cast<u32>(kNx + 1);
                const u32 d = c + 1;
                s.opaque.indices.insert(s.opaque.indices.end(), {a, c, d, a, d, b});
            }
        }
    }

    // ---------------------------------------------------------------- 2. Three liquid-mirror rainwater puddles (Material::Water)
    for (const auto& p : kPuddles) {
        const Vec4 wcol = renderer::material_color(Material::Water, {0.03f, 0.03f, 0.03f});
        constexpr int kSegs = 40;
        const f32 wy = -0.018f;
        const u32 center_idx = static_cast<u32>(s.water.vertices.size());
        s.water.vertices.push_back({{p.cx, wy, p.cz}, {0, 1, 0}, wcol, {0.5f, 0.5f}, {1, 0, 0, 1}});
        const f32 cr = std::cos(p.rot), sr = std::sin(p.rot);
        for (int i = 0; i <= kSegs; ++i) {
            const f32 ang = (static_cast<f32>(i) / kSegs) * 2.0f * math::kPi;
            const f32 wobble = 1.0f + 0.14f * std::sin(ang * 2.0f + p.phase)
                                    + 0.10f * std::cos(ang * 3.0f - p.phase * 1.4f)
                                    + 0.05f * std::sin(ang * 5.0f + 0.8f);
            const f32 lx = std::cos(ang) * p.rx * wobble * 1.14f;
            const f32 lz = std::sin(ang) * p.rz * wobble * 1.14f;
            const f32 wx = p.cx + lx * cr + lz * sr;
            const f32 wz = p.cz - lx * sr + lz * cr;
            s.water.vertices.push_back({{wx, wy, wz}, {0, 1, 0}, wcol, {0.5f + 0.5f * std::cos(ang), 0.5f + 0.5f * std::sin(ang)}, {1, 0, 0, 1}});
        }
        for (int i = 1; i <= kSegs; ++i) {
            s.water.indices.insert(s.water.indices.end(), {center_idx, center_idx + static_cast<u32>(i + 1), center_idx + static_cast<u32>(i)});
        }
    }

    // ---------------------------------------------------------------- 3. Left Foreground House + Doorway + Doorstep + Hanging Lantern
    add_clay_house(s.opaque, {-4.45f, -0.04f, 3.65f}, {4.55f, 3.85f, 6.95f}, wall_col, 1, 1.35f, 1.12f, 2.38f, true, false);
    s.colliders.push_back({{-4.45f, 1.9f, 3.65f}, {2.28f, 1.95f, 3.48f}, 0.0f});

    // Hanging Architectural Clay Lantern mounted left of doorway!
    {
        const Vec3 mount{-2.16f, 2.66f, 6.02f};
        renderer::append(s.opaque, renderer::make_cube(1.0f), math::translate(mount) * math::scale({0.06f, 0.20f, 0.14f}), prop_col);
        renderer::append(s.opaque, renderer::make_cube(1.0f), math::translate(mount + Vec3{0.18f, 0.03f, 0.0f}) * math::scale({0.38f, 0.045f, 0.045f}), prop_col);
        renderer::append(s.opaque, renderer::make_cube(1.0f), math::translate(mount + Vec3{0.34f, -0.03f, 0.0f}) * math::scale({0.045f, 0.12f, 0.045f}), prop_col);
        // Arch bail handle + tapered conical cap + cylindrical bell body
        const Vec3 ltop = mount + Vec3{0.34f, -0.09f, 0.0f};
        add_tube_segment(s.opaque, ltop, ltop + Vec3{0.0f, -0.11f, -0.11f}, 0.015f, 0.015f, 8, prop_col);
        add_tube_segment(s.opaque, ltop, ltop + Vec3{0.0f, -0.11f,  0.11f}, 0.015f, 0.015f, 8, prop_col);
        renderer::append(s.opaque, renderer::make_cylinder(24, 0.135f, 0.06f, 0.08f),
                         math::translate(ltop + Vec3{0.0f, -0.22f, 0.0f}), prop_col);
        renderer::append(s.opaque, renderer::make_cylinder(24, 0.155f, 0.125f, 0.34f),
                         math::translate(ltop + Vec3{0.0f, -0.56f, 0.0f}), clay_col(0.77f));
        renderer::append(s.opaque, renderer::make_cylinder(24, 0.165f, 0.165f, 0.03f),
                         math::translate(ltop + Vec3{0.0f, -0.57f, 0.0f}), clay_col(0.72f));
    }

    // ---------------------------------------------------------------- 4. Left Foreground Beveled Block Wall + Solid 3D Thick Draped Clay Cloth!
    add_beveled_wall_z(s.opaque, -2.18f, 6.25f, 10.60f, 4, 0.62f, 0.42f, 0.84f, 0.038f, 101u);
    s.colliders.push_back({{-2.18f, 0.9f, 8.4f}, {0.35f, 0.9f, 2.2f}, 0.0f});

    // Solid 3D Extruded Draped Clay Cloth (2.2cm thick front+back+all 4 side rims hugging snugly against the beveled stones!)
    {
        const Vec4 cc = clay_col(0.75f);
        constexpr int kU = 16, kV = 22;
        const f32 zc = 7.42f, cw = 0.46f, thick = 0.024f;
        auto cloth_pt = [&](f32 tu, f32 tv, Vec3& out_p, Vec3& out_n) {
            f32 px, py;
            Vec3 base_n;
            if (tv < 0.26f) {
                const f32 s0 = tv / 0.26f;
                px = math::lerp(-2.46f, -1.87f, s0);
                py = 1.725f + 0.008f * (1.0f - s0 * s0);
                base_n = {0.0f, 1.0f, 0.0f};
            } else if (tv < 0.40f) {
                const f32 ang = ((tv - 0.26f) / 0.14f) * (math::kPi * 0.5f);
                px = -1.87f + std::sin(ang) * 0.038f;
                py = 1.687f + std::cos(ang) * 0.038f;
                base_n = {std::sin(ang), std::cos(ang), 0.0f};
            } else {
                const f32 s1 = (tv - 0.40f) / 0.60f;
                px = -1.832f + 0.014f * std::sin(s1 * math::kPi * 1.15f);
                py = math::lerp(1.687f, 0.86f, s1);
                base_n = {1.0f, 0.04f, 0.0f};
            }
            const f32 z = zc + (tu - 0.5f) * cw;
            const f32 fold_amp = 0.011f * std::clamp((tv - 0.20f) * 1.35f, 0.0f, 1.0f);
            const f32 fold = fold_amp * (0.65f * std::sin(tu * math::kPi * 2.0f + 0.3f) + 0.35f * std::cos(tu * math::kPi * 3.0f));
            const f32 dfold_dz = (fold_amp / cw) * (0.65f * math::kPi * 2.0f * std::cos(tu * math::kPi * 2.0f + 0.3f) -
                                                    0.35f * math::kPi * 3.0f * std::sin(tu * math::kPi * 3.0f));
            out_p = Vec3{px, py, z} + base_n * fold;
            out_n = math::normalize(base_n + Vec3{0.0f, 0.0f, -dfold_dz * 0.5f});
        };

        const u32 fbase = static_cast<u32>(s.opaque.vertices.size());
        for (int j = 0; j <= kV; ++j) {
            const f32 tv = static_cast<f32>(j) / kV;
            for (int i = 0; i <= kU; ++i) {
                const f32 tu = static_cast<f32>(i) / kU;
                Vec3 p, n;
                cloth_pt(tu, tv, p, n);
                s.opaque.vertices.push_back({p + n * thick, n, cc, {tu, tv}, {0, 0, 0, 0}});
            }
        }
        const u32 bbase = static_cast<u32>(s.opaque.vertices.size());
        for (int j = 0; j <= kV; ++j) {
            const f32 tv = static_cast<f32>(j) / kV;
            for (int i = 0; i <= kU; ++i) {
                const f32 tu = static_cast<f32>(i) / kU;
                Vec3 p, n;
                cloth_pt(tu, tv, p, n);
                s.opaque.vertices.push_back({p, -n, cc, {tu, tv}, {0, 0, 0, 0}});
            }
        }
        for (int j = 0; j < kV; ++j) {
            for (int i = 0; i < kU; ++i) {
                const u32 a = fbase + static_cast<u32>(j * (kU + 1) + i), b = a + 1, c = a + static_cast<u32>(kU + 1), d = c + 1;
                s.opaque.indices.insert(s.opaque.indices.end(), {a, b, d, a, d, c});
                const u32 ba = bbase + static_cast<u32>(j * (kU + 1) + i), bb = ba + 1, bc = ba + static_cast<u32>(kU + 1), bd = bc + 1;
                s.opaque.indices.insert(s.opaque.indices.end(), {ba, bd, bb, ba, bc, bd});
            }
        }
        // Solid side rims (left/right vertical edges + crisp bottom hem!) so cloth has real 3D physical thickness!
        auto stitch_rim = [&](int i0, int j0, int i1, int j1, Vec3 rim_n) {
            const Vec3 p0 = s.opaque.vertices[fbase + static_cast<usize>(j0 * (kU + 1) + i0)].position;
            const Vec3 p1 = s.opaque.vertices[fbase + static_cast<usize>(j1 * (kU + 1) + i1)].position;
            const Vec3 q0 = s.opaque.vertices[bbase + static_cast<usize>(j0 * (kU + 1) + i0)].position;
            const Vec3 q1 = s.opaque.vertices[bbase + static_cast<usize>(j1 * (kU + 1) + i1)].position;
            const u32 rb = static_cast<u32>(s.opaque.vertices.size());
            s.opaque.vertices.push_back({p0, rim_n, cc, {0, 0}, {0, 0, 0, 0}});
            s.opaque.vertices.push_back({p1, rim_n, cc, {0, 0}, {0, 0, 0, 0}});
            s.opaque.vertices.push_back({q1, rim_n, cc, {0, 0}, {0, 0, 0, 0}});
            s.opaque.vertices.push_back({q0, rim_n, cc, {0, 0}, {0, 0, 0, 0}});
            s.opaque.indices.insert(s.opaque.indices.end(), {rb, rb + 1, rb + 2, rb, rb + 2, rb + 3});
        };
        for (int j = 0; j < kV; ++j) {
            stitch_rim(0, j + 1, 0, j, {0, 0, -1});
            stitch_rim(kU, j, kU, j + 1, {0, 0, 1});
        }
        for (int i = 0; i < kU; ++i) {
            stitch_rim(i, kV, i + 1, kV, {0, -1, 0});
        }
    }

    // ---------------------------------------------------------------- 5. Left Foreground Leaning Low-Poly Faceted Tree (Casts Dappled Shadows on Left Wall!)
    {
        const Vec4 tr_col = clay_col(0.73f);
        add_tube_segment(s.opaque, {-2.55f, 1.35f, 7.95f}, {-1.95f, 2.75f, 7.25f}, 0.24f, 0.18f, 14, tr_col);
        add_tube_segment(s.opaque, {-1.95f, 2.75f, 7.25f}, {-1.15f, 3.65f, 6.45f}, 0.18f, 0.12f, 12, tr_col);
        add_tube_segment(s.opaque, {-1.92f, 2.82f, 7.15f}, {-2.15f, 3.72f, 7.72f}, 0.13f, 0.08f, 10, tr_col);
        // Chunky flat-shaded faceted clay canopy overhanging top-left + casting crisp polygonal leaf shadows onto left wall!
        const std::array<std::pair<Vec3, Vec3>, 14> clusters = {{
            {{-1.55f, 3.85f, 7.35f}, {0.50f, 0.46f, 0.48f}},
            {{-0.95f, 3.68f, 6.65f}, {0.48f, 0.44f, 0.46f}},
            {{-0.38f, 3.78f, 6.05f}, {0.52f, 0.46f, 0.50f}},
            {{-1.22f, 3.55f, 5.95f}, {0.42f, 0.38f, 0.42f}},
            {{-0.25f, 4.18f, 5.35f}, {0.54f, 0.48f, 0.52f}},
            {{-0.88f, 4.25f, 6.45f}, {0.52f, 0.48f, 0.50f}},
            {{-1.75f, 4.15f, 6.65f}, {0.48f, 0.44f, 0.46f}},
            {{-0.55f, 3.62f, 5.15f}, {0.40f, 0.36f, 0.40f}},
            {{-1.15f, 3.98f, 4.85f}, {0.44f, 0.40f, 0.44f}},
            {{ 0.25f, 3.92f, 5.55f}, {0.42f, 0.38f, 0.42f}},
            {{-0.05f, 3.70f, 4.55f}, {0.38f, 0.34f, 0.38f}},
            {{-0.45f, 4.30f, 3.95f}, {0.44f, 0.40f, 0.44f}},
            {{-1.35f, 3.80f, 3.75f}, {0.40f, 0.36f, 0.40f}},
            {{ 0.55f, 3.45f, 4.75f}, {0.36f, 0.34f, 0.36f}},
        }};
        for (usize i = 0; i < clusters.size(); ++i) {
            add_faceted_polyhedron(s.opaque, clusters[i].first, clusters[i].second, 401u + static_cast<u32>(i * 19), clay_col(0.75f), 0);
        }
    }

    // ---------------------------------------------------------------- 6. Mid-Left Beveled Wall + Second Sculptural Low-Poly Tree
    add_beveled_wall_z(s.opaque, -2.18f, -5.60f, 0.45f, 4, 0.56f, 0.42f, 0.80f, 0.036f, 202u);
    s.colliders.push_back({{-2.18f, 0.86f, -2.55f}, {0.32f, 0.86f, 3.05f}, 0.0f});

    // Second Faceted Tree rising cleanly behind the mid-left wall, silhouetted against the cobalt sky!
    {
        const Vec4 tr_col = clay_col(0.73f);
        add_tube_segment(s.opaque, {-2.58f, 1.35f, -1.65f}, {-2.35f, 2.55f, -1.55f}, 0.21f, 0.15f, 12, tr_col);
        add_tube_segment(s.opaque, {-2.35f, 2.55f, -1.55f}, {-1.85f, 3.15f, -1.35f}, 0.12f, 0.07f, 10, tr_col);
        add_tube_segment(s.opaque, {-2.35f, 2.55f, -1.55f}, {-2.68f, 3.25f, -1.85f}, 0.12f, 0.07f, 10, tr_col);
        const std::array<std::pair<Vec3, Vec3>, 9> clusters = {{
            {{-2.30f, 3.65f, -1.55f}, {0.62f, 0.58f, 0.60f}},
            {{-1.72f, 3.35f, -1.25f}, {0.56f, 0.52f, 0.54f}},
            {{-2.72f, 3.42f, -1.85f}, {0.54f, 0.50f, 0.52f}},
            {{-2.05f, 4.18f, -1.48f}, {0.52f, 0.48f, 0.50f}},
            {{-2.55f, 4.08f, -1.78f}, {0.48f, 0.46f, 0.48f}},
            {{-1.55f, 3.72f, -1.72f}, {0.48f, 0.46f, 0.48f}},
            {{-2.10f, 3.05f, -1.95f}, {0.44f, 0.42f, 0.44f}},
            {{-2.95f, 3.05f, -1.45f}, {0.44f, 0.40f, 0.44f}},
            {{-1.82f, 3.85f, -1.05f}, {0.44f, 0.42f, 0.44f}},
        }};
        for (usize i = 0; i < clusters.size(); ++i) {
            add_faceted_polyhedron(s.opaque, clusters[i].first, clusters[i].second, 701u + static_cast<u32>(i * 23), clay_col(0.75f), 0);
        }
    }

    // ---------------------------------------------------------------- 7. Mid-Center Clay House (Facing +Z with Doorway & Rooftop Chimney)
    add_clay_house(s.opaque, {-3.45f, -0.04f, -9.20f}, {4.45f, 3.45f, 5.45f}, wall_col, 3, 0.35f, 1.05f, 2.25f, false, true);
    s.colliders.push_back({{-3.45f, 1.7f, -9.20f}, {2.25f, 1.75f, 2.75f}, 0.0f});

    // ---------------------------------------------------------------- 8. Right Foreground Beveled Block Wall (3 courses front, 5 courses tucked behind!)
    add_beveled_wall_z(s.opaque, 2.42f, 4.65f, 10.60f, 3, 0.72f, 0.37f, 0.88f, 0.036f, 303u);
    add_beveled_wall_z(s.opaque, 3.25f, 1.85f, 4.85f, 5, 0.64f, 0.40f, 0.84f, 0.036f, 353u);
    // Perpendicular connecting block return on right side
    add_beveled_wall_x(s.opaque, 4.95f, 2.65f, 5.20f, 3, 0.66f, 0.37f, 0.86f, 0.036f, 379u);
    s.colliders.push_back({{2.42f, 0.6f, 7.6f}, {0.40f, 0.6f, 2.8f}, 0.0f});

    // ---------------------------------------------------------------- 9. Right Midground House (set cleanly back so golden sunbeam floods the doorway & road!)
    add_clay_house(s.opaque, {5.55f, -0.04f, -2.85f}, {4.65f, 3.50f, 6.80f}, wall_col, 2, -0.45f, 0.92f, 2.05f, false, false);
    s.colliders.push_back({{5.55f, 1.8f, -2.85f}, {2.35f, 1.85f, 3.40f}, 0.0f});

    // Leaning 7-Rung Architectural Clay Ladder against front-left corner of Right House!
    {
        const Vec3 b0{2.66f, -0.02f, -0.25f}, t0{3.22f, 2.52f, -0.25f};
        const Vec3 b1{2.66f, -0.02f,  0.32f}, t1{3.22f, 2.52f,  0.32f};
        add_tube_segment(s.opaque, b0, t0, 0.036f, 0.036f, 10, prop_col);
        add_tube_segment(s.opaque, b1, t1, 0.036f, 0.036f, 10, prop_col);
        for (int r = 1; r <= 7; ++r) {
            const f32 frac = static_cast<f32>(r) / 8.0f;
            add_tube_segment(s.opaque, math::lerp(b0, t0, frac), math::lerp(b1, t1, frac), 0.024f, 0.024f, 8, prop_col);
        }
    }

    // Stacked Clay Crates & Cylindrical Clay Barrel beside the ladder & right wall
    add_beveled_block(s.opaque, {2.44f, 0.36f, 0.88f}, {0.72f, 0.72f, 0.72f}, 0.028f, prop_col, 0.08f);
    add_beveled_block(s.opaque, {2.68f, 1.00f, 1.00f}, {0.56f, 0.56f, 0.56f}, 0.024f, clay_col(0.76f), -0.14f);
    renderer::append(s.opaque, renderer::make_cylinder(24, 0.34f, 0.34f, 0.92f),
                     math::translate({2.54f, -0.02f, 2.45f}), prop_col);
    renderer::append(s.opaque, renderer::make_cylinder(24, 0.36f, 0.36f, 0.06f),
                     math::translate({2.54f, 0.86f, 2.45f}), clay_col(0.77f));
    // Small barrel on left side near mid-center house
    renderer::append(s.opaque, renderer::make_cylinder(20, 0.24f, 0.24f, 0.68f),
                     math::translate({-2.08f, -0.02f, -5.85f}), prop_col);

    // Tall Clay Flagpole + Billowing 3D Clay Flag catching golden sunset light against the clouds!
    {
        const Vec3 pole_base{4.35f, 3.45f, -0.45f};
        add_tube_segment(s.opaque, {pole_base.x, 0.0f, pole_base.z}, {pole_base.x, 5.15f, pole_base.z}, 0.042f, 0.034f, 12, prop_col);
        renderer::append(s.opaque, renderer::make_icosphere(1),
                         math::translate({pole_base.x, 5.18f, pole_base.z}) * math::scale({0.07f, 0.07f, 0.07f}), clay_col(0.78f));

        // 3D sculpted wavy clay flag cloth fluttering gracefully toward +X/-Z
        constexpr int kFu = 16, kFv = 9;
        const f32 fw = 1.32f, fh = 0.78f;
        const Vec4 fcol = clay_col(0.78f);
        const u32 base = static_cast<u32>(s.two_sided.vertices.size());
        for (int j = 0; j <= kFv; ++j) {
            const f32 tv = static_cast<f32>(j) / kFv;
            for (int i = 0; i <= kFu; ++i) {
                const f32 tu = static_cast<f32>(i) / kFu;
                const f32 droop = -0.14f * tu * tu;
                const f32 wave = std::sin(tu * 5.2f + tv * 1.4f) * 0.11f * tu;
                const Vec3 pos{pole_base.x + 0.04f + tu * fw * 0.94f,
                               5.06f - tv * fh + droop,
                               pole_base.z - tu * fw * 0.22f + wave};
                const f32 dw = std::cos(tu * 5.2f + tv * 1.4f) * 0.55f * tu;
                const Vec3 nrm = math::normalize(Vec3{-dw * 0.3f, 0.15f, 1.0f});
                s.two_sided.vertices.push_back({pos, nrm, fcol, {tu, tv}, {0, 0, 0, 0}});
            }
        }
        for (int j = 0; j < kFv; ++j) {
            for (int i = 0; i < kFu; ++i) {
                const u32 a = base + static_cast<u32>(j * (kFu + 1) + i);
                const u32 b = a + 1;
                const u32 c = a + static_cast<u32>(kFu + 1);
                const u32 d = c + 1;
                s.two_sided.indices.insert(s.two_sided.indices.end(), {a, b, d, a, d, c});
            }
        }
    }

    // ---------------------------------------------------------------- 10. Distant Courtyard Tensioned Clay Canopy & Low Block Perimeter
    {
        const Vec3 cc{0.65f, 0.0f, -15.8f};
        const f32 aw = 2.65f, ad = 2.05f, ah = 2.45f;
        const Vec3 p00{cc.x - aw * 0.5f, 0.0f, cc.z - ad * 0.5f};
        const Vec3 p10{cc.x + aw * 0.5f, 0.0f, cc.z - ad * 0.5f};
        const Vec3 p01{cc.x - aw * 0.5f, 0.0f, cc.z + ad * 0.5f};
        const Vec3 p11{cc.x + aw * 0.5f, 0.0f, cc.z + ad * 0.5f};
        for (const Vec3& p : {p00, p10, p01, p11}) {
            add_tube_segment(s.opaque, p, p + Vec3{0, ah, 0}, 0.032f, 0.028f, 8, prop_col);
        }
        // Subdivided catenary sagging clay canopy cloth!
        constexpr int kAu = 14, kAv = 12;
        const Vec4 acol = clay_col(0.79f);
        const u32 base = static_cast<u32>(s.two_sided.vertices.size());
        for (int j = 0; j <= kAv; ++j) {
            const f32 tv = static_cast<f32>(j) / kAv;
            const f32 sv = 4.0f * tv * (1.0f - tv);
            for (int i = 0; i <= kAu; ++i) {
                const f32 tu = static_cast<f32>(i) / kAu;
                const f32 su = 4.0f * tu * (1.0f - tu);
                const f32 sag = -0.36f * (0.65f * su + 0.35f * sv * su);
                const Vec3 pos{cc.x + (tu - 0.5f) * aw, ah - 0.02f + sag, cc.z + (tv - 0.5f) * ad};
                const Vec3 nrm = math::normalize(Vec3{(tu - 0.5f) * 0.55f, 1.0f, (tv - 0.5f) * 0.40f});
                s.two_sided.vertices.push_back({pos, nrm, acol, {tu, tv}, {0, 0, 0, 0}});
            }
        }
        for (int j = 0; j < kAv; ++j) {
            for (int i = 0; i < kAu; ++i) {
                const u32 a = base + static_cast<u32>(j * (kAu + 1) + i);
                const u32 b = a + 1;
                const u32 c = a + static_cast<u32>(kAu + 1);
                const u32 d = c + 1;
                s.two_sided.indices.insert(s.two_sided.indices.end(), {a, c, d, a, d, b});
            }
        }
        // Distant low stacked clay block perimeter & crates framing the open sunset horizon!
        add_beveled_wall_x(s.opaque, -20.5f, -14.0f, 14.0f, 3, 0.56f, 0.42f, 0.92f, 0.06f, 909u);
        add_beveled_block(s.opaque, {-1.8f, 0.42f, -18.2f}, {1.6f, 0.84f, 0.85f}, 0.05f, prop_col);
    }

    // ---------------------------------------------------------------- 11. Low-Poly Faceted Clay Rocks, Pebbles & Grass Tufts!
    const std::array<std::pair<Vec3, Vec3>, 10> rocks = {{
        {{ 1.48f, 0.16f, 4.95f}, {0.42f, 0.24f, 0.34f}}, // Hero angular boulder right-midground!
        {{ 1.46f, 0.11f, 6.95f}, {0.26f, 0.16f, 0.24f}}, // Right foreground faceted stone
        {{ 0.24f, 0.05f, 5.35f}, {0.13f, 0.08f, 0.11f}}, // Center lane pebble
        {{-1.42f, 0.07f, 6.25f}, {0.16f, 0.10f, 0.14f}}, // Near left doorstep
        {{-0.72f, 0.09f, 1.55f}, {0.19f, 0.11f, 0.16f}}, // Mid-left lane stone
        {{-0.15f, 0.06f, 0.15f}, {0.14f, 0.08f, 0.12f}}, // Mid-lane pebble
        {{-1.25f, 0.05f, 7.15f}, {0.09f, 0.06f, 0.09f}}, // Foreground small pebble
        {{-1.48f, 0.04f, 6.90f}, {0.07f, 0.05f, 0.08f}}, // Foreground tiny pebble
        {{ 0.95f, 0.06f, 1.50f}, {0.12f, 0.07f, 0.11f}}, // Near puddle 2
        {{-1.40f, 0.06f, -3.1f}, {0.15f, 0.09f, 0.14f}}, // Far left wall stone
    }};
    for (usize i = 0; i < rocks.size(); ++i) {
        const Vec3 p = rocks[i].first;
        const f32 gy = ground_height(p.x, p.z);
        add_faceted_polyhedron(s.opaque, {p.x, gy + p.y * 0.72f, p.z}, rocks[i].second,
                               1201u + static_cast<u32>(i * 37), clay_col(0.73f), 0);
    }

    // Crisp low-poly clay grass tufts tucked naturally along the left wall, doorstep, and mid-left wall!
    const std::array<std::pair<Vec3, f32>, 9> grasses = {{
        {{-1.68f, 0.0f, 6.45f}, 1.25f},
        {{-1.72f, 0.0f, 6.85f}, 1.05f},
        {{-1.85f, 0.0f, 3.25f}, 1.10f},
        {{-1.88f, 0.0f, 2.85f}, 0.85f},
        {{-1.80f, 0.0f, -0.45f}, 0.95f},
        {{-1.82f, 0.0f, -1.65f}, 0.90f},
        {{-1.80f, 0.0f, -3.30f}, 0.80f},
        {{ 1.75f, 0.0f, 6.20f}, 0.85f},
        {{ 1.78f, 0.0f, 3.90f}, 0.80f},
    }};
    for (usize i = 0; i < grasses.size(); ++i) {
        const Vec3 gp = grasses[i].first;
        add_clay_grass_tuft(s.two_sided, {gp.x, ground_height(gp.x, gp.z), gp.z}, 1501u + static_cast<u32>(i * 53), grasses[i].second);
    }

    return s;
}

} // namespace nova::clay
