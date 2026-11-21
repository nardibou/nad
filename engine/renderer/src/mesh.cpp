#include "nova/renderer/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace nova::renderer {

using math::Vec3;

MeshData make_cube(f32 size) {
    const f32 h = size * 0.5f;
    struct Face {
        Vec3 normal, u, v; // u x v == normal, so (0,1,2),(0,2,3) is CCW from outside
    };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},  {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},  {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},   {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
    };
    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);
    for (const Face& f : faces) {
        const u32 base = static_cast<u32>(mesh.vertices.size());
        const Vec3 c = f.normal * h;
        mesh.vertices.push_back({c - f.u * h - f.v * h, f.normal});
        mesh.vertices.push_back({c + f.u * h - f.v * h, f.normal});
        mesh.vertices.push_back({c + f.u * h + f.v * h, f.normal});
        mesh.vertices.push_back({c - f.u * h + f.v * h, f.normal});
        for (u32 i : {0u, 1u, 2u, 0u, 2u, 3u}) {
            mesh.indices.push_back(base + i);
        }
    }
    return mesh;
}

MeshData make_plane(f32 size) {
    const f32 h = size * 0.5f;
    const Vec3 n{0, 1, 0};
    MeshData mesh;
    mesh.vertices = {{{-h, 0, h}, n}, {{h, 0, h}, n}, {{h, 0, -h}, n}, {{-h, 0, -h}, n}};
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
}

MeshData make_icosphere(u32 subdivisions) {
    const f32 t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<Vec3> pos = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                             {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    std::vector<u32> idx = {0, 11, 5, 0, 5,  1,  0,  1,  7,  0,  7, 10, 0, 10, 11, 1, 5, 9, 5, 11,
                            4, 11, 10, 2, 10, 7,  6,  7,  1,  8,  3, 9,  4,  3,  4, 2, 3, 2, 6,  3,
                            6, 8,  3,  8, 9,  4,  9,  5,  2,  4, 11, 6,  2,  10, 8, 6, 7, 9, 8, 1};
    for (Vec3& p : pos) {
        p = normalize(p);
    }
    for (u32 s = 0; s < std::min(subdivisions, 4u); ++s) {
        std::map<std::pair<u32, u32>, u32> midpoints;
        auto midpoint = [&](u32 a, u32 b) {
            const auto key = std::minmax(a, b);
            if (const auto it = midpoints.find(key); it != midpoints.end()) {
                return it->second;
            }
            pos.push_back(normalize((pos[a] + pos[b]) * 0.5f));
            const u32 id = static_cast<u32>(pos.size() - 1);
            midpoints.emplace(key, id);
            return id;
        };
        std::vector<u32> next;
        next.reserve(idx.size() * 4);
        for (usize i = 0; i < idx.size(); i += 3) {
            const u32 a = idx[i], b = idx[i + 1], c = idx[i + 2];
            const u32 ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        idx = std::move(next);
    }
    MeshData mesh;
    mesh.vertices.reserve(pos.size());
    for (const Vec3& p : pos) {
        mesh.vertices.push_back({p, p});
    }
    mesh.indices = std::move(idx);
    return mesh;
}

MeshData make_cylinder(u32 segments, f32 radius_bottom, f32 radius_top, f32 height) {
    segments = std::max(segments, 3u);
    MeshData mesh;
    // Side normal tilts with the cone slope.
    const f32 slope = (radius_bottom - radius_top) / height;
    for (u32 i = 0; i <= segments; ++i) {
        const f32 a = 2.0f * math::kPi * static_cast<f32>(i) / static_cast<f32>(segments);
        const f32 c = std::cos(a), s = std::sin(a);
        const Vec3 n = normalize(Vec3{c, slope, -s});
        mesh.vertices.push_back({{c * radius_bottom, 0, -s * radius_bottom}, n});
        mesh.vertices.push_back({{c * radius_top, height, -s * radius_top}, n});
    }
    for (u32 i = 0; i < segments; ++i) {
        const u32 b0 = i * 2, t0 = b0 + 1, b1 = b0 + 2, t1 = b0 + 3;
        mesh.indices.insert(mesh.indices.end(), {b0, b1, t1, b0, t1, t0});
    }
    for (int cap = 0; cap < 2; ++cap) {
        const f32 y = cap ? height : 0.0f;
        const f32 r = cap ? radius_top : radius_bottom;
        const Vec3 n{0, cap ? 1.0f : -1.0f, 0};
        const u32 centre = static_cast<u32>(mesh.vertices.size());
        mesh.vertices.push_back({{0, y, 0}, n});
        for (u32 i = 0; i <= segments; ++i) {
            const f32 a = 2.0f * math::kPi * static_cast<f32>(i) / static_cast<f32>(segments);
            mesh.vertices.push_back({{std::cos(a) * r, y, -std::sin(a) * r}, n});
        }
        for (u32 i = 0; i < segments; ++i) {
            const u32 a = centre + 1 + i, b = a + 1;
            if (cap) {
                mesh.indices.insert(mesh.indices.end(), {centre, a, b});
            } else {
                mesh.indices.insert(mesh.indices.end(), {centre, b, a});
            }
        }
    }
    return mesh;
}

void append(MeshData& dst, const MeshData& src, const math::Mat4& transform) {
    const u32 base = static_cast<u32>(dst.vertices.size());
    dst.vertices.reserve(dst.vertices.size() + src.vertices.size());
    for (const Vertex& v : src.vertices) {
        dst.vertices.push_back(
            {math::transform_point(transform, v.position), math::transform_normal(transform, v.normal), v.color});
    }
    dst.indices.reserve(dst.indices.size() + src.indices.size());
    for (u32 i : src.indices) {
        dst.indices.push_back(base + i);
    }
}

void append(MeshData& dst, const MeshData& src, const math::Mat4& transform, const math::Vec4& color) {
    const usize first = dst.vertices.size();
    append(dst, src, transform);
    for (usize i = first; i < dst.vertices.size(); ++i) {
        dst.vertices[i].color = color;
    }
}

} // namespace nova::renderer
