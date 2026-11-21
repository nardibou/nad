#pragma once

#include "nova/math/math.hpp"

#include <vector>

namespace nova::renderer {

struct Vertex {
    math::Vec3 position;
    math::Vec3 normal;
    // rgb = linear albedo, a = material id (see WorldRenderer / world.slang). Ignored by Renderer.
    math::Vec4 color{1.0f, 1.0f, 1.0f, 0.0f};
    math::Vec2 uv;      // TEXCOORD_0, glTF convention (v = 0 at the top of the image)
    math::Vec4 tangent; // xyz + handedness in w; zero when the mesh has no tangent frame
};

// CPU-side mesh. Triangles are counter-clockwise when seen from outside.
struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
};

// Axis-aligned cube centred at the origin with edge length `size` (24 vertices, flat normals).
[[nodiscard]] MeshData make_cube(f32 size = 1.0f);
// XZ plane centred at the origin facing +Y.
[[nodiscard]] MeshData make_plane(f32 size = 1.0f);
// Unit-radius icosphere with smooth normals; `subdivisions` 0..4.
[[nodiscard]] MeshData make_icosphere(u32 subdivisions = 1);
// Capped cylinder/cone along +Y from y=0 to y=height.
[[nodiscard]] MeshData make_cylinder(u32 segments, f32 radius_bottom, f32 radius_top, f32 height);

// Appends `src` transformed by `transform` (normals use the inverse transpose) with every vertex
// colour set to `color`.
void append(MeshData& dst, const MeshData& src, const math::Mat4& transform, const math::Vec4& color);
// Same, keeping the source vertex colours.
void append(MeshData& dst, const MeshData& src, const math::Mat4& transform);

} // namespace nova::renderer
