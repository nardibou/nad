#pragma once

// The procedural mountain village shared by the village viewer and the Guerrilla demo game.

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/world_renderer.hpp"

#include <vector>

namespace nova::village {

// Solid obstacle: a box rotated about +Y. `center` is the middle of the box (not its base).
struct Collider {
    math::Vec3 center;
    math::Vec3 half_extents; // x across, y up, z along (before yaw)
    f32 yaw = 0.0f;
};

struct Scene {
    renderer::MeshData terrain;
    renderer::MeshData opaque;       // buildings, plaster, stone plinths, metal
    renderer::MeshData foliage;      // tree crowns
    renderer::MeshData two_sided;    // grass, flag cloth, draped rug
    renderer::MeshData water;        // puddles (no shadows)
    renderer::MeshData stone_walls;  // 3D displaced PBR stone walls (wall_*.png)
    renderer::MeshData dirt_road;    // 3D displaced PBR dirt & gravel street (dirt_*.png)
    renderer::MeshData mud_patches;  // 3D displaced wet mud & bootprints (mud_*.png)
    renderer::MeshData forest_floor; // 3D displaced twigs, leaves & moss under olive trees (forest_*.png)
    renderer::MeshData wood_props;   // doors, lintels, roof beams, ladder (wood_*.png)
    renderer::MeshData tree_bark;    // gnarled olive tree trunks & branches (bark_*.png)
    std::vector<Collider> colliders;
};

struct VillageMaterials {
    renderer::MaterialId stone_wall;
    renderer::MaterialId dirt_road;
    renderer::MaterialId mud_patches;
    renderer::MaterialId forest_floor;
    renderer::MaterialId wood_props;
    renderer::MaterialId tree_bark;
};

// Terrain height at world (x, z); the village street sits at y ~ 0.
[[nodiscard]] f32 terrain_height(f32 x, f32 z);

// Deterministic: the same scene every call.
[[nodiscard]] Scene build_scene();

// Uploads all 6 PBR PNG material sets (Stone Wall, Dirt Road, Wet Mud, Forest Floor, Wood Planks, Tree Bark).
[[nodiscard]] VillageMaterials upload_village_materials(renderer::WorldRenderer& world);

} // namespace nova::village
