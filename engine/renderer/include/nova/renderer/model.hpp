#pragma once

// CPU-side 3D asset data (decoded, renderer-agnostic) and the glTF 2.0 importer.

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nova::renderer {

// Tightly packed RGBA8 pixels, top row first.
struct ImageData {
    u32 width = 0;
    u32 height = 0;
    std::vector<u8> rgba;
};

enum class AlphaMode : u8 { Opaque, Mask, Blend };

// glTF metallic-roughness material. Image indices refer to ModelData::images; -1 = none.
struct MaterialData {
    math::Vec4 base_color_factor{1.0f, 1.0f, 1.0f, 1.0f};
    f32 metallic_factor = 1.0f;
    f32 roughness_factor = 1.0f;
    math::Vec3 emissive_factor{0.0f, 0.0f, 0.0f};
    f32 normal_scale = 1.0f;
    f32 occlusion_strength = 1.0f;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    f32 alpha_cutoff = 0.5f;
    bool double_sided = false;
    i32 base_color_image = -1;         // sRGB
    i32 metallic_roughness_image = -1; // linear: G = roughness, B = metallic
    i32 normal_image = -1;             // linear, tangent space
    i32 occlusion_image = -1;          // linear: R
    i32 emissive_image = -1;           // sRGB
};

// One glTF primitive. Vertices are in mesh space; `transform` is the owning node's world matrix.
// Every vertex has a unit normal, a uv (0,0 if absent) and a tangent (generated when absent and
// the primitive has uvs). Vertex colour rgb = COLOR_0 (or white), a = 0.
// Skinned primitives (skin >= 0) ignore `transform`: positions come from the skin's joints
// (see animation.hpp), with up to four influences per vertex in `joints` / `weights`.
struct ModelPrimitive {
    MeshData mesh;
    i32 material = -1; // index into ModelData::materials; -1 = default material
    math::Mat4 transform = math::Mat4::identity();
    i32 node = -1;     // owning node
    i32 skin = -1;     // index into ModelData::skins
    std::vector<std::array<u16, 4>> joints; // skin-joint indices (into SkinData::joints)
    std::vector<math::Vec4> weights;         // normalized to sum 1
};

// Scene-graph node in its rest (non-animated) state. Matrix-only nodes are decomposed to TRS.
struct NodeData {
    std::string name;
    i32 parent = -1;
    math::Vec3 translation{0.0f, 0.0f, 0.0f};
    math::Quat rotation;
    math::Vec3 scale{1.0f, 1.0f, 1.0f};
};

struct SkinData {
    std::string name;
    std::vector<i32> joints;             // node indices
    std::vector<math::Mat4> inverse_bind; // one per joint
};

enum class AnimPath : u8 { Translation, Rotation, Scale };
enum class AnimInterpolation : u8 { Linear, Step, CubicSpline };

// Keyframes for one node property. Values are xyz (+w for rotations); for CubicSpline each key
// stores three values: in-tangent, value, out-tangent.
struct AnimChannel {
    i32 node = -1;
    AnimPath path = AnimPath::Translation;
    AnimInterpolation interpolation = AnimInterpolation::Linear;
    std::vector<f32> times;
    std::vector<math::Vec4> values;
};

struct AnimationData {
    std::string name;
    f32 duration = 0.0f; // last key time over all channels
    std::vector<AnimChannel> channels;
};

struct ModelData {
    std::vector<ImageData> images;
    std::vector<MaterialData> materials;
    std::vector<ModelPrimitive> primitives;
    std::vector<NodeData> nodes;
    std::vector<SkinData> skins;
    std::vector<AnimationData> animations;
    // Axis-aligned bounds of all primitives after their node transforms (rest pose).
    math::Vec3 bounds_min{0.0f, 0.0f, 0.0f};
    math::Vec3 bounds_max{0.0f, 0.0f, 0.0f};

    [[nodiscard]] i32 find_node(std::string_view name) const noexcept;
    [[nodiscard]] i32 find_animation(std::string_view name) const noexcept; // exact or suffix match
};

// Loads a .gltf (external or data-URI buffers/images) or .glb file. Supports triangle primitives,
// indexed or not, any accessor component type, and PNG/JPEG images. On failure returns nullopt
// and, if `error` is non-null, a human-readable reason.
[[nodiscard]] std::optional<ModelData> load_gltf(const std::filesystem::path& path, std::string* error = nullptr);

} // namespace nova::renderer
