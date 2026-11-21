#pragma once

// Skeletal animation for glTF models: clip sampling, cross-fade blending, node world matrices
// and CPU skinning.

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/model.hpp"

#include <span>
#include <vector>

namespace nova::renderer {

struct NodeTransform {
    math::Vec3 translation{0.0f, 0.0f, 0.0f};
    math::Quat rotation;
    math::Vec3 scale{1.0f, 1.0f, 1.0f};
};

// Rest pose: every node's TRS from the file.
[[nodiscard]] std::vector<NodeTransform> rest_pose(const ModelData& model);

// Overwrites the nodes animated by `clip` with their values at `time` (clamped to the clip).
void sample_clip(const AnimationData& clip, f32 time, std::span<NodeTransform> pose);

// pose = lerp(pose, other, weight): translation/scale linear, rotation nlerp.
void blend_pose(std::span<NodeTransform> pose, std::span<const NodeTransform> other, f32 weight);

// Model-space matrix of every node (parents are resolved; order of nodes does not matter).
void node_world_matrices(const ModelData& model, std::span<const NodeTransform> pose, std::vector<math::Mat4>& world);

// Linear-blend skinning of a skinned primitive into model space (positions, normals, tangents;
// uv/colour copied). `world` comes from node_world_matrices. Non-skinned primitives are copied
// with their node transform applied.
void skin_primitive(const ModelData& model, const ModelPrimitive& prim, std::span<const math::Mat4> world,
                    std::vector<Vertex>& out);

// Plays clips with cross-fades. Each play() pushes a layer that fades in over `fade` seconds on
// top of the current result; fully faded-in layers drop everything beneath them.
class Animator {
public:
    explicit Animator(const ModelData& model);

    // `start` is the clip time to begin at; negative speeds are not supported.
    void play(i32 clip, f32 fade = 0.15f, bool loop = true, f32 speed = 1.0f, f32 start = 0.0f);
    void set_speed(f32 speed) noexcept; // top layer
    void update(f32 dt);

    [[nodiscard]] i32 current_clip() const noexcept { return layers_.empty() ? -1 : layers_.back().clip; }
    [[nodiscard]] f32 current_time() const noexcept { return layers_.empty() ? 0.0f : layers_.back().time; }
    // True once a non-looping top layer has reached its end.
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] const std::vector<math::Mat4>& world() const noexcept { return world_; }
    [[nodiscard]] const ModelData& model() const noexcept { return model_; }

private:
    struct Layer {
        i32 clip = -1;
        f32 time = 0.0f;
        f32 speed = 1.0f;
        bool loop = true;
        f32 weight = 1.0f;
        f32 fade_rate = 0.0f; // weight per second
    };
    const ModelData& model_;
    std::vector<NodeTransform> rest_;
    std::vector<NodeTransform> pose_;
    std::vector<NodeTransform> scratch_;
    std::vector<math::Mat4> world_;
    std::vector<Layer> layers_;
};

} // namespace nova::renderer
