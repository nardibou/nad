#pragma once

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/rhi/rhi.hpp"

#include <span>
#include <vector>

namespace nova::renderer {

struct Camera {
    math::Vec3 position{0, 2, 5};
    math::Vec3 target{0, 0, 0};
    math::Vec3 up{0, 1, 0};
    f32 fov_y = math::radians(60.0f);
    f32 z_near = 0.1f;
    f32 z_far = 500.0f;

    [[nodiscard]] math::Mat4 view() const { return math::look_at(position, target, up); }
    [[nodiscard]] math::Mat4 projection(f32 aspect) const { return math::perspective(fov_y, aspect, z_near, z_far); }
};

struct DirectionalLight {
    math::Vec3 direction{0.4f, 1.0f, 0.3f}; // points towards the light
    f32 ambient = 0.15f;
};

struct MeshId {
    u32 id = 0; // 0 is invalid
};

struct DrawItem {
    MeshId mesh;
    math::Mat4 transform = math::Mat4::identity();
    math::Vec3 color{0.8f, 0.8f, 0.8f}; // linear albedo
};

struct RenderStats {
    u32 draw_calls = 0;
    u32 triangles = 0;
};

// Baseline forward renderer: one pipeline, one draw per item. Culling, instancing and the
// render graph replace this loop in the Smart Rendering phase, measured against this baseline.
class Renderer {
public:
    explicit Renderer(rhi::Device& device);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return pipeline_.valid(); }
    [[nodiscard]] MeshId upload(const MeshData& mesh);

    // Records, submits and presents one frame. Returns false if the frame was skipped.
    bool render_frame(const Camera& camera, const DirectionalLight& light, std::span<const DrawItem> items,
                      const math::Vec3& clear_color);
    [[nodiscard]] const RenderStats& stats() const noexcept { return stats_; }

private:
    struct GpuMesh {
        rhi::BufferHandle vertices;
        rhi::BufferHandle indices;
        u32 index_count = 0;
    };

    rhi::Device& device_;
    rhi::ShaderHandle shader_vs_;
    rhi::ShaderHandle shader_fs_;
    rhi::PipelineHandle pipeline_;
    std::vector<GpuMesh> meshes_;
    RenderStats stats_;
};

} // namespace nova::renderer
