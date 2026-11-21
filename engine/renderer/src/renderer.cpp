#include "nova/renderer/renderer.hpp"

#include "nova/core/log.hpp"

#include "forward_spv.hpp"

#include <algorithm>
#include <cstddef>

namespace nova::renderer {
namespace {

struct PushConstants {
    math::Mat4 mvp;
    f32 normal_cols[3][4];
    f32 color[4];
};
static_assert(sizeof(PushConstants) == 128, "must match forward.slang and fit the 128-byte minimum");

} // namespace

Renderer::Renderer(rhi::Device& device) : device_(device) {
    shader_vs_ = device_.create_shader({forward_spv, "vsMain", rhi::ShaderStage::Vertex});
    shader_fs_ = device_.create_shader({forward_spv, "fsMain", rhi::ShaderStage::Fragment});
    if (!shader_vs_.valid() || !shader_fs_.valid()) {
        NOVA_ERROR("Renderer", "failed to create shaders");
        return;
    }
    rhi::GraphicsPipelineDesc desc;
    desc.vertex_shader = shader_vs_;
    desc.fragment_shader = shader_fs_;
    desc.vertex_stride = sizeof(Vertex);
    desc.attributes = {
        {0, rhi::VertexFormat::Float3, static_cast<u32>(offsetof(Vertex, position))},
        {1, rhi::VertexFormat::Float3, static_cast<u32>(offsetof(Vertex, normal))},
    };
    desc.push_constant_size = sizeof(PushConstants);
    pipeline_ = device_.create_graphics_pipeline(desc);
    if (!pipeline_.valid()) {
        NOVA_ERROR("Renderer", "failed to create forward pipeline");
    }
}

Renderer::~Renderer() {
    for (const GpuMesh& m : meshes_) {
        device_.destroy(m.vertices);
        device_.destroy(m.indices);
    }
    device_.destroy(pipeline_);
    device_.destroy(shader_vs_);
    device_.destroy(shader_fs_);
}

MeshId Renderer::upload(const MeshData& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        return {};
    }
    GpuMesh gpu;
    gpu.vertices = device_.create_buffer(
        {mesh.vertices.size() * sizeof(Vertex), rhi::BufferUsage::Vertex, mesh.vertices.data()});
    gpu.indices =
        device_.create_buffer({mesh.indices.size() * sizeof(u32), rhi::BufferUsage::Index, mesh.indices.data()});
    gpu.index_count = static_cast<u32>(mesh.indices.size());
    if (!gpu.vertices.valid() || !gpu.indices.valid()) {
        device_.destroy(gpu.vertices);
        device_.destroy(gpu.indices);
        return {};
    }
    meshes_.push_back(gpu);
    return {static_cast<u32>(meshes_.size())};
}

bool Renderer::render_frame(const Camera& camera, const DirectionalLight& light, std::span<const DrawItem> items,
                            const math::Vec3& clear_color) {
    rhi::CommandList* cmd = device_.begin_frame();
    if (!cmd) {
        return false;
    }
    const rhi::Extent extent = device_.back_buffer_extent();
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(std::max(extent.height, 1u));
    const math::Mat4 view_proj = camera.projection(aspect) * camera.view();
    const math::Vec3 l = math::normalize(light.direction);

    rhi::ClearValues clear;
    clear.color[0] = clear_color.x;
    clear.color[1] = clear_color.y;
    clear.color[2] = clear_color.z;
    stats_ = {};

    cmd->begin_render_pass(clear);
    if (valid()) {
        cmd->bind_pipeline(pipeline_);
        for (const DrawItem& item : items) {
            if (item.mesh.id == 0 || item.mesh.id > meshes_.size()) {
                continue;
            }
            const GpuMesh& mesh = meshes_[item.mesh.id - 1];
            PushConstants pc{};
            pc.mvp = view_proj * item.transform;
            const f32 light_dir[3] = {l.x, l.y, l.z};
            for (int c = 0; c < 3; ++c) {
                for (int r = 0; r < 3; ++r) {
                    pc.normal_cols[c][r] = item.transform.m[c][r];
                }
                pc.normal_cols[c][3] = light_dir[c];
            }
            pc.color[0] = item.color.x;
            pc.color[1] = item.color.y;
            pc.color[2] = item.color.z;
            pc.color[3] = light.ambient;

            cmd->push_constants(&pc, sizeof(pc));
            cmd->bind_vertex_buffer(mesh.vertices);
            cmd->bind_index_buffer(mesh.indices);
            cmd->draw_indexed(mesh.index_count);
            ++stats_.draw_calls;
            stats_.triangles += mesh.index_count / 3;
        }
    }
    cmd->end_render_pass();
    device_.end_frame();
    return true;
}

} // namespace nova::renderer
