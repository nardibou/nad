#include "nova/renderer/world_renderer.hpp"

#include "nova/core/log.hpp"

#include "world_spv.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace nova::renderer {
namespace {

using math::Mat4;
using math::Vec3;

// Must match FrameData in world.slang (std140: matrices then float4s).
struct FrameData {
    Mat4 view_proj;
    Mat4 light_view_proj;
    Mat4 height_view_proj;
    f32 camera_pos[4];
    f32 camera_right[4];
    f32 camera_up[4];
    f32 camera_forward[4];
    f32 sun_dir[4];
    f32 sun_color[4];
    f32 sky_zenith[4];
    f32 sky_horizon[4];
    f32 ground_color[4];
    f32 camera_clip[4];
    f32 height_params[4];
    f32 post_params[4];
    f32 screen[4];
    f32 point_light[4]; // xyz position, w range
    f32 point_color[4]; // rgb radiance, w crosshair size (px)
    f32 output_size[4]; // back-buffer pixels
};
static_assert(sizeof(FrameData) == 192 + 16 * 16);

void set4(f32 (&dst)[4], const Vec3& v, f32 w) {
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = w;
}

enum ShaderSlot {
    kSkyVS, kSkyFS, kShadowVS, kShadowFS, kMainVS, kMainFS,
    kPostVS, kPostFS, kCompositeFS, kBloomDownFS, kBloomBlurHFS, kBloomBlurVFS,
    kShadowAssetVS, kShadowAssetFS, kAssetFS, kGiAoFS, kGiAoDenoiseFS,
    kShaderCount
};
constexpr u32 kHeightMapSize = 1024;

// Must match ObjectData in world.slang (exact 128-byte Vulkan push constant block).
struct ObjectPush {
    Mat4 model;
    f32 base_color[4];
    f32 material[4]; // metallic, roughness, alpha cutoff, normal scale
    f32 emissive[4]; // rgb, occlusion strength
    f32 uv_xform[4]; // tile_u, tile_v, packed(off_u, off_v), packed(rot_rad, pom_depth)
};
static_assert(sizeof(ObjectPush) == 128);

f32 pack_u16_pair(f32 a, f32 lo_a, f32 hi_a, f32 b, f32 lo_b, f32 hi_b) noexcept {
    const f32 na = std::clamp((a - lo_a) / (hi_a - lo_a), 0.0f, 1.0f);
    const f32 nb = std::clamp((b - lo_b) / (hi_b - lo_b), 0.0f, 1.0f);
    const u32 qa = static_cast<u32>(na * 65535.0f + 0.5f) & 0xFFFFu;
    const u32 qb = static_cast<u32>(nb * 65535.0f + 0.5f) & 0xFFFFu;
    return std::bit_cast<f32>(qa | (qb << 16u));
}

void hash_bytes(u64& h, const void* data, usize size) noexcept {
    const auto* p = static_cast<const u8*>(data);
    for (usize i = 0; i < size; ++i) {
        h ^= static_cast<u64>(p[i]);
        h *= 1099511628211ull;
    }
}

constexpr rhi::Format kHdrFormat = rhi::Format::RGBA16_Float;

} // namespace

WorldRenderer::WorldRenderer(rhi::Device& device, u32 shadow_map_size)
    : device_(device), shadow_size_(shadow_map_size) {
    struct Entry {
        const char* name;
        rhi::ShaderStage stage;
    };
    const Entry entries[kShaderCount] = {
        {"skyVS", rhi::ShaderStage::Vertex},    {"skyFS", rhi::ShaderStage::Fragment},
        {"shadowVS", rhi::ShaderStage::Vertex}, {"shadowFS", rhi::ShaderStage::Fragment},
        {"mainVS", rhi::ShaderStage::Vertex},   {"mainFS", rhi::ShaderStage::Fragment},
        {"postVS", rhi::ShaderStage::Vertex},   {"postFS", rhi::ShaderStage::Fragment},
        {"compositeFS", rhi::ShaderStage::Fragment}, {"bloomDownFS", rhi::ShaderStage::Fragment},
        {"bloomBlurHFS", rhi::ShaderStage::Fragment}, {"bloomBlurVFS", rhi::ShaderStage::Fragment},
        {"shadowAssetVS", rhi::ShaderStage::Vertex}, {"shadowAssetFS", rhi::ShaderStage::Fragment},
        {"assetFS", rhi::ShaderStage::Fragment},
        {"giAoFS", rhi::ShaderStage::Fragment},
        {"giAoDenoiseFS", rhi::ShaderStage::Fragment},
    };
    for (int i = 0; i < kShaderCount; ++i) {
        shaders_[i] = device_.create_shader({world_spv, entries[i].name, entries[i].stage});
        if (!shaders_[i].valid()) {
            NOVA_ERROR("Renderer", "failed to create world shader {}", entries[i].name);
            return;
        }
    }

    const rhi::BindingDesc frame_binding{0, rhi::BindingType::UniformBuffer};
    const std::vector<rhi::VertexAttribute> attributes = {
        {0, rhi::VertexFormat::Float3, static_cast<u32>(offsetof(Vertex, position))},
        {1, rhi::VertexFormat::Float3, static_cast<u32>(offsetof(Vertex, normal))},
        {2, rhi::VertexFormat::Float4, static_cast<u32>(offsetof(Vertex, color))},
        {3, rhi::VertexFormat::Float2, static_cast<u32>(offsetof(Vertex, uv))},
        {4, rhi::VertexFormat::Float4, static_cast<u32>(offsetof(Vertex, tangent))},
    };

    rhi::GraphicsPipelineDesc shadow;
    shadow.vertex_shader = shaders_[kShadowVS];
    shadow.fragment_shader = shaders_[kShadowFS];
    shadow.vertex_stride = sizeof(Vertex);
    shadow.attributes = attributes;
    shadow.push_constant_size = sizeof(ObjectPush);
    shadow.bindings = {frame_binding};
    shadow.cull = rhi::CullMode::None; // thin walls and foliage must cast from both sides
    shadow.depth_bias = true;
    shadow.color_format = rhi::Format::Undefined;
    shadow_pipeline_ = device_.create_graphics_pipeline(shadow);

    rhi::GraphicsPipelineDesc sky;
    sky.vertex_shader = shaders_[kSkyVS];
    sky.fragment_shader = shaders_[kSkyFS];
    sky.bindings = {frame_binding};
    sky.cull = rhi::CullMode::None;
    sky.depth_test = false;
    sky.color_format = kHdrFormat;
    sky_pipeline_ = device_.create_graphics_pipeline(sky);

    rhi::GraphicsPipelineDesc main;
    main.vertex_shader = shaders_[kMainVS];
    main.fragment_shader = shaders_[kMainFS];
    main.vertex_stride = sizeof(Vertex);
    main.attributes = attributes;
    main.push_constant_size = sizeof(ObjectPush);
    main.bindings = {frame_binding,
                     {1, rhi::BindingType::SampledTexture, rhi::Filter::Nearest},
                     {5, rhi::BindingType::SampledTexture, rhi::Filter::Nearest}};
    main.color_format = kHdrFormat;
    main_pipeline_ = device_.create_graphics_pipeline(main);
    main.cull = rhi::CullMode::None;
    main_two_sided_pipeline_ = device_.create_graphics_pipeline(main);

    using BT = rhi::BindingType;
    rhi::GraphicsPipelineDesc asset = main;
    asset.fragment_shader = shaders_[kAssetFS];
    for (u32 b = 7; b <= 11; ++b) {
        asset.bindings.push_back({b, BT::SampledTexture, rhi::Filter::Material});
    }
    asset.cull = rhi::CullMode::Back;
    asset_pipeline_ = device_.create_graphics_pipeline(asset);
    asset.cull = rhi::CullMode::None;
    asset_two_sided_pipeline_ = device_.create_graphics_pipeline(asset);

    rhi::GraphicsPipelineDesc asset_shadow = shadow;
    asset_shadow.vertex_shader = shaders_[kShadowAssetVS];
    asset_shadow.fragment_shader = shaders_[kShadowAssetFS];
    asset_shadow.bindings = {frame_binding, {7, BT::SampledTexture, rhi::Filter::Material}};
    asset_shadow_pipeline_ = device_.create_graphics_pipeline(asset_shadow);

    // 1x1 fallbacks for materials without a given texture.
    const u8 white[4] = {255, 255, 255, 255};
    const u8 flat[4] = {128, 128, 255, 255};
    white_srgb_ = device_.create_texture({1, 1, rhi::Format::RGBA8_sRGB, false, true, white});
    white_linear_ = device_.create_texture({1, 1, rhi::Format::RGBA8_Unorm, false, true, white});
    flat_normal_ = device_.create_texture({1, 1, rhi::Format::RGBA8_Unorm, false, true, flat});

    auto fullscreen = [&](ShaderSlot fs, std::vector<rhi::BindingDesc> bindings, rhi::Format color) {
        rhi::GraphicsPipelineDesc desc;
        desc.vertex_shader = shaders_[kPostVS];
        desc.fragment_shader = shaders_[fs];
        desc.bindings = std::move(bindings);
        desc.cull = rhi::CullMode::None;
        desc.depth_test = false;
        desc.depth_format = rhi::Format::Undefined;
        desc.color_format = color;
        return device_.create_graphics_pipeline(desc);
    };
    gi_ao_pipeline_ = fullscreen(kGiAoFS,
                                 {frame_binding,
                                  {2, BT::SampledTexture, rhi::Filter::Linear},
                                  {3, BT::SampledTexture, rhi::Filter::Nearest}},
                                 kHdrFormat);
    gi_ao_denoise_pipeline_ = fullscreen(kGiAoDenoiseFS,
                                         {frame_binding,
                                          {2, BT::SampledTexture, rhi::Filter::Linear},
                                          {3, BT::SampledTexture, rhi::Filter::Nearest}},
                                         kHdrFormat);
    composite_pipeline_ = fullscreen(kCompositeFS,
                                     {frame_binding,
                                      {1, BT::SampledTexture, rhi::Filter::Nearest},
                                      {2, BT::SampledTexture, rhi::Filter::Linear},
                                      {3, BT::SampledTexture, rhi::Filter::Nearest},
                                      {4, BT::SampledTexture, rhi::Filter::Linear}},
                                     kHdrFormat);
    const std::vector<rhi::BindingDesc> src_bindings = {frame_binding, {6, BT::SampledTexture, rhi::Filter::Linear}};
    bloom_down_pipeline_ = fullscreen(kBloomDownFS, src_bindings, kHdrFormat);
    bloom_blur_h_pipeline_ = fullscreen(kBloomBlurHFS, src_bindings, kHdrFormat);
    bloom_blur_v_pipeline_ = fullscreen(kBloomBlurVFS, src_bindings, kHdrFormat);
    post_pipeline_ = fullscreen(kPostFS,
                                {frame_binding,
                                 {2, BT::SampledTexture, rhi::Filter::Linear},
                                 {4, BT::SampledTexture, rhi::Filter::Linear}},
                                rhi::Format::BackBuffer);

    shadow_map_ = device_.create_texture({shadow_size_, shadow_size_, rhi::Format::D32_Float, true, true});
    height_map_ = device_.create_texture({kHeightMapSize, kHeightMapSize, rhi::Format::D32_Float, true, true});
    for (u32 i = 0; i < device_.frames_in_flight(); ++i) {
        frame_buffers_.push_back(device_.create_buffer({sizeof(FrameData), rhi::BufferUsage::Uniform, nullptr}));
    }
    height_frame_buffer_ = device_.create_buffer({sizeof(FrameData), rhi::BufferUsage::Uniform, nullptr});

    valid_ = shadow_pipeline_.valid() && sky_pipeline_.valid() && main_pipeline_.valid() &&
             main_two_sided_pipeline_.valid() && asset_pipeline_.valid() && asset_two_sided_pipeline_.valid() &&
             asset_shadow_pipeline_.valid() && white_srgb_.valid() && white_linear_.valid() && flat_normal_.valid() &&
             gi_ao_pipeline_.valid() && gi_ao_denoise_pipeline_.valid() &&
             composite_pipeline_.valid() && bloom_down_pipeline_.valid() &&
             bloom_blur_h_pipeline_.valid() && bloom_blur_v_pipeline_.valid() && post_pipeline_.valid() &&
             shadow_map_.valid() && height_map_.valid() && height_frame_buffer_.valid() &&
             std::ranges::all_of(frame_buffers_, [](rhi::BufferHandle b) { return b.valid(); });
    if (!valid_) {
        NOVA_ERROR("Renderer", "world renderer initialisation failed");
    }
}

WorldRenderer::~WorldRenderer() {
    for (const GpuMesh& m : meshes_) {
        for (rhi::BufferHandle b : m.dynamic) {
            device_.destroy(b);
        }
        device_.destroy(m.vertices);
        device_.destroy(m.indices);
    }
    for (rhi::BufferHandle b : frame_buffers_) {
        device_.destroy(b);
    }
    device_.destroy(height_frame_buffer_);
    for (rhi::TextureHandle t : owned_textures_) {
        device_.destroy(t);
    }
    for (rhi::TextureHandle t : {hdr_target_, scene_depth_, gi_ao_raw_, gi_ao_blur_, lit_target_, bloom_a_, bloom_b_,
                                 shadow_map_, height_map_, white_srgb_, white_linear_, flat_normal_}) {
        device_.destroy(t);
    }
    for (rhi::PipelineHandle p : {shadow_pipeline_, sky_pipeline_, main_pipeline_, main_two_sided_pipeline_,
                                  asset_pipeline_, asset_two_sided_pipeline_, asset_shadow_pipeline_,
                                  gi_ao_pipeline_, gi_ao_denoise_pipeline_,
                                  composite_pipeline_, bloom_down_pipeline_, bloom_blur_h_pipeline_,
                                  bloom_blur_v_pipeline_, post_pipeline_}) {
        device_.destroy(p);
    }
    for (rhi::ShaderHandle s : shaders_) {
        device_.destroy(s);
    }
}

MeshId WorldRenderer::upload(const MeshData& mesh) {
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

rhi::TextureHandle WorldRenderer::upload_texture(const ImageData& image, bool srgb) {
    if (image.width == 0 || image.height == 0 ||
        image.rgba.size() < static_cast<usize>(image.width) * image.height * 4) {
        return {};
    }
    rhi::TextureDesc desc;
    desc.width = image.width;
    desc.height = image.height;
    desc.format = srgb ? rhi::Format::RGBA8_sRGB : rhi::Format::RGBA8_Unorm;
    desc.render_target = false;
    desc.sampled = true;
    desc.initial_data = image.rgba.data();
    desc.mipmaps = true;
    const rhi::TextureHandle t = device_.create_texture(desc);
    if (t.valid()) {
        owned_textures_.push_back(t);
    }
    return t;
}

MaterialId WorldRenderer::add_material(const PbrMaterial& material) {
    PbrMaterial m = material;
    auto fallback = [](rhi::TextureHandle& t, rhi::TextureHandle def) {
        if (!t.valid()) {
            t = def;
        }
    };
    fallback(m.base_color, white_srgb_);
    fallback(m.metallic_roughness, white_linear_);
    fallback(m.normal, flat_normal_);
    fallback(m.occlusion, white_linear_);
    fallback(m.emissive, white_srgb_);
    materials_.push_back(m);
    return {static_cast<u32>(materials_.size())};
}

void WorldRenderer::update_material(MaterialId id, const PbrMaterial& material) {
    if (!id.valid() || id.id > materials_.size()) {
        return;
    }
    PbrMaterial m = material;
    auto fallback = [](rhi::TextureHandle& t, rhi::TextureHandle def) {
        if (!t.valid()) {
            t = def;
        }
    };
    fallback(m.base_color, white_srgb_);
    fallback(m.metallic_roughness, white_linear_);
    fallback(m.normal, flat_normal_);
    fallback(m.occlusion, white_linear_);
    fallback(m.emissive, white_srgb_);
    materials_[id.id - 1] = m;
}

const PbrMaterial* WorldRenderer::get_material(MaterialId id) const noexcept {
    if (!id.valid() || id.id > materials_.size()) {
        return nullptr;
    }
    return &materials_[id.id - 1];
}

std::vector<MaterialId> WorldRenderer::upload_materials(const ModelData& model) {
    // Each image is uploaded once per colour space it is used in.
    std::vector<rhi::TextureHandle> srgb(model.images.size()), linear(model.images.size());
    auto texture = [&](i32 index, bool is_srgb) -> rhi::TextureHandle {
        if (index < 0 || static_cast<usize>(index) >= model.images.size()) {
            return {};
        }
        rhi::TextureHandle& slot = (is_srgb ? srgb : linear)[static_cast<usize>(index)];
        if (!slot.valid()) {
            slot = upload_texture(model.images[static_cast<usize>(index)], is_srgb);
        }
        return slot;
    };
    std::vector<MaterialId> materials;
    for (const MaterialData& src : model.materials) {
        PbrMaterial m;
        m.base_color = texture(src.base_color_image, true);
        m.metallic_roughness = texture(src.metallic_roughness_image, false);
        m.normal = texture(src.normal_image, false);
        m.occlusion = texture(src.occlusion_image, false);
        m.emissive = texture(src.emissive_image, true);
        m.base_color_factor = src.base_color_factor;
        m.metallic = src.metallic_factor;
        m.roughness = src.roughness_factor;
        m.emissive_factor = src.emissive_factor;
        m.normal_scale = src.normal_scale;
        m.occlusion_strength = src.occlusion_strength;
        m.alpha_cutoff = src.alpha_mode == AlphaMode::Mask    ? std::max(src.alpha_cutoff, 1e-3f)
                         : src.alpha_mode == AlphaMode::Blend ? 0.5f
                                                              : 0.0f;
        m.double_sided = src.double_sided;
        materials.push_back(add_material(m));
    }
    return materials;
}

MeshId WorldRenderer::upload_dynamic(const MeshData& mesh) {
    const MeshId id = upload(mesh);
    if (id.id == 0) {
        return id;
    }
    GpuMesh& gpu = meshes_[id.id - 1];
    const usize bytes = mesh.vertices.size() * sizeof(Vertex);
    for (u32 i = 0; i < device_.frames_in_flight(); ++i) {
        gpu.dynamic.push_back(device_.create_buffer({bytes, rhi::BufferUsage::Vertex, mesh.vertices.data()}));
        gpu.slot_version.push_back(0);
    }
    gpu.pending = mesh.vertices;
    return id;
}

void WorldRenderer::update_vertices(MeshId mesh, std::span<const Vertex> vertices) {
    if (mesh.id == 0 || mesh.id > meshes_.size()) {
        return;
    }
    GpuMesh& gpu = meshes_[mesh.id - 1];
    if (gpu.dynamic.empty() || vertices.size() != gpu.pending.size()) {
        NOVA_ERROR("Renderer", "update_vertices: mesh {} is not dynamic or the vertex count changed", mesh.id);
        return;
    }
    std::copy(vertices.begin(), vertices.end(), gpu.pending.begin());
    ++gpu.version;
}

std::vector<WorldDrawItem> WorldRenderer::upload_model(const ModelData& model, const Mat4& placement) {
    const std::vector<MaterialId> materials = upload_materials(model);
    std::vector<bool> double_sided;
    for (const MaterialData& src : model.materials) {
        double_sided.push_back(src.double_sided);
    }
    MaterialId default_material;

    std::vector<WorldDrawItem> items;
    for (const ModelPrimitive& prim : model.primitives) {
        WorldDrawItem item;
        item.mesh = upload(prim.mesh);
        if (item.mesh.id == 0) {
            continue;
        }
        item.transform = placement * prim.transform;
        if (prim.material >= 0 && static_cast<usize>(prim.material) < materials.size()) {
            item.material = materials[static_cast<usize>(prim.material)];
            item.two_sided = double_sided[static_cast<usize>(prim.material)];
        } else {
            if (!default_material.valid()) {
                PbrMaterial m;
                m.metallic = 0.0f;
                default_material = add_material(m);
            }
            item.material = default_material;
        }
        items.push_back(item);
    }
    return items;
}

bool WorldRenderer::ensure_targets(rhi::Extent extent) {
    if (hdr_target_.valid() && extent.width == hdr_extent_.width && extent.height == hdr_extent_.height) {
        return true;
    }
    for (rhi::TextureHandle t : {hdr_target_, scene_depth_, gi_ao_raw_, gi_ao_blur_, lit_target_, bloom_a_, bloom_b_}) {
        device_.destroy(t);
    }
    const u32 qw = std::max(1u, extent.width / 4);
    const u32 qh = std::max(1u, extent.height / 4);
    hdr_target_ = device_.create_texture({extent.width, extent.height, kHdrFormat, true, true});
    scene_depth_ = device_.create_texture({extent.width, extent.height, rhi::Format::D32_Float, true, true});
    gi_ao_raw_ = device_.create_texture({extent.width, extent.height, kHdrFormat, true, true});
    gi_ao_blur_ = device_.create_texture({extent.width, extent.height, kHdrFormat, true, true});
    lit_target_ = device_.create_texture({extent.width, extent.height, kHdrFormat, true, true});
    bloom_a_ = device_.create_texture({qw, qh, kHdrFormat, true, true});
    bloom_b_ = device_.create_texture({qw, qh, kHdrFormat, true, true});
    hdr_extent_ = extent;
    return hdr_target_.valid() && scene_depth_.valid() && gi_ao_raw_.valid() && gi_ao_blur_.valid() &&
           lit_target_.valid() && bloom_a_.valid() && bloom_b_.valid();
}

bool WorldRenderer::render_frame(const Camera& camera, const WorldSettings& settings,
                                 std::span<const WorldDrawItem> items, const OverlayCallback& overlay) {
    if (!valid_) {
        return false;
    }
    const rhi::Extent output = device_.back_buffer_extent();
    // Internal resolution: > 1 supersamples (the post pass filters it down to the back buffer).
    const f32 scale = std::clamp(settings.render_scale, 0.25f, 2.0f);
    const rhi::Extent extent{std::max(1u, static_cast<u32>(static_cast<f32>(output.width) * scale + 0.5f)),
                             std::max(1u, static_cast<u32>(static_cast<f32>(output.height) * scale + 0.5f))};
    if (output.width == 0 || output.height == 0 || !ensure_targets(extent)) {
        return false;
    }
    rhi::CommandList* cmd = device_.begin_frame();
    if (!cmd) {
        return false;
    }

    // ---- per-frame constants
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
    const Vec3 sun = math::normalize(settings.sun_direction);
    const Vec3 forward = math::normalize(camera.target - camera.position);
    const Vec3 right = math::normalize(math::cross(forward, camera.up));
    const Vec3 up = math::cross(right, forward);
    const f32 tan_half = std::tan(camera.fov_y * 0.5f);

    const f32 r = settings.shadow_radius;
    const Mat4 light_view = math::look_at(settings.shadow_center + sun * (r * 3.0f), settings.shadow_center,
                                          std::abs(sun.y) > 0.99f ? Vec3{0, 0, 1} : Vec3{0, 1, 0});
    const Mat4 light_proj = math::orthographic(-r, r, -r, r, 0.1f, r * 6.0f);

    // Top-down camera for the contact-AO height map; depth is linear in height (orthographic).
    // Fixed at capture time so the map stays valid while the shadow centre follows the player.
    if (height_dirty_) {
        const f32 he = settings.contact_ao_extent;
        constexpr f32 kHeightNear = 0.1f;
        const f32 height_top = settings.shadow_center.y + he * 0.5f;
        const Vec3 height_eye{settings.shadow_center.x, height_top, settings.shadow_center.z};
        const Mat4 height_view = math::look_at(height_eye, height_eye - Vec3{0, 1, 0}, Vec3{0, 0, -1});
        height_view_proj_ = math::orthographic(-he, he, -he, he, kHeightNear, kHeightNear + he) * height_view;
        height_params_ = {height_top, kHeightNear, he}; // depth range covers shadow_center.y +- extent / 2
    }

    FrameData fd{};
    fd.view_proj = camera.projection(aspect) * camera.view();
    fd.light_view_proj = light_proj * light_view;
    fd.height_view_proj = height_view_proj_;
    set4(fd.camera_pos, camera.position, settings.time);
    set4(fd.camera_right, right, tan_half * aspect);
    set4(fd.camera_up, up, tan_half);
    set4(fd.camera_forward, forward, settings.exposure);
    set4(fd.sun_dir, sun, 1.0f / static_cast<f32>(shadow_size_));
    set4(fd.sun_color, settings.sun_color * settings.sun_intensity, 0.0f);
    set4(fd.sky_zenith, settings.sky_zenith, settings.fog_density);
    set4(fd.sky_horizon, settings.sky_horizon, settings.fog_height_falloff);
    set4(fd.ground_color, settings.ground_color, settings.ambient);
    set4(fd.camera_clip, {camera.z_near, camera.z_far, settings.contact_ao}, settings.ssao);
    set4(fd.height_params, height_params_, settings.contact_ao_radius);
    set4(fd.post_params, {settings.bloom, settings.saturation, settings.bounce}, settings.ssao_radius);
    set4(fd.screen, {1.0f / static_cast<f32>(extent.width), 1.0f / static_cast<f32>(extent.height),
                     static_cast<f32>(extent.width)},
         static_cast<f32>(extent.height));
    set4(fd.point_light, settings.point_light_position, settings.point_light_range);
    set4(fd.point_color, settings.point_light_color, settings.crosshair_px);
    set4(fd.output_size, {static_cast<f32>(output.width), static_cast<f32>(output.height), 0.0f}, 0.0f);

    // One buffer per frame in flight: begin_frame() has waited for the GPU to finish this slot.
    const rhi::BufferHandle frame_buffer = frame_buffers_[frame_counter_ % frame_buffers_.size()];
    ++frame_counter_;
    device_.update_buffer(frame_buffer, &fd, sizeof(fd));
    // Dynamic vertices: begin_frame() has also waited for this slot's previous use.
    const usize slot = (frame_counter_ - 1) % frame_buffers_.size();
    for (GpuMesh& m : meshes_) {
        if (!m.dynamic.empty() && m.slot_version[slot] != m.version) {
            device_.update_buffer(m.dynamic[slot], m.pending.data(), m.pending.size() * sizeof(Vertex));
            m.slot_version[slot] = m.version;
        }
    }
    const rhi::ResourceBinding frame_res{0, frame_buffer, {}};

    stats_ = {};
    auto draw_mesh = [&](const WorldDrawItem& item) {
        if (item.mesh.id == 0 || item.mesh.id > meshes_.size()) {
            return;
        }
        const GpuMesh& mesh = meshes_[item.mesh.id - 1];
        if (item.material.valid()) {
            const PbrMaterial& m = materials_[item.material.id - 1];
            ObjectPush push{};
            push.model = item.transform;
            std::memcpy(push.base_color, &m.base_color_factor, sizeof(push.base_color));
            push.material[0] = m.metallic;
            push.material[1] = m.roughness;
            push.material[2] = m.alpha_cutoff;
            push.material[3] = m.normal_scale;
            set4(push.emissive, m.emissive_factor, m.occlusion_strength);
            f32 tu = item.uv_tile.x;
            f32 tv = item.uv_tile.y;
            if (item.world_lock_uv) {
                const f32 sx = std::sqrt(item.transform.m[0][0] * item.transform.m[0][0] +
                                         item.transform.m[0][1] * item.transform.m[0][1] +
                                         item.transform.m[0][2] * item.transform.m[0][2]);
                const f32 sy = std::sqrt(item.transform.m[1][0] * item.transform.m[1][0] +
                                         item.transform.m[1][1] * item.transform.m[1][1] +
                                         item.transform.m[1][2] * item.transform.m[1][2]);
                const f32 sz = std::sqrt(item.transform.m[2][0] * item.transform.m[2][0] +
                                         item.transform.m[2][1] * item.transform.m[2][1] +
                                         item.transform.m[2][2] * item.transform.m[2][2]);
                tu *= std::max(0.05f, std::max(sx, sz));
                tv *= std::max(0.05f, sy);
            }
            push.uv_xform[0] = tu;
            push.uv_xform[1] = tv;
            push.uv_xform[2] = pack_u16_pair(item.uv_offset.x, -8.0f, 8.0f, item.uv_offset.y, -8.0f, 8.0f);
            push.uv_xform[3] = pack_u16_pair(math::radians(item.uv_rotation_deg), -math::kPi, math::kPi,
                                             item.pom_depth, 0.0f, 0.15f);
            cmd->push_constants(&push, sizeof(push));
        } else {
            cmd->push_constants(&item.transform, sizeof(Mat4));
        }
        cmd->bind_vertex_buffer(mesh.dynamic.empty() ? mesh.vertices : mesh.dynamic[slot]);
        cmd->bind_index_buffer(mesh.indices);
        cmd->draw_indexed(mesh.index_count);
        ++stats_.draw_calls;
        stats_.triangles += mesh.index_count / 3;
    };
    auto valid_material = [&](const WorldDrawItem& item) {
        return item.material.valid() && item.material.id <= materials_.size();
    };
    // Depth-only render of the shadow casters with `light_view_proj` taken from `constants`.
    auto depth_pass = [&](rhi::TextureHandle target, const rhi::ResourceBinding& constants) {
        rhi::RenderPassDesc pass;
        pass.back_buffer = false;
        pass.depth = target;
        cmd->begin_render_pass(pass);
        cmd->bind_pipeline(shadow_pipeline_);
        cmd->bind_resources({&constants, 1});
        for (const WorldDrawItem& item : items) {
            if (item.casts_shadow && !item.material.valid()) {
                draw_mesh(item);
            }
        }
        bool asset_bound = false;
        for (const WorldDrawItem& item : items) {
            if (!item.casts_shadow || !valid_material(item)) {
                continue;
            }
            if (!asset_bound) {
                cmd->bind_pipeline(asset_shadow_pipeline_);
                asset_bound = true;
            }
            const rhi::ResourceBinding res[] = {constants, {7, {}, materials_[item.material.id - 1].base_color}};
            cmd->bind_resources(res);
            draw_mesh(item);
        }
        cmd->end_render_pass();
    };
    auto fullscreen_pass = [&](rhi::TextureHandle target, rhi::PipelineHandle pipeline,
                               std::initializer_list<rhi::ResourceBinding> textures) {
        rhi::RenderPassDesc pass;
        pass.back_buffer = !target.valid();
        pass.color = target;
        pass.default_depth = false;
        cmd->begin_render_pass(pass);
        cmd->bind_pipeline(pipeline);
        std::vector<rhi::ResourceBinding> res{frame_res};
        res.insert(res.end(), textures);
        cmd->bind_resources(res);
        cmd->draw(3);
        cmd->end_render_pass();
    };

    // ---- 0. contact-AO height map (static level: captured once)
    if (height_dirty_) {
        FrameData hd = fd;
        hd.light_view_proj = fd.height_view_proj;
        device_.update_buffer(height_frame_buffer_, &hd, sizeof(hd));
        depth_pass(height_map_, {0, height_frame_buffer_, {}});
        height_dirty_ = false;
    }

    // ---- 1. PRISM-VSM smart cached 4K sun shadow map: skips static re-rasterization when sun & casters are unchanged!
    u64 sh = 14695981039346656037ull;
    hash_bytes(sh, &fd.light_view_proj, sizeof(fd.light_view_proj));
    for (const WorldDrawItem& item : items) {
        if (!item.casts_shadow || item.mesh.id == 0 || item.mesh.id > meshes_.size()) {
            continue;
        }
        const GpuMesh& gm = meshes_[item.mesh.id - 1];
        hash_bytes(sh, &item.mesh.id, sizeof(item.mesh.id));
        hash_bytes(sh, &gm.version, sizeof(gm.version));
        hash_bytes(sh, &item.transform, sizeof(item.transform));
        if (valid_material(item)) {
            const PbrMaterial& m = materials_[item.material.id - 1];
            hash_bytes(sh, &m.alpha_cutoff, sizeof(m.alpha_cutoff));
            hash_bytes(sh, &m.base_color.id, sizeof(m.base_color.id));
        }
    }
    if (sh == 0) sh = 1;
    if (sh != shadow_hash_) {
        depth_pass(shadow_map_, frame_res);
        shadow_hash_ = sh;
        stats_.shadow_cached = false;
    } else {
        stats_.shadow_cached = true;
    }

    // ---- 2. sky + lit geometry into HDR colour + sampled depth
    {
        rhi::RenderPassDesc pass;
        pass.back_buffer = false;
        pass.color = hdr_target_;
        pass.depth = scene_depth_;
        cmd->begin_render_pass(pass);

        cmd->bind_pipeline(sky_pipeline_);
        cmd->bind_resources({&frame_res, 1});
        cmd->draw(3);

        const rhi::ResourceBinding main_res[] = {frame_res, {1, {}, shadow_map_}, {5, {}, height_map_}};
        for (bool two_sided : {false, true}) {
            cmd->bind_pipeline(two_sided ? main_two_sided_pipeline_ : main_pipeline_);
            cmd->bind_resources(main_res);
            for (const WorldDrawItem& item : items) {
                if (item.two_sided == two_sided && !item.material.valid()) {
                    draw_mesh(item);
                }
            }
        }
        // Imported assets: one descriptor set per draw for its material textures.
        for (bool two_sided : {false, true}) {
            bool bound = false;
            for (const WorldDrawItem& item : items) {
                if (!valid_material(item) || item.two_sided != two_sided) {
                    continue;
                }
                if (!bound) {
                    cmd->bind_pipeline(two_sided ? asset_two_sided_pipeline_ : asset_pipeline_);
                    bound = true;
                }
                const PbrMaterial& m = materials_[item.material.id - 1];
                const rhi::ResourceBinding res[] = {
                    frame_res, {1, {}, shadow_map_}, {5, {}, height_map_}, {7, {}, m.base_color},
                    {8, {}, m.metallic_roughness}, {9, {}, m.normal}, {10, {}, m.occlusion}, {11, {}, m.emissive},
                };
                cmd->bind_resources(res);
                draw_mesh(item);
            }
        }
        cmd->end_render_pass();
    }

    // ---- 2.5a. PRISM Real-Time Dual-Scale Horizon-Angle XeGTAO + Multi-Bounce Colored SSGI
    fullscreen_pass(gi_ao_raw_, gi_ao_pipeline_, {{2, {}, hdr_target_}, {3, {}, scene_depth_}});
    // ---- 2.5b. Edge-Preserving 5x5 Depth+Normal Joint-Bilateral Denoiser (silky noise-free GI & crevice AO!)
    fullscreen_pass(gi_ao_blur_, gi_ao_denoise_pipeline_, {{2, {}, gi_ao_raw_}, {3, {}, scene_depth_}});

    // ---- 3. Composite: Denoised Multi-Bounce AO + Denoised Colored SSGI + Volumetric Sun Shafts + SSR Puddles
    fullscreen_pass(lit_target_, composite_pipeline_,
                    {{1, {}, shadow_map_}, {2, {}, hdr_target_}, {3, {}, scene_depth_}, {4, {}, gi_ao_blur_}});

    // ---- 4. bloom: 13-tap Jimenez bright pass at quarter resolution + 2x separable 13-tap Gaussian blur
    fullscreen_pass(bloom_a_, bloom_down_pipeline_, {{6, {}, lit_target_}});
    for (int pass = 0; pass < 2; ++pass) {
        fullscreen_pass(bloom_b_, bloom_blur_h_pipeline_, {{6, {}, bloom_a_}});
        fullscreen_pass(bloom_a_, bloom_blur_v_pipeline_, {{6, {}, bloom_b_}});
    }

    // ---- 5. tonemap + FXAA + CAS + grade into the back buffer
    fullscreen_pass({}, post_pipeline_, {{2, {}, lit_target_}, {4, {}, bloom_a_}});

    // ---- 6. optional crisp native 2K UI & 3D gizmo overlay (NOVA Craft Studio)
    if (overlay) {
        overlay(*cmd);
    }

    device_.end_frame();
    return true;
}

} // namespace nova::renderer
