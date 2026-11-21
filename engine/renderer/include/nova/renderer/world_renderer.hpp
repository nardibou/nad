#pragma once

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/model.hpp"
#include "nova/renderer/renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace nova::renderer {

// Material ids stored in Vertex::color.a; must match world.slang.
enum class Material : u8 {
    Plaster = 0,
    Stone = 1,
    Dirt = 2,
    Wood = 3,
    Foliage = 4,
    Water = 5,
    Terrain = 6, // shader blends dirt / dry grass / rock / snow by height, slope and noise
    Grass = 7,
    Emissive = 8,
    Flag = 9, // vertex colour rg = flag uv
    Cloth = 10,
    Metal = 11,
    Clay = 12, // pure matte architectural clay (GI & lighting benchmark material: clean surface, soft bevel sheen)
};

[[nodiscard]] constexpr math::Vec4 material_color(Material m, const math::Vec3& albedo) noexcept {
    return {albedo.x, albedo.y, albedo.z, static_cast<f32>(m)};
}

// Lighting and atmosphere for one view. Defaults give a late-afternoon sun.
struct WorldSettings {
    math::Vec3 sun_direction{0.8f, 0.27f, -0.36f}; // towards the sun
    math::Vec3 sun_color{1.0f, 0.80f, 0.65f};
    f32 sun_intensity = 2.85f;
    math::Vec3 sky_zenith{0.10f, 0.25f, 0.62f};
    math::Vec3 sky_horizon{0.56f, 0.68f, 0.86f};
    math::Vec3 ground_color{0.42f, 0.30f, 0.20f};
    f32 ambient = 0.56f;
    f32 fog_density = 0.0006f;
    f32 fog_height_falloff = 0.016f;
    f32 exposure = 0.68f;
    f32 time = 0.0f;
    // Area covered by the sun shadow map (orthographic, centred on shadow_center).
    math::Vec3 shadow_center{0, 0, 0};
    f32 shadow_radius = 50.0f;
    // Contact AO: height map of the level covering shadow_center +- contact_ao_extent.
    f32 contact_ao = 0.7f;
    f32 contact_ao_radius = 2.5f;
    f32 contact_ao_extent = 60.0f;
    f32 ssao = 0.8f;         // 0 disables
    f32 ssao_radius = 0.7f;  // metres
    f32 bounce = 0.45f;      // screen-space colour bleed
    f32 bloom = 0.14f;
    f32 saturation = 1.0f;
    // Internal resolution relative to the back buffer: 1.5 = supersampling ("ultra"), < 1 = faster.
    f32 render_scale = 1.0f;
    // One unshadowed point light (muzzle flash, torch...); zero colour disables it.
    math::Vec3 point_light_position{0.0f, 0.0f, 0.0f};
    math::Vec3 point_light_color{0.0f, 0.0f, 0.0f}; // radiance
    f32 point_light_range = 8.0f;
    f32 crosshair_px = 0.0f; // centred crosshair arm length in back-buffer pixels; 0 = none

    // NOVA Lumina RT (used when the device reports rhi::DeviceInfo::ray_tracing): ray-traced sun
    // shadows, AO, one-bounce diffuse GI and water reflections, temporally accumulated and denoised.
    // Without RT support the raster path (shadow map, contact AO, SSAO/SSGI, SSR) is used unchanged.
    bool ray_tracing = true;
    f32 sun_angular_radius = 0.0093f; // radians; ~2x the real sun, widened by low-sun scattering
    f32 rt_indirect = 1.0f;           // scale of ray-traced sky + bounce light
    f32 rt_gi_distance = 16.0f;       // metres a GI ray may travel before it reads the sky
    f32 rt_ao_radius = 0.9f;          // metres
    // Sun shadow rays see every occluder within this range, including distant terrain the raster
    // shadow map (shadow_radius) never covered. Lower it to keep only local occluders.
    f32 rt_shadow_distance = 2000.0f;
    u32 rt_gi_rays = 1;               // GI rays per pixel per frame (1-4), accumulated over time
    bool rt_reflections = true;       // ray-traced water reflections (else screen-space)
    u32 rt_debug_view = 0;            // 0 off, 1 sun shadow, 2 AO, 3 indirect light (denoised signals)
};

// glTF-style metallic-roughness material for imported assets. Invalid textures fall back to
// neutral defaults (white / flat normal), so factors alone also work.
struct PbrMaterial {
    rhi::TextureHandle base_color;         // sRGB
    rhi::TextureHandle metallic_roughness; // linear: G roughness, B metallic
    rhi::TextureHandle normal;             // linear, tangent space
    rhi::TextureHandle occlusion;          // linear: R
    rhi::TextureHandle emissive;           // sRGB
    math::Vec4 base_color_factor{1.0f, 1.0f, 1.0f, 1.0f};
    f32 metallic = 1.0f;
    f32 roughness = 1.0f;
    math::Vec3 emissive_factor{0.0f, 0.0f, 0.0f};
    f32 normal_scale = 1.0f;
    f32 occlusion_strength = 1.0f;
    f32 alpha_cutoff = 0.0f; // > 0 enables alpha testing (glTF MASK; BLEND is approximated by 0.5)
    bool double_sided = false;
};

struct MaterialId {
    u32 id = 0; // 0 = procedural (Vertex::color.a material ids)
    [[nodiscard]] constexpr bool valid() const noexcept { return id != 0; }
};

struct WorldDrawItem {
    MeshId mesh;
    math::Mat4 transform = math::Mat4::identity();
    bool two_sided = false; // foliage, grass, flags (asset materials use PbrMaterial::double_sided)
    bool casts_shadow = true;
    MaterialId material;    // valid => textured PBR asset shading
    // NOVA PRISM / Craft live per-object UV & Micro-Geo controls:
    math::Vec2 uv_tile{1.0f, 1.0f};
    math::Vec2 uv_offset{0.0f, 0.0f};
    f32 uv_rotation_deg = 0.0f;
    f32 pom_depth = 0.0f;       // > 0 enables 8-step Steep POM + 4-step Heightfield Micro-Self-Shadowing
    bool world_lock_uv = false; // scales UVs automatically by object scale so textures never stretch
};

struct WorldRenderStats {
    u32 draw_calls = 0;
    u32 triangles = 0;
    bool shadow_cached = false; // true when PRISM-VSM reused the cached 4K shadow map (0 ms shadow pass!)
    bool ray_traced = false;    // NOVA Lumina RT lighting was used this frame
    u32 rt_instances = 0;       // instances in the scene TLAS
    bool tlas_rebuilt = false;  // false when the cached TLAS was reused
};

// HDR forward renderer: contact-AO height map (once) + sun shadow map -> sky and lit geometry into
// RGBA16F + depth -> SSAO and colour bleed -> quarter-res bloom -> tonemap, FXAA and warm grade.
class WorldRenderer {
public:
    explicit WorldRenderer(rhi::Device& device, u32 shadow_map_size = 2048);
    ~WorldRenderer();
    WorldRenderer(const WorldRenderer&) = delete;
    WorldRenderer& operator=(const WorldRenderer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] MeshId upload(const MeshData& mesh);
    // Mesh whose vertices change every frame (skinned characters, viewmodels). Indices are fixed;
    // update_vertices() must pass the same vertex count and takes effect from the next frame.
    [[nodiscard]] MeshId upload_dynamic(const MeshData& mesh);
    void update_vertices(MeshId mesh, std::span<const Vertex> vertices);
    // Textures + materials of `model`, one MaterialId per ModelData::materials entry.
    [[nodiscard]] std::vector<MaterialId> upload_materials(const ModelData& model);
    // Asset resources are owned by the renderer and live until it is destroyed.
    [[nodiscard]] rhi::TextureHandle upload_texture(const ImageData& image, bool srgb);
    [[nodiscard]] MaterialId add_material(const PbrMaterial& material);
    void update_material(MaterialId id, const PbrMaterial& material);
    [[nodiscard]] const PbrMaterial* get_material(MaterialId id) const noexcept;
    // Uploads every image, material and primitive of `model` and returns ready-to-draw items
    // (transform = placement * node transform). Call before the first render_frame so the
    // contact-AO height map includes the model, or call invalidate_contact_ao() afterwards.
    [[nodiscard]] std::vector<WorldDrawItem> upload_model(const ModelData& model,
                                                          const math::Mat4& placement = math::Mat4::identity());

    using OverlayCallback = std::function<void(rhi::CommandList&)>;
    bool render_frame(const Camera& camera, const WorldSettings& settings, std::span<const WorldDrawItem> items,
                      const OverlayCallback& overlay = {});
    [[nodiscard]] const WorldRenderStats& stats() const noexcept { return stats_; }
    // True when the device and renderer can run NOVA Lumina RT (WorldSettings::ray_tracing).
    [[nodiscard]] bool ray_tracing_available() const noexcept { return rt_ != nullptr; }
    // The contact-AO height map & PRISM-VSM shadow cache auto-invalidate on scene/sun changes; call manually if needed.
    void invalidate_contact_ao() noexcept { height_dirty_ = true; shadow_hash_ = 0; }
    void invalidate_shadows() noexcept { shadow_hash_ = 0; }

private:
    struct GpuMesh {
        rhi::BufferHandle vertices;
        rhi::BufferHandle indices;
        u32 index_count = 0;
        // Dynamic meshes: one vertex buffer per frame in flight, refreshed from `pending`.
        std::vector<rhi::BufferHandle> dynamic;
        std::vector<u64> slot_version;
        std::vector<Vertex> pending;
        u64 version = 0;
        rhi::BlasHandle blas; // static meshes on ray-tracing devices
    };
    struct RtState;

    bool ensure_targets(rhi::Extent extent);
    [[nodiscard]] MeshId upload_mesh(const MeshData& mesh, bool build_blas);
    void init_ray_tracing();
    bool ensure_rt_targets(rhi::Extent extent);
    void destroy_ray_tracing();

    rhi::Device& device_;
    std::unique_ptr<RtState> rt_; // null without ray-tracing support
    std::vector<math::Vec3> material_albedo_; // average linear base colour per material (RT hit shading)
    u32 shadow_size_;
    bool valid_ = false;
    bool height_dirty_ = true;
    u64 shadow_hash_ = 0; // PRISM-VSM static shadow cache signature
    rhi::ShaderHandle shaders_[17]{};
    rhi::PipelineHandle shadow_pipeline_;
    rhi::PipelineHandle sky_pipeline_;
    rhi::PipelineHandle main_pipeline_;
    rhi::PipelineHandle main_two_sided_pipeline_;
    rhi::PipelineHandle asset_shadow_pipeline_;
    rhi::PipelineHandle asset_pipeline_;
    rhi::PipelineHandle asset_two_sided_pipeline_;
    rhi::PipelineHandle gi_ao_pipeline_;
    rhi::PipelineHandle gi_ao_denoise_pipeline_;
    rhi::PipelineHandle composite_pipeline_;
    rhi::PipelineHandle bloom_down_pipeline_;
    rhi::PipelineHandle bloom_blur_h_pipeline_;
    rhi::PipelineHandle bloom_blur_v_pipeline_;
    rhi::PipelineHandle post_pipeline_;
    rhi::TextureHandle shadow_map_;
    rhi::TextureHandle height_map_;
    rhi::TextureHandle hdr_target_;
    rhi::TextureHandle scene_depth_;
    rhi::TextureHandle gi_ao_raw_;
    rhi::TextureHandle gi_ao_blur_;
    rhi::TextureHandle lit_target_;
    rhi::TextureHandle bloom_a_;
    rhi::TextureHandle bloom_b_;
    rhi::Extent hdr_extent_{};
    std::vector<rhi::BufferHandle> frame_buffers_;
    rhi::BufferHandle height_frame_buffer_;
    math::Mat4 height_view_proj_ = math::Mat4::identity();
    math::Vec3 height_params_{0.0f, 0.0f, 1.0f};
    u32 frame_counter_ = 0;
    std::vector<GpuMesh> meshes_;
    std::vector<PbrMaterial> materials_; // resolved: every texture valid
    std::vector<rhi::TextureHandle> owned_textures_;
    rhi::TextureHandle white_srgb_;
    rhi::TextureHandle white_linear_;
    rhi::TextureHandle flat_normal_;
    WorldRenderStats stats_;
};

} // namespace nova::renderer
