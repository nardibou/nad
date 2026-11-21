#pragma once

// Render Hardware Interface: the engine's only view of the GPU. Backend types (Vulkan) never
// appear here (ADR 0003, 0005). Deliberately small: it grows with the renderer's needs.

#include "nova/core/types.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nova::platform {
class Window;
}

namespace nova::rhi {

// Typed 32-bit handles; 0 is invalid.
template <typename Tag>
struct Handle {
    u32 id = 0;
    [[nodiscard]] constexpr bool valid() const noexcept { return id != 0; }
    constexpr bool operator==(const Handle&) const noexcept = default;
};
using BufferHandle = Handle<struct BufferTag>;
using ShaderHandle = Handle<struct ShaderTag>;
using PipelineHandle = Handle<struct PipelineTag>;
using TextureHandle = Handle<struct TextureTag>;
using BlasHandle = Handle<struct BlasTag>; // bottom-level acceleration structure (one mesh)
using TlasHandle = Handle<struct TlasTag>; // top-level acceleration structure (scene instances)

// Storage: read-only shader storage buffer (std430), host-written like the other buffer kinds.
enum class BufferUsage : u8 { Vertex, Index, Uniform, Storage };
enum class ShaderStage : u8 { Vertex, Fragment };
enum class VertexFormat : u8 { Float2, Float3, Float4 };
enum class CullMode : u8 { None, Back };

enum class Format : u8 {
    Undefined,  // "no attachment" in pipeline descriptions
    BackBuffer, // whatever format the swapchain / offscreen back buffer uses
    RGBA8_sRGB,
    RGBA8_Unorm,
    RGBA16_Float,
    D32_Float,
};

enum class CompareOp : u8 { Less, LessOrEqual, Always };
// Linear/Nearest: clamp-to-edge, no mip filtering (render targets).
// Material: trilinear + anisotropic, repeat addressing (asset textures).
enum class Filter : u8 { Linear, Nearest, Material };

struct TextureDesc {
    u32 width = 0;
    u32 height = 0;
    Format format = Format::RGBA8_sRGB;
    bool render_target = true; // usable as a colour or depth attachment
    bool sampled = true;       // readable from shaders after the pass that wrote it
    // Optional tightly packed RGBA8 pixels (RGBA8 formats only, render_target must be false). The
    // texture is shader-readable immediately; with `mipmaps` the full chain is generated on the GPU.
    const void* initial_data = nullptr;
    bool mipmaps = false;
};

// AccelerationStructure and StorageBuffer require DeviceInfo::ray_tracing / are fragment-visible.
enum class BindingType : u8 { UniformBuffer, SampledTexture, StorageBuffer, AccelerationStructure };

// One descriptor in set 0, visible to vertex and fragment stages.
struct BindingDesc {
    u32 binding = 0;
    BindingType type = BindingType::UniformBuffer;
    Filter filter = Filter::Linear; // SampledTexture only
};

// Value for one BindingDesc at draw time.
struct ResourceBinding {
    u32 binding = 0;
    BufferHandle buffer;   // UniformBuffer, StorageBuffer
    TextureHandle texture; // SampledTexture
    TlasHandle tlas;       // AccelerationStructure
};

// Triangle geometry for a BLAS: positions are Float3 at offset 0 of each `vertex_stride` vertex,
// indices are u32. The buffers must stay alive (and unchanged) while the BLAS is in use.
struct BlasDesc {
    BufferHandle vertices;
    u32 vertex_count = 0;
    u32 vertex_stride = 0;
    BufferHandle indices;
    u32 index_count = 0;
};

struct TlasInstance {
    BlasHandle blas;
    f32 transform[3][4]{}; // row-major 3x4 object-to-world
    u32 custom_index = 0;  // 24 bits, visible to shaders as the instance id
    u8 mask = 0xFF;
};

struct BufferDesc {
    usize size = 0;
    BufferUsage usage = BufferUsage::Vertex;
    const void* initial_data = nullptr; // optional, `size` bytes
};

struct ShaderDesc {
    std::span<const u32> spirv;
    std::string entry_point = "main";
    ShaderStage stage = ShaderStage::Vertex;
};

struct VertexAttribute {
    u32 location = 0;
    VertexFormat format = VertexFormat::Float3;
    u32 offset = 0;
};

struct GraphicsPipelineDesc {
    ShaderHandle vertex_shader;
    ShaderHandle fragment_shader;
    u32 vertex_stride = 0;
    std::vector<VertexAttribute> attributes;
    u32 push_constant_size = 0; // visible to vertex and fragment stages, <= 128 bytes
    std::vector<BindingDesc> bindings;
    CullMode cull = CullMode::Back;
    bool depth_test = true;
    bool depth_write = true;
    CompareOp depth_compare = CompareOp::Less;
    bool depth_bias = false; // slope-scaled bias for shadow maps
    bool alpha_blend = false; // standard SRC_ALPHA / ONE_MINUS_SRC_ALPHA hardware blending
    Format color_format = Format::BackBuffer; // Undefined => depth-only pipeline
    Format depth_format = Format::D32_Float;  // Undefined => no depth attachment
};

struct ClearValues {
    f32 color[4] = {0.0f, 0.0f, 0.0f, 1.0f}; // linear RGBA
    f32 depth = 1.0f;
};

// Attachments for a render pass. Attachments are cleared unless `load_color` preserves colour contents.
// Colour: the back buffer if `back_buffer`, else `color` if valid, else none.
// Depth: `depth` if valid, else the device's back-buffer-sized depth buffer if `default_depth`, else none.
struct RenderPassDesc {
    bool back_buffer = true;
    TextureHandle color;
    TextureHandle depth;
    bool default_depth = true;
    bool load_color = false; // preserve previous colour attachment contents (for UI overlays)
    ClearValues clear;
};

// Records commands for the current frame. Valid between Device::begin_frame and end_frame.
class CommandList {
public:
    virtual ~CommandList() = default;

    virtual void begin_render_pass(const RenderPassDesc& desc) = 0;
    // Back buffer + default depth.
    void begin_render_pass(const ClearValues& clear) {
        RenderPassDesc desc;
        desc.clear = clear;
        begin_render_pass(desc);
    }
    // Sampled render targets become shader-readable when their pass ends.
    virtual void end_render_pass() = 0;

    virtual void bind_pipeline(PipelineHandle pipeline) = 0;
    // Binds descriptor set 0 of the current pipeline. Textures must not be attachments of the open pass.
    virtual void bind_resources(std::span<const ResourceBinding> resources) = 0;
    virtual void bind_vertex_buffer(BufferHandle buffer, usize offset = 0) = 0;
    virtual void bind_index_buffer(BufferHandle buffer, usize offset = 0) = 0; // u32 indices
    virtual void push_constants(const void* data, u32 size) = 0;
    virtual void draw(u32 vertex_count, u32 first_vertex = 0) = 0;
    virtual void draw_indexed(u32 index_count, u32 first_index = 0, i32 vertex_offset = 0) = 0;

    // Ray tracing (DeviceInfo::ray_tracing). Outside render passes only. Rebuilds `tlas` from
    // `instances` (<= its capacity); later passes in this frame may trace against it.
    virtual void build_tlas(TlasHandle tlas, std::span<const TlasInstance> instances) = 0;
};

struct DeviceDesc {
    // nullptr => headless: render into an offscreen target of width x height.
    platform::Window* window = nullptr;
    u32 width = 1280;
    u32 height = 720;
    bool vsync = true;
    bool validation = false; // Vulkan validation layers + debug messenger, if installed
    bool ray_tracing = true; // use hardware ray queries when available (NOVA_NO_RT=1 also disables)
};

struct DeviceInfo {
    std::string backend;    // "Vulkan"
    std::string device_name;
    std::string api_version;
    bool software = false;  // CPU implementation (e.g. lavapipe)
    bool validation = false;
    bool gpu_timing = false; // timestamp queries supported
    bool ray_tracing = false; // acceleration structures + inline ray queries in fragment shaders
};

struct Extent {
    u32 width = 0;
    u32 height = 0;
};

class Device {
public:
    // Returns nullptr (logged) when no suitable GPU/driver is available.
    [[nodiscard]] static std::unique_ptr<Device> create(const DeviceDesc& desc);
    virtual ~Device() = default;

    [[nodiscard]] virtual const DeviceInfo& info() const noexcept = 0;
    [[nodiscard]] virtual Extent back_buffer_extent() const noexcept = 0;

    [[nodiscard]] virtual BufferHandle create_buffer(const BufferDesc& desc) = 0;
    // Host-visible write; caller must ensure the GPU is not reading the range this frame.
    virtual void update_buffer(BufferHandle buffer, const void* data, usize size, usize offset = 0) = 0;
    [[nodiscard]] virtual ShaderHandle create_shader(const ShaderDesc& desc) = 0;
    [[nodiscard]] virtual PipelineHandle create_graphics_pipeline(const GraphicsPipelineDesc& desc) = 0;
    [[nodiscard]] virtual TextureHandle create_texture(const TextureDesc& desc) = 0;
    // Destruction waits for the GPU to go idle (simple and safe; fine outside the hot path).
    virtual void destroy(BufferHandle handle) = 0;
    virtual void destroy(ShaderHandle handle) = 0;
    virtual void destroy(PipelineHandle handle) = 0;
    virtual void destroy(TextureHandle handle) = 0;

    // Ray tracing (DeviceInfo::ray_tracing); invalid handles / 0 otherwise. BLAS builds complete
    // before returning (outside begin_frame/end_frame). Buffer addresses are for shader pointers.
    [[nodiscard]] virtual BlasHandle create_blas(const BlasDesc& desc) = 0;
    [[nodiscard]] virtual TlasHandle create_tlas(u32 max_instances) = 0;
    [[nodiscard]] virtual u64 buffer_address(BufferHandle buffer) = 0;
    virtual void destroy(BlasHandle handle) = 0;
    virtual void destroy(TlasHandle handle) = 0;

    // Frames the CPU may record ahead of the GPU; per-frame buffers need this many copies.
    [[nodiscard]] virtual u32 frames_in_flight() const noexcept = 0;

    // Returns nullptr when the frame must be skipped (e.g. minimized window).
    [[nodiscard]] virtual CommandList* begin_frame() = 0;
    virtual void end_frame() = 0;
    // Call after the window size changes; the swapchain is also recreated lazily when stale.
    virtual void resize(u32 width, u32 height) = 0;
    virtual void wait_idle() = 0;

    // GPU time of the most recently completed frame in ms (negative until available).
    [[nodiscard]] virtual f64 last_gpu_frame_ms() const noexcept = 0;
    // Validation-layer errors reported so far (always 0 without validation).
    [[nodiscard]] virtual u32 validation_error_count() const noexcept = 0;

    // Headless only: copies the last rendered frame as tightly packed RGBA8 (sRGB) pixels.
    [[nodiscard]] virtual bool read_back(std::vector<u8>& rgba) = 0;
};

} // namespace nova::rhi
