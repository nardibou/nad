#pragma once

// NOVA Craft Native 2K Vulkan UI & 3D Gizmo Overlay Renderer
// - Zero external UI framework dependencies; runs directly through nova::rhi::Device
// - Crisp TrueType anti-aliased font atlas (Segoe UI / Consolas via stb_truetype, fallback to stb_easy_font)
// - Supports sleek rounded dark-glass panels, real PBR texture thumbnails, interactive sliders & 3D viewport gizmos

#include "nova/core/types.hpp"
#include "nova/math/math.hpp"
#include "nova/renderer/model.hpp"
#include "nova/rhi/rhi.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace nova::renderer {

struct UiVertex {
    f32 x = 0, y = 0;
    f32 u = 0, v = 0;
    f32 r = 1, g = 1, b = 1, a = 1;
};

// Loads any PNG/JPG/BMP/TGA image from disk into RGBA8 ImageData (for OS drag-and-drop into NOVA Craft).
[[nodiscard]] bool load_image_file(const std::string& path, ImageData& out);

class UiOverlay {
public:
    explicit UiOverlay(rhi::Device& device);
    ~UiOverlay();
    UiOverlay(const UiOverlay&) = delete;
    UiOverlay& operator=(const UiOverlay&) = delete;

    [[nodiscard]] bool valid() const noexcept { return valid_; }

    void begin(u32 screen_width, u32 screen_height);

    // Solid & rounded UI primitives (coordinates in back-buffer pixels, origin top-left)
    void rect(f32 x, f32 y, f32 w, f32 h, const math::Vec4& col);
    void rect_outline(f32 x, f32 y, f32 w, f32 h, f32 thickness, const math::Vec4& col);
    void rect_rounded(f32 x, f32 y, f32 w, f32 h, f32 radius, const math::Vec4& fill,
                      const math::Vec4& border = {0, 0, 0, 0}, f32 border_w = 1.0f);
    void textured_rect(f32 x, f32 y, f32 w, f32 h, rhi::TextureHandle tex,
                       const math::Vec4& tint = {1, 1, 1, 1});
    void line(f32 x0, f32 y0, f32 x1, f32 y1, f32 thickness, const math::Vec4& col);
    void circle_filled(f32 cx, f32 cy, f32 r, const math::Vec4& col, int segments = 20);
    void circle_outline(f32 cx, f32 cy, f32 r, f32 thickness, const math::Vec4& col, int segments = 32);
    void triangle(f32 x0, f32 y0, f32 x1, f32 y1, f32 x2, f32 y2, const math::Vec4& col);

    // Crisp TrueType text rendering
    void text(f32 x, f32 y, std::string_view str, f32 size_px, const math::Vec4& col);
    [[nodiscard]] f32 measure_text(std::string_view str, f32 size_px) const noexcept;

    // Flushes all queued quads & text onto the back buffer (preserving the 3D scene via load_color = true)
    void render(rhi::CommandList& cmd);

private:
    struct Batch {
        rhi::TextureHandle texture;
        u32 index_offset = 0;
        u32 index_count = 0;
    };

    struct Glyph {
        f32 x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        f32 u0 = 0, v0 = 0, u1 = 0, v1 = 0;
        f32 advance = 10.0f;
    };

    void ensure_batch(rhi::TextureHandle tex);
    void push_quad(f32 x0, f32 y0, f32 x1, f32 y1, f32 u0, f32 v0, f32 u1, f32 v1, const math::Vec4& col);
    void bake_font();

    rhi::Device& device_;
    bool valid_ = false;
    u32 screen_w_ = 1280;
    u32 screen_h_ = 720;
    rhi::ShaderHandle vs_;
    rhi::ShaderHandle fs_;
    rhi::PipelineHandle pipeline_;
    rhi::TextureHandle white_tex_;
    rhi::TextureHandle font_tex_;
    f32 font_base_px_ = 28.0f;
    f32 font_ascent_ = 22.0f;
    Glyph glyphs_[96]{}; // ASCII 32..127
    bool has_ttf_ = false;

    // Per-frame dynamic vertex/index buffers
    std::vector<rhi::BufferHandle> vbufs_;
    std::vector<rhi::BufferHandle> ibufs_;
    usize vbuf_cap_ = 0;
    usize ibuf_cap_ = 0;
    u32 frame_idx_ = 0;

    std::vector<UiVertex> vertices_;
    std::vector<u32> indices_;
    std::vector<Batch> batches_;
};

} // namespace nova::renderer
