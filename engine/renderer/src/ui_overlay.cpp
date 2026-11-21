#include "nova/renderer/ui_overlay.hpp"

#include "nova/core/log.hpp"

#include "ui_spv.hpp"

#include <stb_easy_font.h>
#include <stb_image.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace nova::renderer {

bool load_image_file(const std::string& path, ImageData& out) {
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!pixels || w <= 0 || h <= 0) {
        if (pixels) stbi_image_free(pixels);
        return false;
    }
    out.width = static_cast<u32>(w);
    out.height = static_cast<u32>(h);
    out.rgba.assign(pixels, pixels + static_cast<usize>(w) * static_cast<usize>(h) * 4);
    stbi_image_free(pixels);
    return true;
}

UiOverlay::UiOverlay(rhi::Device& device) : device_(device) {
    vs_ = device_.create_shader({ui_spv, "uiVS", rhi::ShaderStage::Vertex});
    fs_ = device_.create_shader({ui_spv, "uiFS", rhi::ShaderStage::Fragment});
    if (!vs_.valid() || !fs_.valid()) {
        NOVA_ERROR("UI", "failed to create UI shaders");
        return;
    }

    rhi::GraphicsPipelineDesc desc;
    desc.vertex_shader = vs_;
    desc.fragment_shader = fs_;
    desc.vertex_stride = sizeof(UiVertex);
    desc.attributes = {
        {0, rhi::VertexFormat::Float2, static_cast<u32>(offsetof(UiVertex, x))},
        {1, rhi::VertexFormat::Float2, static_cast<u32>(offsetof(UiVertex, u))},
        {2, rhi::VertexFormat::Float4, static_cast<u32>(offsetof(UiVertex, r))},
    };
    desc.push_constant_size = 16;
    desc.bindings = {{0, rhi::BindingType::SampledTexture, rhi::Filter::Linear}};
    desc.cull = rhi::CullMode::None;
    desc.depth_test = false;
    desc.depth_write = false;
    desc.alpha_blend = true;
    desc.color_format = rhi::Format::BackBuffer;
    desc.depth_format = rhi::Format::Undefined;
    pipeline_ = device_.create_graphics_pipeline(desc);

    const u8 white[4] = {255, 255, 255, 255};
    white_tex_ = device_.create_texture({1, 1, rhi::Format::RGBA8_Unorm, false, true, white, false});

    bake_font();

    const u32 frames = std::max(1u, device_.frames_in_flight());
    vbuf_cap_ = 65536;
    ibuf_cap_ = 131072;
    vbufs_.resize(frames);
    ibufs_.resize(frames);
    for (u32 i = 0; i < frames; ++i) {
        vbufs_[i] = device_.create_buffer({vbuf_cap_ * sizeof(UiVertex), rhi::BufferUsage::Vertex});
        ibufs_[i] = device_.create_buffer({ibuf_cap_ * sizeof(u32), rhi::BufferUsage::Index});
    }

    valid_ = pipeline_.valid() && white_tex_.valid() && !vbufs_.empty() && vbufs_[0].valid();
    if (valid_) {
        NOVA_INFO("UI", "NOVA Craft 2K UI overlay ready (TrueType font: {})", has_ttf_ ? "Segoe UI" : "built-in");
    }
}

UiOverlay::~UiOverlay() {
    for (auto b : vbufs_) device_.destroy(b);
    for (auto b : ibufs_) device_.destroy(b);
    device_.destroy(font_tex_);
    device_.destroy(white_tex_);
    device_.destroy(pipeline_);
    device_.destroy(vs_);
    device_.destroy(fs_);
}

void UiOverlay::bake_font() {
    const std::array<const char*, 5> font_paths = {
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    };
    std::vector<u8> ttf;
    for (const char* p : font_paths) {
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) continue;
        std::FILE* f = std::fopen(p, "rb");
        if (!f) continue;
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz > 1024) {
            ttf.resize(static_cast<usize>(sz));
            if (std::fread(ttf.data(), 1, ttf.size(), f) == ttf.size()) {
                std::fclose(f);
                break;
            }
            ttf.clear();
        }
        std::fclose(f);
    }
    if (ttf.empty()) {
        return;
    }

    constexpr int kW = 512, kH = 512;
    font_base_px_ = 32.0f;
    std::vector<u8> alpha(static_cast<usize>(kW * kH), 0u);
    stbtt_bakedchar baked[96]{};
    const int res = stbtt_BakeFontBitmap(ttf.data(), 0, font_base_px_, alpha.data(), kW, kH, 32, 96, baked);
    if (res <= 0) {
        return;
    }
    stbtt_fontinfo info{};
    if (stbtt_InitFont(&info, ttf.data(), 0)) {
        int asc = 0, desc = 0, lg = 0;
        stbtt_GetFontVMetrics(&info, &asc, &desc, &lg);
        const f32 sc = stbtt_ScaleForPixelHeight(&info, font_base_px_);
        font_ascent_ = static_cast<f32>(asc) * sc;
    } else {
        font_ascent_ = font_base_px_ * 0.78f;
    }

    for (int i = 0; i < 96; ++i) {
        const stbtt_bakedchar& bc = baked[i];
        glyphs_[i] = {
            bc.xoff, bc.yoff, bc.xoff + static_cast<f32>(bc.x1 - bc.x0), bc.yoff + static_cast<f32>(bc.y1 - bc.y0),
            static_cast<f32>(bc.x0) / static_cast<f32>(kW), static_cast<f32>(bc.y0) / static_cast<f32>(kH),
            static_cast<f32>(bc.x1) / static_cast<f32>(kW), static_cast<f32>(bc.y1) / static_cast<f32>(kH),
            bc.xadvance,
        };
    }

    std::vector<u8> rgba(static_cast<usize>(kW * kH * 4), 255u);
    for (usize i = 0; i < alpha.size(); ++i) {
        rgba[i * 4 + 3] = alpha[i];
    }
    font_tex_ = device_.create_texture({kW, kH, rhi::Format::RGBA8_Unorm, false, true, rgba.data(), false});
    has_ttf_ = font_tex_.valid();
}

void UiOverlay::begin(u32 screen_width, u32 screen_height) {
    screen_w_ = std::max(1u, screen_width);
    screen_h_ = std::max(1u, screen_height);
    vertices_.clear();
    indices_.clear();
    batches_.clear();
}

void UiOverlay::ensure_batch(rhi::TextureHandle tex) {
    if (!tex.valid()) tex = white_tex_;
    if (batches_.empty() || batches_.back().texture != tex) {
        batches_.push_back({tex, static_cast<u32>(indices_.size()), 0});
    }
}

void UiOverlay::push_quad(f32 x0, f32 y0, f32 x1, f32 y1, f32 u0, f32 v0, f32 u1, f32 v1, const math::Vec4& col) {
    const u32 base = static_cast<u32>(vertices_.size());
    vertices_.push_back({x0, y0, u0, v0, col.x, col.y, col.z, col.w});
    vertices_.push_back({x1, y0, u1, v0, col.x, col.y, col.z, col.w});
    vertices_.push_back({x1, y1, u1, v1, col.x, col.y, col.z, col.w});
    vertices_.push_back({x0, y1, u0, v1, col.x, col.y, col.z, col.w});
    indices_.insert(indices_.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    batches_.back().index_count += 6;
}

void UiOverlay::rect(f32 x, f32 y, f32 w, f32 h, const math::Vec4& col) {
    if (w <= 0.0f || h <= 0.0f || col.w <= 0.001f) return;
    ensure_batch(white_tex_);
    push_quad(x, y, x + w, y + h, 0, 0, 1, 1, col);
}

void UiOverlay::rect_outline(f32 x, f32 y, f32 w, f32 h, f32 t, const math::Vec4& col) {
    rect(x, y, w, t, col);
    rect(x, y + h - t, w, t, col);
    rect(x, y + t, t, h - 2.0f * t, col);
    rect(x + w - t, y + t, t, h - 2.0f * t, col);
}

void UiOverlay::rect_rounded(f32 x, f32 y, f32 w, f32 h, f32 radius, const math::Vec4& fill,
                             const math::Vec4& border, f32 border_w) {
    if (w <= 0.0f || h <= 0.0f) return;
    const f32 r = std::clamp(radius, 0.0f, std::min(w, h) * 0.5f);
    if (r < 1.5f) {
        rect(x, y, w, h, fill);
        if (border.w > 0.001f && border_w > 0.0f) rect_outline(x, y, w, h, border_w, border);
        return;
    }
    constexpr int kCornerSegs = 6;
    auto build_ring = [&](f32 rx, f32 ry, f32 rw, f32 rh, f32 rad, std::vector<math::Vec2>& pts) {
        pts.clear();
        const f32 cx[4] = {rx + rw - rad, rx + rad, rx + rad, rx + rw - rad};
        const f32 cy[4] = {ry + rad, ry + rad, ry + rh - rad, ry + rh - rad};
        const f32 a0[4] = {-math::kPi * 0.5f, -math::kPi, math::kPi * 0.5f, 0.0f};
        // Clockwise order starting top-right -> bottom-right -> bottom-left -> top-left
        const int order[4] = {0, 3, 2, 1};
        const f32 start_a[4] = {-math::kPi * 0.5f, 0.0f, math::kPi * 0.5f, math::kPi};
        for (int c = 0; c < 4; ++c) {
            const int idx = order[c];
            for (int s = 0; s <= kCornerSegs; ++s) {
                const f32 a = start_a[c] + (static_cast<f32>(s) / kCornerSegs) * (math::kPi * 0.5f);
                pts.push_back({cx[idx] + std::cos(a) * rad, cy[idx] + std::sin(a) * rad});
            }
        }
        (void)a0;
    };

    std::vector<math::Vec2> outer;
    build_ring(x, y, w, h, r, outer);

    ensure_batch(white_tex_);
    if (fill.w > 0.001f) {
        const u32 center_v = static_cast<u32>(vertices_.size());
        vertices_.push_back({x + w * 0.5f, y + h * 0.5f, 0.5f, 0.5f, fill.x, fill.y, fill.z, fill.w});
        for (const auto& pt : outer) {
            vertices_.push_back({pt.x, pt.y, 0.5f, 0.5f, fill.x, fill.y, fill.z, fill.w});
        }
        const u32 n = static_cast<u32>(outer.size());
        for (u32 i = 0; i < n; ++i) {
            indices_.insert(indices_.end(), {center_v, center_v + 1 + i, center_v + 1 + ((i + 1) % n)});
            batches_.back().index_count += 3;
        }
    }
    if (border.w > 0.001f && border_w > 0.0f) {
        std::vector<math::Vec2> inner;
        const f32 ri = std::max(0.5f, r - border_w);
        build_ring(x + border_w, y + border_w, std::max(1.0f, w - 2.0f * border_w),
                   std::max(1.0f, h - 2.0f * border_w), ri, inner);
        const u32 base = static_cast<u32>(vertices_.size());
        const u32 n = static_cast<u32>(outer.size());
        for (u32 i = 0; i < n; ++i) {
            vertices_.push_back({outer[i].x, outer[i].y, 0.5f, 0.5f, border.x, border.y, border.z, border.w});
            vertices_.push_back({inner[i].x, inner[i].y, 0.5f, 0.5f, border.x, border.y, border.z, border.w});
        }
        for (u32 i = 0; i < n; ++i) {
            const u32 next = (i + 1) % n;
            const u32 o0 = base + i * 2, i0 = o0 + 1;
            const u32 o1 = base + next * 2, i1 = o1 + 1;
            indices_.insert(indices_.end(), {o0, o1, i1, o0, i1, i0});
            batches_.back().index_count += 6;
        }
    }
}

void UiOverlay::textured_rect(f32 x, f32 y, f32 w, f32 h, rhi::TextureHandle tex, const math::Vec4& tint) {
    if (w <= 0.0f || h <= 0.0f) return;
    ensure_batch(tex.valid() ? tex : white_tex_);
    push_quad(x, y, x + w, y + h, 0, 0, 1, 1, tint);
}

void UiOverlay::line(f32 x0, f32 y0, f32 x1, f32 y1, f32 thickness, const math::Vec4& col) {
    const f32 dx = x1 - x0, dy = y1 - y0;
    const f32 len = std::hypot(dx, dy);
    if (len < 0.25f || col.w <= 0.001f) return;
    const f32 nx = -dy / len * (thickness * 0.5f);
    const f32 ny =  dx / len * (thickness * 0.5f);
    ensure_batch(white_tex_);
    const u32 base = static_cast<u32>(vertices_.size());
    vertices_.push_back({x0 + nx, y0 + ny, 0, 0, col.x, col.y, col.z, col.w});
    vertices_.push_back({x1 + nx, y1 + ny, 1, 0, col.x, col.y, col.z, col.w});
    vertices_.push_back({x1 - nx, y1 - ny, 1, 1, col.x, col.y, col.z, col.w});
    vertices_.push_back({x0 - nx, y0 - ny, 0, 1, col.x, col.y, col.z, col.w});
    indices_.insert(indices_.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    batches_.back().index_count += 6;
}

void UiOverlay::circle_filled(f32 cx, f32 cy, f32 r, const math::Vec4& col, int segments) {
    if (r <= 0.25f || col.w <= 0.001f) return;
    ensure_batch(white_tex_);
    const u32 center = static_cast<u32>(vertices_.size());
    vertices_.push_back({cx, cy, 0.5f, 0.5f, col.x, col.y, col.z, col.w});
    for (int i = 0; i < segments; ++i) {
        const f32 a = (static_cast<f32>(i) / segments) * 2.0f * math::kPi;
        vertices_.push_back({cx + std::cos(a) * r, cy + std::sin(a) * r, 0.5f, 0.5f, col.x, col.y, col.z, col.w});
    }
    for (int i = 0; i < segments; ++i) {
        indices_.insert(indices_.end(), {center, center + 1 + static_cast<u32>(i),
                                         center + 1 + static_cast<u32>((i + 1) % segments)});
        batches_.back().index_count += 3;
    }
}

void UiOverlay::circle_outline(f32 cx, f32 cy, f32 r, f32 thickness, const math::Vec4& col, int segments) {
    if (r <= 0.5f || col.w <= 0.001f) return;
    ensure_batch(white_tex_);
    const f32 r0 = std::max(0.25f, r - thickness * 0.5f);
    const f32 r1 = r + thickness * 0.5f;
    const u32 base = static_cast<u32>(vertices_.size());
    for (int i = 0; i < segments; ++i) {
        const f32 a = (static_cast<f32>(i) / segments) * 2.0f * math::kPi;
        const f32 ca = std::cos(a), sa = std::sin(a);
        vertices_.push_back({cx + ca * r1, cy + sa * r1, 0.5f, 0.5f, col.x, col.y, col.z, col.w});
        vertices_.push_back({cx + ca * r0, cy + sa * r0, 0.5f, 0.5f, col.x, col.y, col.z, col.w});
    }
    for (int i = 0; i < segments; ++i) {
        const u32 next = static_cast<u32>((i + 1) % segments);
        const u32 o0 = base + static_cast<u32>(i) * 2, i0 = o0 + 1;
        const u32 o1 = base + next * 2, i1 = o1 + 1;
        indices_.insert(indices_.end(), {o0, o1, i1, o0, i1, i0});
        batches_.back().index_count += 6;
    }
}

void UiOverlay::triangle(f32 x0, f32 y0, f32 x1, f32 y1, f32 x2, f32 y2, const math::Vec4& col) {
    if (col.w <= 0.001f) return;
    ensure_batch(white_tex_);
    const u32 base = static_cast<u32>(vertices_.size());
    vertices_.push_back({x0, y0, 0, 0, col.x, col.y, col.z, col.w});
    vertices_.push_back({x1, y1, 1, 0, col.x, col.y, col.z, col.w});
    vertices_.push_back({x2, y2, 0.5f, 1, col.x, col.y, col.z, col.w});
    indices_.insert(indices_.end(), {base, base + 1, base + 2});
    batches_.back().index_count += 3;
}

f32 UiOverlay::measure_text(std::string_view str, f32 size_px) const noexcept {
    if (!has_ttf_) {
        return static_cast<f32>(str.size()) * size_px * 0.62f;
    }
    const f32 sc = size_px / font_base_px_;
    f32 w = 0.0f;
    for (char ch : str) {
        const int idx = static_cast<int>(static_cast<unsigned char>(ch)) - 32;
        if (idx >= 0 && idx < 96) {
            w += glyphs_[idx].advance * sc;
        }
    }
    return w;
}

void UiOverlay::text(f32 x, f32 y, std::string_view str, f32 size_px, const math::Vec4& col) {
    if (str.empty() || col.w <= 0.001f) return;
    if (has_ttf_) {
        ensure_batch(font_tex_);
        const f32 sc = size_px / font_base_px_;
        f32 cx = x;
        const f32 baseline = y + font_ascent_ * sc;
        for (char ch : str) {
            const int idx = static_cast<int>(static_cast<unsigned char>(ch)) - 32;
            if (idx < 0 || idx >= 96) continue;
            const Glyph& g = glyphs_[idx];
            if (g.x1 > g.x0 && g.y1 > g.y0) {
                push_quad(cx + g.x0 * sc, baseline + g.y0 * sc,
                          cx + g.x1 * sc, baseline + g.y1 * sc,
                          g.u0, g.v0, g.u1, g.v1, col);
            }
            cx += g.advance * sc;
        }
        return;
    }

    // Fallback to stb_easy_font if no system TTF was found
    std::string tmp(str);
    static char buf[65536];
    const int nquads = stb_easy_font_print(0, 0, tmp.data(), nullptr, buf, sizeof(buf));
    const f32 sc = size_px / 11.0f;
    ensure_batch(white_tex_);
    const auto* v = reinterpret_cast<const f32*>(buf);
    for (int q = 0; q < nquads; ++q) {
        const f32 x0 = x + v[q * 16 + 0] * sc;
        const f32 y0 = y + v[q * 16 + 1] * sc;
        const f32 x1 = x + v[q * 16 + 8] * sc;
        const f32 y1 = y + v[q * 16 + 9] * sc;
        push_quad(x0, y0, x1, y1, 0, 0, 1, 1, col);
    }
}

void UiOverlay::render(rhi::CommandList& cmd) {
    if (!valid_ || indices_.empty() || batches_.empty()) return;

    const usize vcount = std::min(vertices_.size(), vbuf_cap_);
    const usize icount = std::min(indices_.size(), ibuf_cap_);
    const u32 slot = frame_idx_ % static_cast<u32>(vbufs_.size());
    ++frame_idx_;

    device_.update_buffer(vbufs_[slot], vertices_.data(), vcount * sizeof(UiVertex));
    device_.update_buffer(ibufs_[slot], indices_.data(), icount * sizeof(u32));

    rhi::RenderPassDesc pass;
    pass.back_buffer = true;
    pass.default_depth = false;
    pass.load_color = true; // preserve the rendered 2K 3D scene!
    cmd.begin_render_pass(pass);
    cmd.bind_pipeline(pipeline_);
    cmd.bind_vertex_buffer(vbufs_[slot]);
    cmd.bind_index_buffer(ibufs_[slot]);

    const f32 push[4] = {1.0f / static_cast<f32>(screen_w_), 1.0f / static_cast<f32>(screen_h_), 0.0f, 0.0f};
    cmd.push_constants(push, sizeof(push));

    for (const Batch& b : batches_) {
        if (b.index_count == 0 || b.index_offset >= icount) continue;
        const u32 count = static_cast<u32>(std::min<usize>(b.index_count, icount - b.index_offset));
        const rhi::ResourceBinding res{0, {}, b.texture};
        cmd.bind_resources({&res, 1});
        cmd.draw_indexed(count, b.index_offset);
    }
    cmd.end_render_pass();
}

} // namespace nova::renderer
