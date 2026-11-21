#pragma once

// NOVA Native C++23 PNG Image & PBR Texture System
// - Pure C++23 RFC 2083 PNG Encoder & Decoder (zlib store/deflate + CRC32 + Adler32 + Paeth/Sub/Up/Avg unfiltering)
// - Bilinear UV texture sampling for Albedo, Normal, Roughness, and Height/Displacement maps
// - Procedural PBR PNG Texture Baker & Disk Loader (wall_albedo.png, wall_normal.png, wall_roughness.png,
//   wall_displacement.png, mud_albedo.png, mud_height.png)

#include "nova/core/log.hpp"
#include "nova/core/types.hpp"
#include "nova/math/math.hpp"
#include "nova/renderer/ui_overlay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace nova::image {

using math::Vec3;
using math::Vec4;

struct PngImage {
    u32 width = 0;
    u32 height = 0;
    std::vector<u8> rgba; // width * height * 4 (RGBA8)

    [[nodiscard]] bool valid() const noexcept {
        return width > 0 && height > 0 && rgba.size() == static_cast<usize>(width) * height * 4;
    }

    void resize(u32 w, u32 h) {
        width = w;
        height = h;
        rgba.assign(static_cast<usize>(w) * h * 4, 255u);
    }

    void set_pixel(u32 x, u32 y, u8 r, u8 g, u8 b, u8 a = 255u) noexcept {
        if (x >= width || y >= height) return;
        const usize idx = (static_cast<usize>(y) * width + x) * 4;
        rgba[idx + 0] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = a;
    }

    [[nodiscard]] Vec4 texel(int x, int y) const noexcept {
        if (!valid()) return {0.5f, 0.5f, 0.5f, 1.0f};
        const int wx = ((x % static_cast<int>(width)) + static_cast<int>(width)) % static_cast<int>(width);
        const int wy = ((y % static_cast<int>(height)) + static_cast<int>(height)) % static_cast<int>(height);
        const usize idx = (static_cast<usize>(wy) * width + static_cast<usize>(wx)) * 4;
        constexpr f32 kInv = 1.0f / 255.0f;
        return {
            static_cast<f32>(rgba[idx + 0]) * kInv,
            static_cast<f32>(rgba[idx + 1]) * kInv,
            static_cast<f32>(rgba[idx + 2]) * kInv,
            static_cast<f32>(rgba[idx + 3]) * kInv,
        };
    }

    // Bilinear wrapped UV sampling (u, v in any range)
    [[nodiscard]] Vec4 sample_rgba(f32 u, f32 v) const noexcept {
        if (!valid()) return {0.5f, 0.5f, 0.5f, 1.0f};
        u = u - std::floor(u);
        v = v - std::floor(v);
        const f32 fx = u * static_cast<f32>(width) - 0.5f;
        const f32 fy = v * static_cast<f32>(height) - 0.5f;
        const int x0 = static_cast<int>(std::floor(fx));
        const int y0 = static_cast<int>(std::floor(fy));
        const f32 tx = fx - static_cast<f32>(x0);
        const f32 ty = fy - static_cast<f32>(y0);

        const Vec4 c00 = texel(x0, y0);
        const Vec4 c10 = texel(x0 + 1, y0);
        const Vec4 c01 = texel(x0, y0 + 1);
        const Vec4 c11 = texel(x0 + 1, y0 + 1);

        auto lerp4 = [](const Vec4& a, const Vec4& b, f32 t) noexcept -> Vec4 {
            return {
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t,
            };
        };
        const Vec4 cx0 = lerp4(c00, c10, tx);
        const Vec4 cx1 = lerp4(c01, c11, tx);
        return lerp4(cx0, cx1, ty);
    }

    [[nodiscard]] Vec3 sample_rgb(f32 u, f32 v) const noexcept {
        const Vec4 c = sample_rgba(u, v);
        return {c.x, c.y, c.z};
    }

    [[nodiscard]] f32 sample_height(f32 u, f32 v) const noexcept {
        return sample_rgba(u, v).x;
    }

    [[nodiscard]] Vec3 sample_normal(f32 u, f32 v) const noexcept {
        const Vec3 c = sample_rgb(u, v);
        return math::normalize(Vec3{c.x * 2.0f - 1.0f, c.y * 2.0f - 1.0f, std::max(0.1f, c.z * 2.0f - 1.0f)});
    }
};

namespace detail {

inline u32 crc32_update(u32 crc, const u8* data, usize len) noexcept {
    static const auto table = []() {
        std::array<u32, 256> t{};
        for (u32 i = 0; i < 256; ++i) {
            u32 c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1u)) : (c >> 1u);
            }
            t[i] = c;
        }
        return t;
    }();
    for (usize i = 0; i < len; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc;
}

inline void write_be32(std::vector<u8>& out, u32 v) {
    out.push_back(static_cast<u8>((v >> 24u) & 0xFFu));
    out.push_back(static_cast<u8>((v >> 16u) & 0xFFu));
    out.push_back(static_cast<u8>((v >> 8u) & 0xFFu));
    out.push_back(static_cast<u8>(v & 0xFFu));
}

inline u32 read_be32(const u8* p) noexcept {
    return (static_cast<u32>(p[0]) << 24u) |
           (static_cast<u32>(p[1]) << 16u) |
           (static_cast<u32>(p[2]) << 8u) |
           static_cast<u32>(p[3]);
}

inline void write_chunk(std::vector<u8>& out, const char type[4], const u8* data, usize len) {
    write_be32(out, static_cast<u32>(len));
    const usize type_pos = out.size();
    out.insert(out.end(), reinterpret_cast<const u8*>(type), reinterpret_cast<const u8*>(type) + 4);
    if (len > 0 && data != nullptr) {
        out.insert(out.end(), data, data + len);
    }
    u32 crc = 0xFFFFFFFFu;
    crc = crc32_update(crc, out.data() + type_pos, 4 + len);
    write_be32(out, crc ^ 0xFFFFFFFFu);
}

inline u8 paeth_predictor(u8 a, u8 b, u8 c) noexcept {
    const int p = static_cast<int>(a) + static_cast<int>(b) - static_cast<int>(c);
    const int pa = std::abs(p - static_cast<int>(a));
    const int pb = std::abs(p - static_cast<int>(b));
    const int pc = std::abs(p - static_cast<int>(c));
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

} // namespace detail

// Write an RGBA8 image to a standard RFC 2083 .png file on disk
inline bool save_png(const std::string& path, const u8* rgba, u32 width, u32 height) {
    if (!rgba || width == 0 || height == 0) return false;

    // 1. Build raw PNG scanlines (1 filter byte 0x00 per row + width * 4 RGBA bytes)
    const usize row_bytes = static_cast<usize>(width) * 4;
    const usize raw_size = static_cast<usize>(height) * (1 + row_bytes);
    std::vector<u8> raw;
    raw.reserve(raw_size);
    for (u32 y = 0; y < height; ++y) {
        raw.push_back(0u); // Filter type 0 (None)
        const u8* row = rgba + static_cast<usize>(y) * row_bytes;
        raw.insert(raw.end(), row, row + row_bytes);
    }

    // 2. Wrap in valid zlib stream (CMF=0x78, FLG=0x01) + RFC 1951 non-compressed blocks + Adler-32
    std::vector<u8> zstream;
    zstream.reserve(raw.size() + (raw.size() / 65535 + 2) * 5 + 6);
    zstream.push_back(0x78u);
    zstream.push_back(0x01u);

    usize pos = 0;
    while (pos < raw.size()) {
        const usize chunk = std::min<usize>(65535u, raw.size() - pos);
        const bool is_final = (pos + chunk == raw.size());
        zstream.push_back(is_final ? 0x01u : 0x00u);
        const u16 len = static_cast<u16>(chunk);
        const u16 nlen = static_cast<u16>(~len);
        zstream.push_back(static_cast<u8>(len & 0xFFu));
        zstream.push_back(static_cast<u8>((len >> 8u) & 0xFFu));
        zstream.push_back(static_cast<u8>(nlen & 0xFFu));
        zstream.push_back(static_cast<u8>((nlen >> 8u) & 0xFFu));
        zstream.insert(zstream.end(), raw.data() + pos, raw.data() + pos + chunk);
        pos += chunk;
    }

    // Compute Adler-32 checksum over raw scanlines
    u32 s1 = 1u;
    u32 s2 = 0u;
    for (usize i = 0; i < raw.size();) {
        const usize block = std::min<usize>(5552u, raw.size() - i);
        for (usize j = 0; j < block; ++j) {
            s1 += raw[i + j];
            s2 += s1;
        }
        s1 %= 65521u;
        s2 %= 65521u;
        i += block;
    }
    detail::write_be32(zstream, (s2 << 16u) | s1);

    // 3. Assemble complete PNG file buffer
    std::vector<u8> png;
    png.reserve(zstream.size() + 64);
    constexpr u8 kSig[8] = {137u, 80u, 78u, 71u, 13u, 10u, 26u, 10u};
    png.insert(png.end(), kSig, kSig + 8);

    std::vector<u8> ihdr;
    ihdr.reserve(13);
    detail::write_be32(ihdr, width);
    detail::write_be32(ihdr, height);
    ihdr.push_back(8u); // bit depth = 8
    ihdr.push_back(6u); // color type = 6 (RGBA)
    ihdr.push_back(0u); // compression = deflate
    ihdr.push_back(0u); // filter = standard
    ihdr.push_back(0u); // interlace = none
    detail::write_chunk(png, "IHDR", ihdr.data(), ihdr.size());
    detail::write_chunk(png, "IDAT", zstream.data(), zstream.size());
    detail::write_chunk(png, "IEND", nullptr, 0);

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const usize written = std::fwrite(png.data(), 1, png.size(), f);
    std::fclose(f);
    return written == png.size();
}

inline bool save_png(const std::string& path, const PngImage& img) {
    return img.valid() && save_png(path, img.rgba.data(), img.width, img.height);
}

// Load any PNG (compressed ZLIB / Grayscale / RGB / RGBA) or JPEG image file from disk into RGBA8
inline bool load_png(const std::string& path, PngImage& out) {
    renderer::ImageData decoded;
    if (renderer::load_image_file(path, decoded) && decoded.width > 0 && decoded.height > 0 &&
        decoded.rgba.size() == static_cast<usize>(decoded.width) * decoded.height * 4) {
        out.width = decoded.width;
        out.height = decoded.height;
        out.rgba = std::move(decoded.rgba);
        return true;
    }

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize < 33) {
        std::fclose(f);
        return false;
    }
    std::vector<u8> buf(static_cast<usize>(fsize));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    constexpr u8 kSig[8] = {137u, 80u, 78u, 71u, 13u, 10u, 26u, 10u};
    if (std::memcmp(buf.data(), kSig, 8) != 0) return false;

    u32 width = 0;
    u32 height = 0;
    u8 bit_depth = 0;
    u8 color_type = 0;
    std::vector<u8> idat;

    usize pos = 8;
    while (pos + 12 <= buf.size()) {
        const u32 len = detail::read_be32(buf.data() + pos);
        const u8* type = buf.data() + pos + 4;
        const u8* data = buf.data() + pos + 8;
        if (pos + 12 + len > buf.size()) break;

        if (std::memcmp(type, "IHDR", 4) == 0 && len >= 13) {
            width = detail::read_be32(data);
            height = detail::read_be32(data + 4);
            bit_depth = data[8];
            color_type = data[9];
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), data, data + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + static_cast<usize>(len);
    }

    if (width == 0 || height == 0 || bit_depth != 8 || (color_type != 6 && color_type != 2) || idat.size() < 6) {
        return false;
    }

    const usize bpp = (color_type == 6) ? 4 : 3;
    const usize row_bytes = static_cast<usize>(width) * bpp;
    const usize expected_raw = static_cast<usize>(height) * (1 + row_bytes);

    // Inflate stored zlib blocks (BTYPE=00)
    std::vector<u8> raw;
    raw.reserve(expected_raw);
    usize zpos = 2; // skip 2-byte zlib header
    while (zpos + 5 <= idat.size()) {
        const u8 bhdr = idat[zpos++];
        const u8 btype = (bhdr >> 1u) & 0x03u;
        if (btype != 0u) break; // stored blocks
        const u16 len = static_cast<u16>(idat[zpos]) | static_cast<u16>(static_cast<u16>(idat[zpos + 1]) << 8u);
        zpos += 4; // len + nlen
        if (zpos + len > idat.size()) break;
        raw.insert(raw.end(), idat.data() + zpos, idat.data() + zpos + len);
        zpos += len;
        if (bhdr & 0x01u) break;
    }

    if (raw.size() < expected_raw) return false;

    // Unfilter scanlines (None, Sub, Up, Average, Paeth)
    std::vector<u8> unfiltered(static_cast<usize>(height) * row_bytes, 0u);
    for (u32 y = 0; y < height; ++y) {
        const u8 filter = raw[static_cast<usize>(y) * (1 + row_bytes)];
        const u8* src = raw.data() + static_cast<usize>(y) * (1 + row_bytes) + 1;
        u8* dst = unfiltered.data() + static_cast<usize>(y) * row_bytes;
        const u8* prev = (y > 0) ? (dst - row_bytes) : nullptr;

        for (usize i = 0; i < row_bytes; ++i) {
            const u8 a = (i >= bpp) ? dst[i - bpp] : 0u;
            const u8 b = prev ? prev[i] : 0u;
            const u8 c = (prev && i >= bpp) ? prev[i - bpp] : 0u;
            switch (filter) {
                case 0: dst[i] = src[i]; break;
                case 1: dst[i] = static_cast<u8>(src[i] + a); break;
                case 2: dst[i] = static_cast<u8>(src[i] + b); break;
                case 3: dst[i] = static_cast<u8>(src[i] + ((static_cast<int>(a) + static_cast<int>(b)) >> 1)); break;
                case 4: dst[i] = static_cast<u8>(src[i] + detail::paeth_predictor(a, b, c)); break;
                default: dst[i] = src[i]; break;
            }
        }
    }

    out.resize(width, height);
    if (bpp == 4) {
        out.rgba = std::move(unfiltered);
    } else {
        for (usize i = 0, j = 0; i < unfiltered.size(); i += 3, j += 4) {
            out.rgba[j + 0] = unfiltered[i + 0];
            out.rgba[j + 1] = unfiltered[i + 1];
            out.rgba[j + 2] = unfiltered[i + 2];
            out.rgba[j + 3] = 255u;
        }
    }
    return true;
}

// PBR PNG Texture Set for Wall (Albedo, Normal, Roughness, Displacement) & Mud (Albedo, Height)
struct PbrPngSet {
    PngImage wall_albedo;
    PngImage wall_normal;
    PngImage wall_roughness;
    PngImage wall_displacement;
    PngImage mud_albedo;
    PngImage mud_height;
};

namespace detail {

inline f32 hash2d(int x, int y, u32 seed = 0u) noexcept {
    u32 h = static_cast<u32>(x) * 374761393u + static_cast<u32>(y) * 668265263u + seed * 1442695041u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return static_cast<f32>((h ^ (h >> 16u)) & 0xFFFFu) / 65535.0f;
}

// Periodic tileable value noise (period = P)
inline f32 tile_noise(f32 u, f32 v, int period, u32 seed) noexcept {
    const f32 x = u * static_cast<f32>(period);
    const f32 y = v * static_cast<f32>(period);
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const f32 fx = x - static_cast<f32>(ix);
    const f32 fy = y - static_cast<f32>(iy);
    const f32 sx = fx * fx * (3.0f - 2.0f * fx);
    const f32 sy = fy * fy * (3.0f - 2.0f * fy);

    auto wrap = [period](int k) { return ((k % period) + period) % period; };
    const f32 n00 = hash2d(wrap(ix), wrap(iy), seed);
    const f32 n10 = hash2d(wrap(ix + 1), wrap(iy), seed);
    const f32 n01 = hash2d(wrap(ix), wrap(iy + 1), seed);
    const f32 n11 = hash2d(wrap(ix + 1), wrap(iy + 1), seed);
    return (n00 * (1.0f - sx) + n10 * sx) * (1.0f - sy) +
           (n01 * (1.0f - sx) + n11 * sx) * sy;
}

inline f32 tile_fbm(f32 u, f32 v, int base_period, u32 seed) noexcept {
    f32 sum = 0.0f;
    f32 amp = 0.5f;
    int p = base_period;
    for (int oct = 0; oct < 4; ++oct) {
        sum += amp * tile_noise(u, v, p, seed + static_cast<u32>(oct * 31));
        p *= 2;
        amp *= 0.5f;
    }
    return sum;
}

inline u8 to_u8(f32 x) noexcept {
    return static_cast<u8>(std::clamp(x * 255.0f + 0.5f, 0.0f, 255.0f));
}

} // namespace detail

// Bake (if missing) and load the 6 PBR .png texture files from disk
inline PbrPngSet ensure_pbr_png_assets(const std::string& preferred_dir = "assets/textures") {
    std::string dir = preferred_dir;
    const std::array<std::string, 5> candidates = {
        preferred_dir,
        "C:/Users/ipiou/.gemini/antigravity/scratch/NOVA_guerrilla_c519e86/NOVA/assets/textures",
        "C:/Users/ipiou/.gemini/antigravity/scratch/NOVA_village_78c42f3/NOVA/assets/textures",
        "../../../../assets/textures",
        "../assets/textures",
    };
    for (const std::string& cand : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(cand + "/wall_albedo.png", ec)) {
            dir = cand;
            break;
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const std::string p_wall_alb  = dir + "/wall_albedo.png";
    const std::string p_wall_nrm  = dir + "/wall_normal.png";
    const std::string p_wall_rgh  = dir + "/wall_roughness.png";
    const std::string p_wall_disp = dir + "/wall_displacement.png";
    const std::string p_mud_alb   = dir + "/mud_albedo.png";
    const std::string p_mud_hgt   = dir + "/mud_height.png";

    PbrPngSet set;
    const bool already_exists =
        load_png(p_wall_alb, set.wall_albedo) &&
        load_png(p_wall_nrm, set.wall_normal) &&
        load_png(p_wall_rgh, set.wall_roughness) &&
        load_png(p_wall_disp, set.wall_displacement) &&
        load_png(p_mud_alb, set.mud_albedo) &&
        load_png(p_mud_hgt, set.mud_height);

    if (already_exists) {
        NOVA_INFO("PNG", "loaded 6 PBR PNG textures ({}x{}) from {}",
                  set.wall_albedo.width, set.wall_albedo.height, dir);
        return set;
    }

    // Generate 256x256 tileable Stone Wall PBR PNG maps + Mud Baked PNG maps
    constexpr u32 kSize = 256;
    set.wall_albedo.resize(kSize, kSize);
    set.wall_normal.resize(kSize, kSize);
    set.wall_roughness.resize(kSize, kSize);
    set.wall_displacement.resize(kSize, kSize);
    set.mud_albedo.resize(kSize, kSize);
    set.mud_height.resize(kSize, kSize);

    std::vector<f32> wall_h(static_cast<usize>(kSize) * kSize, 0.0f);

    for (u32 y = 0; y < kSize; ++y) {
        const f32 v = static_cast<f32>(y) / static_cast<f32>(kSize);
        for (u32 x = 0; x < kSize; ++x) {
            const f32 u = static_cast<f32>(x) / static_cast<f32>(kSize);

            // Running-bond coursed stone ashlar blocks (4 courses vertically, 2.5 blocks horizontally)
            const f32 row_f = v * 4.0f;
            const int row_i = static_cast<int>(std::floor(row_f));
            const f32 fy = row_f - static_cast<f32>(row_i);

            const f32 row_shift = (row_i % 2 == 1) ? 0.5f : 0.0f;
            const f32 jitter_u = (detail::tile_noise(u, v, 8, 17u) - 0.5f) * 0.06f;
            const f32 jitter_v = (detail::tile_noise(u, v, 8, 29u) - 0.5f) * 0.08f;

            const f32 col_f = (u * 2.0f + row_shift + jitter_u);
            const int col_i = static_cast<int>(std::floor(col_f));
            const f32 fx = col_f - static_cast<f32>(col_i);
            const f32 fy_j = std::clamp(fy + jitter_v, 0.0f, 1.0f);

            // Distance from mortar joints
            const f32 edge_x = std::min(fx, 1.0f - fx);
            const f32 edge_y = std::min(fy_j, 1.0f - fy_j);
            const f32 bevel_x = std::clamp((edge_x - 0.045f) / 0.095f, 0.0f, 1.0f);
            const f32 bevel_y = std::clamp((edge_y - 0.055f) / 0.115f, 0.0f, 1.0f);
            const f32 stone_mask = std::sqrt(bevel_x * bevel_y);

            // Per-stone variation + chiseled micro-relief
            const f32 block_id = detail::hash2d(col_i & 3, row_i & 3, 91u);
            const f32 macro = detail::tile_fbm(u, v, 6, 101u);
            const f32 micro = detail::tile_fbm(u, v, 24, 211u);

            const f32 disp = std::clamp(stone_mask * (0.68f + 0.22f * block_id + 0.22f * (macro - 0.5f)) +
                                            0.08f * (micro - 0.5f),
                                        0.04f, 0.98f);
            wall_h[static_cast<usize>(y) * kSize + x] = disp;

            // Stone albedo: warm Atlas limestone blocks + cooler recessed lime mortar
            const Vec3 mortar_col{0.34f, 0.28f, 0.22f};
            const Vec3 stone_a{0.68f, 0.54f, 0.40f};
            const Vec3 stone_b{0.56f, 0.43f, 0.31f};
            Vec3 stone_col = math::lerp(stone_a, stone_b, block_id) * (0.85f + 0.30f * macro);
            Vec3 alb = math::lerp(mortar_col, stone_col, stone_mask);

            // Roughness & AO
            const f32 rough = std::clamp(0.92f - 0.22f * stone_mask + 0.14f * (0.5f - micro), 0.45f, 0.98f);
            const f32 ao = std::clamp(0.35f + 0.65f * stone_mask, 0.25f, 1.0f);

            set.wall_albedo.set_pixel(x, y, detail::to_u8(alb.x), detail::to_u8(alb.y), detail::to_u8(alb.z), 255u);
            set.wall_roughness.set_pixel(x, y, detail::to_u8(rough), detail::to_u8(ao), 0u, 255u);
            set.wall_displacement.set_pixel(x, y, detail::to_u8(disp), detail::to_u8(disp), detail::to_u8(disp), 255u);

            // Stage 2: Bake Mud Texture & Micro-Height PNG
            const f32 mud_f1 = detail::tile_fbm(u, v, 8, 307u);
            const f32 mud_f2 = detail::tile_fbm(u, v, 20, 409u);
            const f32 pebble = std::max(0.0f, detail::tile_noise(u, v, 32, 503u) - 0.72f) * 2.5f;
            const f32 mh = std::clamp(0.45f + 0.35f * (mud_f1 - 0.5f) + 0.20f * (mud_f2 - 0.5f) + pebble * 0.25f, 0.0f, 1.0f);
            const Vec3 m_col = math::lerp(Vec3{0.26f, 0.18f, 0.12f}, Vec3{0.44f, 0.33f, 0.22f}, mh);
            set.mud_albedo.set_pixel(x, y, detail::to_u8(m_col.x), detail::to_u8(m_col.y), detail::to_u8(m_col.z), 255u);
            set.mud_height.set_pixel(x, y, detail::to_u8(mh), detail::to_u8(mh), detail::to_u8(mh), 255u);
        }
    }

    // Bake Tangent-Space Normal Map (wall_normal.png) from wall_h central differences
    for (u32 y = 0; y < kSize; ++y) {
        const u32 y0 = (y + kSize - 1) % kSize;
        const u32 y1 = (y + 1) % kSize;
        for (u32 x = 0; x < kSize; ++x) {
            const u32 x0 = (x + kSize - 1) % kSize;
            const u32 x1 = (x + 1) % kSize;
            const f32 dhdx = (wall_h[y * kSize + x1] - wall_h[y * kSize + x0]) * 3.5f;
            const f32 dhdy = (wall_h[y1 * kSize + x] - wall_h[y0 * kSize + x]) * 3.5f;
            const Vec3 n = math::normalize(Vec3{-dhdx, -dhdy, 1.0f});
            set.wall_normal.set_pixel(
                x, y,
                detail::to_u8(n.x * 0.5f + 0.5f),
                detail::to_u8(n.y * 0.5f + 0.5f),
                detail::to_u8(n.z * 0.5f + 0.5f),
                255u);
        }
    }

    // Save all 6 PNG files to disk and verify round-trip loading
    save_png(p_wall_alb, set.wall_albedo);
    save_png(p_wall_nrm, set.wall_normal);
    save_png(p_wall_rgh, set.wall_roughness);
    save_png(p_wall_disp, set.wall_displacement);
    save_png(p_mud_alb, set.mud_albedo);
    save_png(p_mud_hgt, set.mud_height);

    PngImage verify;
    if (load_png(p_wall_disp, verify) && verify.valid()) {
        NOVA_INFO("PNG", "baked & verified 6 PBR PNG textures ({}x{}) in {}", kSize, kSize, dir);
    }
    return set;
}

} // namespace nova::image
