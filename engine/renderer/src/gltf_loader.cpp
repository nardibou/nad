#include "nova/renderer/model.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>

namespace nova::renderer {
namespace {

using math::Mat4;
using math::Vec2;
using math::Vec3;
using math::Vec4;

struct GltfDeleter {
    void operator()(cgltf_data* d) const noexcept { cgltf_free(d); }
};

bool decode_image(const cgltf_image& img, const std::filesystem::path& dir, ImageData& out) {
    std::vector<u8> file_bytes;
    const u8* bytes = nullptr;
    std::size_t size = 0;
    void* owned = nullptr; // cgltf-allocated base64 result

    if (img.buffer_view != nullptr) {
        const u8* base = static_cast<const u8*>(cgltf_buffer_view_data(img.buffer_view));
        if (base == nullptr) return false;
        bytes = base;
        size = img.buffer_view->size;
    } else if (img.uri != nullptr) {
        const std::string uri = img.uri;
        if (uri.rfind("data:", 0) == 0) {
            const auto comma = uri.find(',');
            if (comma == std::string::npos) return false;
            cgltf_options opts{};
            const std::string payload = uri.substr(comma + 1);
            const std::size_t padding = (payload.size() >= 1 && payload.back() == '=' ? 1u : 0u) +
                                        (payload.size() >= 2 && payload[payload.size() - 2] == '=' ? 1u : 0u);
            const std::size_t decoded = payload.size() / 4 * 3 - padding;
            if (cgltf_load_buffer_base64(&opts, decoded, payload.c_str(), &owned) != cgltf_result_success) return false;
            bytes = static_cast<const u8*>(owned);
            size = decoded;
        } else {
            std::string decoded_uri = uri;
            cgltf_decode_uri(decoded_uri.data());
            decoded_uri.resize(std::strlen(decoded_uri.c_str()));
            std::ifstream f(dir / std::filesystem::path(decoded_uri), std::ios::binary);
            if (!f) return false;
            file_bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            bytes = file_bytes.data();
            size = file_bytes.size();
        }
    } else {
        return false;
    }

    int w = 0, h = 0, comp = 0;
    stbi_uc* px = size > 0 && size <= 0x7fffffffu
                      ? stbi_load_from_memory(bytes, static_cast<int>(size), &w, &h, &comp, 4)
                      : nullptr;
    if (owned != nullptr) std::free(owned);
    if (px == nullptr) return false;
    out.width = static_cast<u32>(w);
    out.height = static_cast<u32>(h);
    out.rgba.assign(px, px + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    stbi_image_free(px);
    return true;
}

i32 image_slot(const cgltf_data& data, const cgltf_texture_view& view, const std::vector<bool>& ok) {
    if (view.texture == nullptr || view.texture->image == nullptr) return -1;
    const auto idx = static_cast<std::size_t>(view.texture->image - data.images);
    return idx < ok.size() && ok[idx] ? static_cast<i32>(idx) : -1;
}

MaterialData convert_material(const cgltf_data& data, const cgltf_material& m, const std::vector<bool>& ok) {
    MaterialData r;
    if (m.has_pbr_metallic_roughness) {
        const auto& p = m.pbr_metallic_roughness;
        r.base_color_factor = {p.base_color_factor[0], p.base_color_factor[1], p.base_color_factor[2],
                               p.base_color_factor[3]};
        r.metallic_factor = p.metallic_factor;
        r.roughness_factor = p.roughness_factor;
        r.base_color_image = image_slot(data, p.base_color_texture, ok);
        r.metallic_roughness_image = image_slot(data, p.metallic_roughness_texture, ok);
    }
    r.emissive_factor = {m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]};
    if (m.has_emissive_strength) {
        r.emissive_factor = {r.emissive_factor.x * m.emissive_strength.emissive_strength,
                             r.emissive_factor.y * m.emissive_strength.emissive_strength,
                             r.emissive_factor.z * m.emissive_strength.emissive_strength};
    }
    r.normal_scale = m.normal_texture.texture != nullptr ? m.normal_texture.scale : 1.0f;
    r.occlusion_strength = m.occlusion_texture.texture != nullptr ? m.occlusion_texture.scale : 1.0f;
    r.normal_image = image_slot(data, m.normal_texture, ok);
    r.occlusion_image = image_slot(data, m.occlusion_texture, ok);
    r.emissive_image = image_slot(data, m.emissive_texture, ok);
    switch (m.alpha_mode) {
    case cgltf_alpha_mode_mask: r.alpha_mode = AlphaMode::Mask; break;
    case cgltf_alpha_mode_blend: r.alpha_mode = AlphaMode::Blend; break;
    default: r.alpha_mode = AlphaMode::Opaque; break;
    }
    r.alpha_cutoff = m.alpha_cutoff;
    r.double_sided = m.double_sided != 0;
    return r;
}

Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 mul(const Vec3& a, f32 s) { return {a.x * s, a.y * s, a.z * s}; }
Vec3 normalized_or(const Vec3& v, const Vec3& fallback) {
    const f32 l = std::sqrt(dot(v, v));
    return l > 1e-20f ? mul(v, 1.0f / l) : fallback;
}

void generate_normals(MeshData& mesh) {
    std::vector<Vec3> acc(mesh.vertices.size(), Vec3{0.0f, 0.0f, 0.0f});
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const u32 a = mesh.indices[i], b = mesh.indices[i + 1], c = mesh.indices[i + 2];
        // Unnormalised cross product is area-weighted.
        const Vec3 n = cross(sub(mesh.vertices[b].position, mesh.vertices[a].position),
                             sub(mesh.vertices[c].position, mesh.vertices[a].position));
        acc[a] = add(acc[a], n);
        acc[b] = add(acc[b], n);
        acc[c] = add(acc[c], n);
    }
    for (std::size_t i = 0; i < acc.size(); ++i) mesh.vertices[i].normal = normalized_or(acc[i], Vec3{0.0f, 1.0f, 0.0f});
}

void generate_tangents(MeshData& mesh) {
    const std::size_t n = mesh.vertices.size();
    std::vector<Vec3> tan(n, Vec3{0.0f, 0.0f, 0.0f}), bit(n, Vec3{0.0f, 0.0f, 0.0f});
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const u32 ia = mesh.indices[i], ib = mesh.indices[i + 1], ic = mesh.indices[i + 2];
        const Vertex &a = mesh.vertices[ia], &b = mesh.vertices[ib], &c = mesh.vertices[ic];
        const Vec3 e1 = sub(b.position, a.position), e2 = sub(c.position, a.position);
        const f32 du1 = b.uv.x - a.uv.x, dv1 = b.uv.y - a.uv.y;
        const f32 du2 = c.uv.x - a.uv.x, dv2 = c.uv.y - a.uv.y;
        const f32 det = du1 * dv2 - du2 * dv1;
        if (std::fabs(det) < 1e-20f) continue;
        const f32 r = 1.0f / det;
        const Vec3 t = mul(sub(mul(e1, dv2), mul(e2, dv1)), r);
        const Vec3 s = mul(sub(mul(e2, du1), mul(e1, du2)), r);
        for (const u32 v : {ia, ib, ic}) {
            tan[v] = add(tan[v], t);
            bit[v] = add(bit[v], s);
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        Vertex& v = mesh.vertices[i];
        const Vec3 nrm = v.normal;
        Vec3 t = sub(tan[i], mul(nrm, dot(nrm, tan[i])));
        if (dot(t, t) < 1e-20f) {
            // Degenerate uv mapping: any vector orthogonal to the normal.
            const Vec3 axis = std::fabs(nrm.x) < 0.9f ? Vec3{1.0f, 0.0f, 0.0f} : Vec3{0.0f, 1.0f, 0.0f};
            t = sub(axis, mul(nrm, dot(nrm, axis)));
        }
        t = normalized_or(t, Vec3{1.0f, 0.0f, 0.0f});
        const f32 w = dot(cross(nrm, t), bit[i]) < 0.0f ? -1.0f : 1.0f;
        v.tangent = {t.x, t.y, t.z, w};
    }
}

} // namespace

std::optional<ModelData> load_gltf(const std::filesystem::path& path, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<ModelData> {
        if (error != nullptr) *error = msg;
        return std::nullopt;
    };

    const std::string path_str = path.string();
    cgltf_options options{};
    cgltf_data* raw = nullptr;
    cgltf_result res = cgltf_parse_file(&options, path_str.c_str(), &raw);
    if (res != cgltf_result_success) return fail("glTF parse failed for '" + path_str + "' (cgltf result " + std::to_string(static_cast<int>(res)) + ")");
    std::unique_ptr<cgltf_data, GltfDeleter> data(raw);
    res = cgltf_load_buffers(&options, raw, path_str.c_str());
    if (res != cgltf_result_success) return fail("glTF buffer load failed for '" + path_str + "' (cgltf result " + std::to_string(static_cast<int>(res)) + ")");
    res = cgltf_validate(raw);
    if (res != cgltf_result_success) return fail("glTF validation failed for '" + path_str + "' (cgltf result " + std::to_string(static_cast<int>(res)) + ")");

    ModelData model;
    const std::filesystem::path dir = path.parent_path();

    model.images.resize(raw->images_count);
    std::vector<bool> image_ok(raw->images_count, false);
    for (std::size_t i = 0; i < raw->images_count; ++i) image_ok[i] = decode_image(raw->images[i], dir, model.images[i]);

    model.materials.reserve(raw->materials_count);
    for (std::size_t i = 0; i < raw->materials_count; ++i)
        model.materials.push_back(convert_material(*raw, raw->materials[i], image_ok));

    bool have_bounds = false;
    Vec3 bmin{0.0f, 0.0f, 0.0f}, bmax{0.0f, 0.0f, 0.0f};

    for (std::size_t ni = 0; ni < raw->nodes_count; ++ni) {
        const cgltf_node& node = raw->nodes[ni];
        if (node.mesh == nullptr) continue;
        f32 world[16];
        cgltf_node_transform_world(&node, world);
        Mat4 xf;
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) xf.m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = world[c * 4 + r];

        for (std::size_t pi = 0; pi < node.mesh->primitives_count; ++pi) {
            const cgltf_primitive& prim = node.mesh->primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;

            const cgltf_accessor *pos = nullptr, *nrm = nullptr, *uv = nullptr, *tan = nullptr, *col = nullptr;
            for (std::size_t ai = 0; ai < prim.attributes_count; ++ai) {
                const cgltf_attribute& a = prim.attributes[ai];
                if (a.type == cgltf_attribute_type_position && a.index == 0) pos = a.data;
                else if (a.type == cgltf_attribute_type_normal && a.index == 0) nrm = a.data;
                else if (a.type == cgltf_attribute_type_texcoord && a.index == 0) uv = a.data;
                else if (a.type == cgltf_attribute_type_tangent && a.index == 0) tan = a.data;
                else if (a.type == cgltf_attribute_type_color && a.index == 0) col = a.data;
            }
            if (pos == nullptr) return fail("glTF primitive without POSITION in '" + path_str + "'");

            ModelPrimitive out;
            out.transform = xf;
            out.node = static_cast<i32>(ni);
            if (prim.material != nullptr) out.material = static_cast<i32>(prim.material - raw->materials);
            if (node.skin != nullptr) out.skin = static_cast<i32>(node.skin - raw->skins);

            const cgltf_accessor *joints = nullptr, *weights = nullptr;
            for (std::size_t ai = 0; ai < prim.attributes_count; ++ai) {
                const cgltf_attribute& a = prim.attributes[ai];
                if (a.type == cgltf_attribute_type_joints && a.index == 0) joints = a.data;
                else if (a.type == cgltf_attribute_type_weights && a.index == 0) weights = a.data;
            }
            if (out.skin >= 0 && joints != nullptr && weights != nullptr) {
                const std::size_t joint_count = node.skin->joints_count;
                out.joints.resize(pos->count);
                out.weights.resize(pos->count);
                for (std::size_t v = 0; v < pos->count; ++v) {
                    cgltf_uint j[4] = {};
                    f32 w[4] = {};
                    if (v < joints->count) cgltf_accessor_read_uint(joints, v, j, 4);
                    if (v < weights->count) cgltf_accessor_read_float(weights, v, w, 4);
                    f32 sum = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        if (j[k] >= joint_count) w[k] = 0.0f; // invalid joint: drop the influence
                        sum += w[k];
                    }
                    if (sum <= 0.0f) {
                        w[0] = 1.0f;
                        j[0] = 0;
                        sum = 1.0f;
                    }
                    for (int k = 0; k < 4; ++k) out.joints[v][static_cast<std::size_t>(k)] = static_cast<u16>(j[k] < joint_count ? j[k] : 0);
                    out.weights[v] = {w[0] / sum, w[1] / sum, w[2] / sum, w[3] / sum};
                }
            } else {
                out.skin = -1;
            }

            out.mesh.vertices.resize(pos->count);
            for (std::size_t v = 0; v < pos->count; ++v) {
                Vertex& vert = out.mesh.vertices[v];
                f32 f[4] = {};
                cgltf_accessor_read_float(pos, v, f, 3);
                vert.position = {f[0], f[1], f[2]};
                vert.normal = {0.0f, 1.0f, 0.0f};
                vert.uv = {0.0f, 0.0f};
                vert.tangent = {0.0f, 0.0f, 0.0f, 0.0f};
                vert.color = {1.0f, 1.0f, 1.0f, 0.0f};
                if (nrm != nullptr && v < nrm->count) {
                    cgltf_accessor_read_float(nrm, v, f, 3);
                    vert.normal = normalized_or({f[0], f[1], f[2]}, Vec3{0.0f, 1.0f, 0.0f});
                }
                if (uv != nullptr && v < uv->count) {
                    cgltf_accessor_read_float(uv, v, f, 2);
                    vert.uv = {f[0], f[1]};
                }
                if (tan != nullptr && v < tan->count) {
                    cgltf_accessor_read_float(tan, v, f, 4);
                    vert.tangent = {f[0], f[1], f[2], f[3]};
                }
                if (col != nullptr && v < col->count) {
                    f32 c[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                    cgltf_accessor_read_float(col, v, c, cgltf_num_components(col->type));
                    vert.color = {c[0], c[1], c[2], 0.0f};
                }
            }

            if (prim.indices != nullptr) {
                out.mesh.indices.resize(prim.indices->count);
                for (std::size_t i = 0; i < prim.indices->count; ++i) {
                    const auto idx = cgltf_accessor_read_index(prim.indices, i);
                    if (idx >= pos->count) return fail("glTF index out of range in '" + path_str + "'");
                    out.mesh.indices[i] = static_cast<u32>(idx);
                }
            } else {
                out.mesh.indices.resize(pos->count);
                for (std::size_t i = 0; i < pos->count; ++i) out.mesh.indices[i] = static_cast<u32>(i);
            }
            out.mesh.indices.resize(out.mesh.indices.size() / 3 * 3);

            if (nrm == nullptr) generate_normals(out.mesh);
            if (tan == nullptr && uv != nullptr) generate_tangents(out.mesh);

            for (const Vertex& v : out.mesh.vertices) {
                const Vec3 p = math::transform_point(xf, v.position);
                if (!have_bounds) {
                    bmin = bmax = p;
                    have_bounds = true;
                } else {
                    bmin = {std::fmin(bmin.x, p.x), std::fmin(bmin.y, p.y), std::fmin(bmin.z, p.z)};
                    bmax = {std::fmax(bmax.x, p.x), std::fmax(bmax.y, p.y), std::fmax(bmax.z, p.z)};
                }
            }
            model.primitives.push_back(std::move(out));
        }
    }
    model.bounds_min = bmin;
    model.bounds_max = bmax;

    // ---- scene graph (rest pose), skins and animations
    model.nodes.resize(raw->nodes_count);
    for (std::size_t ni = 0; ni < raw->nodes_count; ++ni) {
        const cgltf_node& n = raw->nodes[ni];
        NodeData& out = model.nodes[ni];
        out.name = n.name != nullptr ? n.name : "";
        out.parent = n.parent != nullptr ? static_cast<i32>(n.parent - raw->nodes) : -1;
        if (n.has_matrix) {
            Mat4 m;
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) m.m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = n.matrix[c * 4 + r];
            out.translation = {m.m[3][0], m.m[3][1], m.m[3][2]};
            Vec3 cols[3];
            for (int c = 0; c < 3; ++c) cols[c] = {m.m[static_cast<std::size_t>(c)][0], m.m[static_cast<std::size_t>(c)][1], m.m[static_cast<std::size_t>(c)][2]};
            Vec3 s{math::length(cols[0]), math::length(cols[1]), math::length(cols[2])};
            if (math::dot(cols[0], math::cross(cols[1], cols[2])) < 0.0f) s.x = -s.x; // mirrored: fold into one axis
            Mat4 rot = Mat4::identity();
            const f32 sv[3] = {s.x, s.y, s.z};
            for (int c = 0; c < 3; ++c) {
                const f32 inv = sv[c] != 0.0f ? 1.0f / sv[c] : 0.0f;
                for (int r = 0; r < 3; ++r) rot.m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = m.m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] * inv;
            }
            out.rotation = math::to_quat(rot);
            out.scale = s;
        } else {
            if (n.has_translation) out.translation = {n.translation[0], n.translation[1], n.translation[2]};
            if (n.has_rotation) out.rotation = math::normalize(math::Quat{n.rotation[0], n.rotation[1], n.rotation[2], n.rotation[3]});
            if (n.has_scale) out.scale = {n.scale[0], n.scale[1], n.scale[2]};
        }
    }

    model.skins.resize(raw->skins_count);
    for (std::size_t si = 0; si < raw->skins_count; ++si) {
        const cgltf_skin& s = raw->skins[si];
        SkinData& out = model.skins[si];
        out.name = s.name != nullptr ? s.name : "";
        out.joints.resize(s.joints_count);
        out.inverse_bind.assign(s.joints_count, Mat4::identity());
        for (std::size_t k = 0; k < s.joints_count; ++k) {
            out.joints[k] = static_cast<i32>(s.joints[k] - raw->nodes);
            if (s.inverse_bind_matrices != nullptr && k < s.inverse_bind_matrices->count) {
                f32 f[16] = {};
                cgltf_accessor_read_float(s.inverse_bind_matrices, k, f, 16);
                for (int c = 0; c < 4; ++c)
                    for (int r = 0; r < 4; ++r) out.inverse_bind[k].m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = f[c * 4 + r];
            }
        }
    }

    model.animations.reserve(raw->animations_count);
    for (std::size_t ai = 0; ai < raw->animations_count; ++ai) {
        const cgltf_animation& a = raw->animations[ai];
        AnimationData out;
        out.name = a.name != nullptr ? a.name : ("animation_" + std::to_string(ai));
        for (std::size_t ci = 0; ci < a.channels_count; ++ci) {
            const cgltf_animation_channel& ch = a.channels[ci];
            if (ch.target_node == nullptr || ch.sampler == nullptr || ch.sampler->input == nullptr || ch.sampler->output == nullptr) continue;
            AnimChannel c;
            c.node = static_cast<i32>(ch.target_node - raw->nodes);
            int comps = 3;
            switch (ch.target_path) {
            case cgltf_animation_path_type_translation: c.path = AnimPath::Translation; break;
            case cgltf_animation_path_type_rotation: c.path = AnimPath::Rotation; comps = 4; break;
            case cgltf_animation_path_type_scale: c.path = AnimPath::Scale; break;
            default: continue; // morph weights are not supported
            }
            switch (ch.sampler->interpolation) {
            case cgltf_interpolation_type_step: c.interpolation = AnimInterpolation::Step; break;
            case cgltf_interpolation_type_cubic_spline: c.interpolation = AnimInterpolation::CubicSpline; break;
            default: c.interpolation = AnimInterpolation::Linear; break;
            }
            const cgltf_accessor* in = ch.sampler->input;
            const cgltf_accessor* outv = ch.sampler->output;
            c.times.resize(in->count);
            for (std::size_t k = 0; k < in->count; ++k) cgltf_accessor_read_float(in, k, &c.times[k], 1);
            const std::size_t per_key = c.interpolation == AnimInterpolation::CubicSpline ? 3 : 1;
            if (c.times.empty() || outv->count < c.times.size() * per_key) continue;
            c.values.resize(c.times.size() * per_key);
            for (std::size_t k = 0; k < c.values.size(); ++k) {
                f32 f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                cgltf_accessor_read_float(outv, k, f, static_cast<cgltf_size>(comps));
                c.values[k] = {f[0], f[1], f[2], f[3]};
            }
            out.duration = std::max(out.duration, c.times.back());
            out.channels.push_back(std::move(c));
        }
        model.animations.push_back(std::move(out));
    }
    return model;
}

i32 ModelData::find_node(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].name == name) return static_cast<i32>(i);
    }
    return -1;
}

i32 ModelData::find_animation(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < animations.size(); ++i) {
        if (animations[i].name == name) return static_cast<i32>(i);
    }
    // Exporters prefix clip names ("Rig|AK_Idle", "Armature|Walk"): accept a "|name" suffix.
    for (std::size_t i = 0; i < animations.size(); ++i) {
        const std::string& n = animations[i].name;
        if (n.size() > name.size() && n.compare(n.size() - name.size(), name.size(), name) == 0 &&
            n[n.size() - name.size() - 1] == '|') {
            return static_cast<i32>(i);
        }
    }
    return -1;
}

} // namespace nova::renderer
