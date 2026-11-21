#include "nova/renderer/model.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

using namespace nova;
using namespace nova::math;
using namespace nova::renderer;

namespace {

std::string base64(const std::vector<unsigned char>& in) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (std::size_t i = 0; i < in.size(); i += 3) {
        const unsigned v = (static_cast<unsigned>(in[i]) << 16) |
                           (i + 1 < in.size() ? static_cast<unsigned>(in[i + 1]) << 8 : 0u) |
                           (i + 2 < in.size() ? static_cast<unsigned>(in[i + 2]) : 0u);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < in.size() ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? tbl[v & 63] : '=';
    }
    return out;
}

std::vector<unsigned char> quad_buffer() {
    const float verts[4][8] = {
        {0, 0, 0, 0, 0, 1, 0, 1},
        {1, 0, 0, 0, 0, 1, 1, 1},
        {1, 1, 0, 0, 0, 1, 1, 0},
        {0, 1, 0, 0, 0, 1, 0, 0},
    };
    const unsigned short idx[6] = {0, 1, 2, 0, 2, 3};
    std::vector<unsigned char> buf(sizeof(verts) + sizeof(idx));
    std::memcpy(buf.data(), verts, sizeof(verts));
    std::memcpy(buf.data() + sizeof(verts), idx, sizeof(idx));
    return buf;
}

std::string quad_json(const std::string& buffer_uri_field, std::size_t byte_length, const char* image_uri) {
    std::string j = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],)"
                    R"("nodes":[{"mesh":0,"translation":[1,2,3]}],)"
                    R"("meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0}]}],)"
                    R"("materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.5,0.25,0.75,1],"metallicFactor":0.2,"roughnessFactor":0.8,)"
                    R"("baseColorTexture":{"index":0}},"alphaMode":"MASK","alphaCutoff":0.3,"doubleSided":true}],)"
                    R"("textures":[{"source":0}],"images":[{"uri":")";
    j += image_uri;
    j += R"("}],"buffers":[{)" + buffer_uri_field + R"("byteLength":)" + std::to_string(byte_length) + R"(}],)";
    j += R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":128,"byteStride":32},{"buffer":0,"byteOffset":128,"byteLength":12}],)";
    j += R"("accessors":[)"
         R"({"bufferView":0,"byteOffset":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},)"
         R"({"bufferView":0,"byteOffset":12,"componentType":5126,"count":4,"type":"VEC3"},)"
         R"({"bufferView":0,"byteOffset":24,"componentType":5126,"count":4,"type":"VEC2"},)"
         R"({"bufferView":1,"componentType":5123,"count":6,"type":"SCALAR"}]})";
    return j;
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() / "nova_model_tests";
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void write_text(const std::filesystem::path& p, const std::string& s) {
    std::ofstream(p, std::ios::binary) << s;
}

bool near(f32 a, f32 b, f32 eps = 1e-5f) { return std::fabs(a - b) <= eps; }

} // namespace

TEST_CASE("model: load_gltf imports a textured quad with external PNG") {
    TempDir dir;
    const unsigned char pixels[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
    const auto png = (dir.path / "tex.png").string();
    REQUIRE(stbi_write_png(png.c_str(), 2, 2, 4, pixels, 8) != 0);

    const auto buf = quad_buffer();
    const std::string uri = "\"uri\":\"data:application/octet-stream;base64," + base64(buf) + "\",";
    write_text(dir.path / "quad.gltf", quad_json(uri, buf.size(), "tex.png"));

    std::string err;
    auto model = load_gltf(dir.path / "quad.gltf", &err);
    REQUIRE_MESSAGE(model.has_value(), err);

    REQUIRE(model->primitives.size() == 1);
    const auto& prim = model->primitives[0];
    CHECK(prim.mesh.vertices.size() == 4);
    CHECK(prim.mesh.indices.size() == 6);
    CHECK(prim.mesh.indices[4] == 2);
    CHECK(prim.material == 0);

    CHECK(near(prim.mesh.vertices[1].uv.x, 1.0f));
    CHECK(near(prim.mesh.vertices[1].uv.y, 1.0f));
    CHECK(near(prim.mesh.vertices[3].uv.x, 0.0f));
    CHECK(near(prim.mesh.vertices[3].uv.y, 0.0f));

    for (const Vertex& v : prim.mesh.vertices) {
        const Vec3 t{v.tangent.x, v.tangent.y, v.tangent.z};
        CHECK(near(length(t), 1.0f, 1e-4f));
        CHECK(near(dot(t, v.normal), 0.0f, 1e-4f));
        CHECK(std::fabs(v.tangent.w) == 1.0f);
        CHECK(near(v.color.x, 1.0f));
        CHECK(near(v.color.w, 0.0f));
    }

    CHECK(near(prim.transform.m[3][0], 1.0f));
    CHECK(near(prim.transform.m[3][1], 2.0f));
    CHECK(near(prim.transform.m[3][2], 3.0f));
    CHECK(near(model->bounds_min.x, 1.0f));
    CHECK(near(model->bounds_min.y, 2.0f));
    CHECK(near(model->bounds_min.z, 3.0f));
    CHECK(near(model->bounds_max.x, 2.0f));
    CHECK(near(model->bounds_max.y, 3.0f));
    CHECK(near(model->bounds_max.z, 3.0f));

    REQUIRE(model->images.size() == 1);
    const auto& img = model->images[0];
    CHECK(img.width == 2);
    CHECK(img.height == 2);
    REQUIRE(img.rgba.size() == 16);
    CHECK(std::memcmp(img.rgba.data(), pixels, 16) == 0);

    REQUIRE(model->materials.size() == 1);
    const auto& m = model->materials[0];
    CHECK(near(m.base_color_factor.x, 0.5f));
    CHECK(near(m.base_color_factor.y, 0.25f));
    CHECK(near(m.base_color_factor.z, 0.75f));
    CHECK(near(m.metallic_factor, 0.2f));
    CHECK(near(m.roughness_factor, 0.8f));
    CHECK(m.alpha_mode == AlphaMode::Mask);
    CHECK(near(m.alpha_cutoff, 0.3f));
    CHECK(m.double_sided);
    CHECK(m.base_color_image == 0);
    CHECK(m.normal_image == -1);
}

TEST_CASE("model: load_gltf round-trips a .glb with embedded buffer") {
    TempDir dir;
    const auto bin = quad_buffer();
    std::string json = quad_json("", bin.size(), "missing.png");
    while (json.size() % 4 != 0) json += ' ';
    std::vector<unsigned char> bin_padded = bin;
    while (bin_padded.size() % 4 != 0) bin_padded.push_back(0);
    // Patch byteLength stays valid: padded BIN chunk is >= declared length.

    auto put32 = [](std::string& s, std::uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); };
    std::string glb;
    put32(glb, 0x46546C67u);
    put32(glb, 2u);
    put32(glb, static_cast<std::uint32_t>(12 + 8 + json.size() + 8 + bin_padded.size()));
    put32(glb, static_cast<std::uint32_t>(json.size()));
    put32(glb, 0x4E4F534Au);
    glb += json;
    put32(glb, static_cast<std::uint32_t>(bin_padded.size()));
    put32(glb, 0x004E4942u);
    glb.append(reinterpret_cast<const char*>(bin_padded.data()), bin_padded.size());
    write_text(dir.path / "quad.glb", glb);

    std::string err;
    auto model = load_gltf(dir.path / "quad.glb", &err);
    REQUIRE_MESSAGE(model.has_value(), err);
    REQUIRE(model->primitives.size() == 1);
    CHECK(model->primitives[0].mesh.vertices.size() == 4);
    // The missing image is not fatal: slot becomes -1.
    CHECK(model->materials[0].base_color_image == -1);
}

TEST_CASE("model: load_gltf reports an error for a missing file") {
    std::string err;
    auto model = load_gltf("/definitely/not/here.gltf", &err);
    CHECK_FALSE(model.has_value());
    CHECK_FALSE(err.empty());
}
