#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"

#include <doctest/doctest.h>

using namespace nova;
using namespace nova::math;

namespace {
bool near(f32 a, f32 b, f32 eps = 1e-5f) { return std::abs(a - b) <= eps; }
bool near(const Vec4& a, const Vec4& b, f32 eps = 1e-5f) {
    return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps) && near(a.w, b.w, eps);
}
} // namespace

TEST_CASE("math: vector basics") {
    CHECK(dot(Vec3{1, 2, 3}, Vec3{4, 5, 6}) == 32.0f);
    CHECK(cross(Vec3{1, 0, 0}, Vec3{0, 1, 0}) == Vec3{0, 0, 1});
    CHECK(near(length(normalize(Vec3{3, 4, 0})), 1.0f));
    CHECK(normalize(Vec3{}) == Vec3{});
}

TEST_CASE("math: matrices compose right-to-left") {
    const Mat4 t = translate({1, 2, 3});
    const Mat4 s = scale({2, 2, 2});
    CHECK(near((t * s) * Vec4{1, 1, 1, 1}, Vec4{3, 4, 5, 1})); // scale first, then translate
    CHECK(near((s * t) * Vec4{1, 1, 1, 1}, Vec4{4, 6, 8, 1}));
    CHECK(near((Mat4::identity() * t) * Vec4{0, 0, 0, 1}, Vec4{1, 2, 3, 1}));
}

TEST_CASE("math: rotate follows the right-hand rule") {
    const Vec4 r = rotate({0, 0, 1}, kPi / 2) * Vec4{1, 0, 0, 0};
    CHECK(near(r, Vec4{0, 1, 0, 0}));
    const Vec4 y = rotate({0, 1, 0}, kPi / 2) * Vec4{0, 0, -1, 0};
    CHECK(near(y, Vec4{-1, 0, 0, 0}));
}

TEST_CASE("math: look_at maps the target onto -Z") {
    const Mat4 v = look_at({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
    CHECK(near(v * Vec4{0, 0, 0, 1}, Vec4{0, 0, -5, 1}));
    CHECK(near(v * Vec4{0, 1, 0, 1}, Vec4{0, 1, -5, 1})); // up stays up
}

TEST_CASE("math: perspective produces Vulkan clip space") {
    const Mat4 p = perspective(radians(90.0f), 1.0f, 1.0f, 100.0f);
    auto ndc = [&](Vec4 v) {
        const Vec4 c = p * v;
        return Vec4{c.x / c.w, c.y / c.w, c.z / c.w, 1};
    };
    CHECK(near(ndc({0, 0, -1, 1}).z, 0.0f));         // near plane -> 0
    CHECK(near(ndc({0, 0, -100, 1}).z, 1.0f, 1e-4f)); // far plane -> 1
    CHECK(ndc({0, 0.5f, -1, 1}).y < 0.0f);            // +Y world is up on screen (Vulkan Y down)
    CHECK(near(ndc({1, 0, -1, 1}).x, 1.0f));          // 90 degree FOV edge
}

TEST_CASE("mesh: cube faces are outward and counter-clockwise") {
    const auto cube = renderer::make_cube(2.0f);
    REQUIRE(cube.vertices.size() == 24);
    REQUIRE(cube.indices.size() == 36);
    for (usize i = 0; i < cube.indices.size(); i += 3) {
        const auto& a = cube.vertices[cube.indices[i]];
        const auto& b = cube.vertices[cube.indices[i + 1]];
        const auto& c = cube.vertices[cube.indices[i + 2]];
        const Vec3 n = cross(b.position - a.position, c.position - a.position);
        CHECK(dot(n, a.normal) > 0.0f);        // CCW seen from outside
        CHECK(dot(a.position, a.normal) > 0.0f); // normal points away from the centre
        const bool on_surface = near(std::abs(a.position.x), 1.0f) || near(std::abs(a.position.y), 1.0f) ||
                                near(std::abs(a.position.z), 1.0f);
        CHECK(on_surface);
    }
}

TEST_CASE("mesh: plane faces +Y") {
    const auto plane = renderer::make_plane(4.0f);
    REQUIRE(plane.indices.size() == 6);
    const auto& v = plane.vertices;
    const Vec3 n = cross(v[plane.indices[1]].position - v[plane.indices[0]].position,
                         v[plane.indices[2]].position - v[plane.indices[0]].position);
    CHECK(n.y > 0.0f);
}
