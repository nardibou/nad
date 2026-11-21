#include "nova/math/math.hpp"
#include "nova/renderer/animation.hpp"
#include "nova/renderer/model.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>

using namespace nova;
using math::Mat4;
using math::Quat;
using math::Vec3;

namespace {

bool near(f32 a, f32 b, f32 eps = 1e-4f) { return std::abs(a - b) <= eps; }
bool near(const Vec3& a, const Vec3& b, f32 eps = 1e-4f) {
    return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps);
}

// Two bones along +Y: root at the origin, child 1 m up. One vertex fully on each bone and one
// split 50/50.
renderer::ModelData two_bone_model() {
    renderer::ModelData m;
    m.nodes.resize(2);
    m.nodes[0].name = "root";
    m.nodes[1].name = "tip";
    m.nodes[1].parent = 0;
    m.nodes[1].translation = {0, 1, 0};
    renderer::SkinData skin;
    skin.joints = {0, 1};
    skin.inverse_bind = {Mat4::identity(), math::translate({0, -1, 0})};
    m.skins.push_back(skin);
    renderer::ModelPrimitive p;
    p.skin = 0;
    for (const Vec3& pos : {Vec3{1, 0, 0}, Vec3{1, 2, 0}, Vec3{1, 1, 0}}) {
        renderer::Vertex v;
        v.position = pos;
        v.normal = {1, 0, 0};
        v.tangent = {0, 1, 0, 1};
        p.mesh.vertices.push_back(v);
    }
    p.mesh.indices = {0, 1, 2};
    p.joints = {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}};
    p.weights = {{1, 0, 0, 0}, {1, 0, 0, 0}, {0.5f, 0.5f, 0, 0}};
    m.primitives.push_back(p);
    // "bend": the tip joint turns 90 degrees about +Z over one second.
    renderer::AnimationData bend;
    bend.name = "Rig|Bend";
    bend.duration = 1.0f;
    renderer::AnimChannel ch;
    ch.node = 1;
    ch.path = renderer::AnimPath::Rotation;
    ch.times = {0.0f, 1.0f};
    const Quat q = math::axis_angle({0, 0, 1}, math::kPi * 0.5f);
    ch.values = {{0, 0, 0, 1}, {q.x, q.y, q.z, q.w}};
    bend.channels.push_back(ch);
    m.animations.push_back(bend);
    return m;
}

} // namespace

TEST_CASE("math: quaternion matrix round trip, slerp and affine inverse") {
    const Quat q = math::axis_angle(Vec3{1, 2, 3}, 0.7f);
    const Quat back = math::to_quat(math::to_mat4(q));
    CHECK(std::abs(back.x * q.x + back.y * q.y + back.z * q.z + back.w * q.w) == doctest::Approx(1.0f).epsilon(1e-5));
    // to_mat4 agrees with the axis-angle rotate().
    const Vec3 p{0.3f, -1.2f, 2.0f};
    CHECK(near(math::transform_point(math::to_mat4(q), p), math::transform_point(math::rotate({1, 2, 3}, 0.7f), p)));
    const Quat half = math::slerp(Quat{}, math::axis_angle({0, 1, 0}, 1.0f), 0.5f);
    CHECK(near(math::transform_point(math::to_mat4(half), {1, 0, 0}),
               math::transform_point(math::rotate({0, 1, 0}, 0.5f), {1, 0, 0})));
    const Mat4 m = math::compose({1, 2, 3}, q, {2, 2, 2});
    const Mat4 id = math::inverse_affine(m) * m;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            CHECK(near(id.m[static_cast<usize>(c)][static_cast<usize>(r)], c == r ? 1.0f : 0.0f));
        }
    }
}

TEST_CASE("animation: sampling, cross-fade and linear-blend skinning") {
    const renderer::ModelData model = two_bone_model();
    CHECK(model.find_animation("Bend") == 0); // "Rig|" prefix is optional
    CHECK(model.find_animation("Rig|Bend") == 0);
    CHECK(model.find_animation("Missing") == -1);

    std::vector<renderer::NodeTransform> pose = renderer::rest_pose(model);
    renderer::sample_clip(model.animations[0], 0.5f, pose);
    const Quat expect = math::axis_angle({0, 0, 1}, math::kPi * 0.25f);
    CHECK(near(pose[1].rotation.z, expect.z, 1e-4f));
    CHECK(near(pose[1].rotation.w, expect.w, 1e-4f));

    // Bind pose reproduces the mesh.
    std::vector<Mat4> world;
    std::vector<renderer::Vertex> out;
    renderer::node_world_matrices(model, renderer::rest_pose(model), world);
    renderer::skin_primitive(model, model.primitives[0], world, out);
    for (usize i = 0; i < out.size(); ++i) {
        CHECK(near(out[i].position, model.primitives[0].mesh.vertices[i].position));
    }

    // Fully bent: the tip vertex (1, 2, 0) rotates 90 degrees about the joint at (0, 1, 0).
    renderer::Animator animator(model);
    animator.play(0, 0.0f, false);
    animator.update(2.0f);
    CHECK(animator.finished());
    renderer::skin_primitive(model, model.primitives[0], animator.world(), out);
    CHECK(near(out[0].position, {1, 0, 0}));
    CHECK(near(out[1].position, {-1, 2, 0}));
    CHECK(near(out[1].normal, {0, 1, 0}));
    CHECK(near(out[2].position, {0.5f, 1.5f, 0})); // 50/50 of (1, 1, 0) and the bent (0, 2, 0)
    // A cross-faded restart blends towards the rest pose of the new layer's first frame.
    animator.play(0, 1.0f, false);
    animator.update(0.5f); // new layer: weight 0.5 at t = 0.5 (45 degrees), old: 90 degrees
    const Vec3 tip = math::transform_point(animator.world()[1], {1, 0, 0});
    const f32 angle = std::atan2(tip.y - 1.0f, tip.x) * 180.0f / math::kPi;
    CHECK(angle == doctest::Approx(67.5f).epsilon(0.02)); // nlerp(90, 45, 0.5)
}

TEST_CASE("animation: the FPS AK-74M rig loads with its skin and clips") {
    const std::filesystem::path path = std::filesystem::path(NOVA_TEST_ASSET_DIR) / "viewmodels/fps_ak74m.glb";
    if (!std::filesystem::exists(path)) {
        MESSAGE("asset not present, skipping");
        return;
    }
    std::string error;
    const auto model = renderer::load_gltf(path, &error);
    REQUIRE_MESSAGE(model.has_value(), error);
    REQUIRE(model->skins.size() == 1);
    CHECK(model->skins[0].joints.size() == 85);
    CHECK(model->animations.size() == 8);
    for (const char* clip : {"AK_Idle", "AK_Draw", "AK_Walk", "AK_Run", "AK_Shot", "AK_Reload", "AK_Reload_var2",
                             "AK_Reload_full"}) {
        CHECK_MESSAGE(model->find_animation(clip) >= 0, clip);
    }
    CHECK(model->animations[static_cast<usize>(model->find_animation("AK_Reload_full"))].duration ==
          doctest::Approx(3.5f).epsilon(0.01));
    CHECK(model->find_node("Head_Cam_014") >= 0);
    CHECK(model->find_node("PBody_058") >= 0);
    REQUIRE(model->primitives.size() == 3);
    for (const auto& p : model->primitives) {
        CHECK(p.skin == 0);
        CHECK(p.joints.size() == p.mesh.vertices.size());
    }
    // Idle pose: the rifle spans ~2.76 units along +Z (forward) in model space.
    renderer::Animator animator(*model);
    animator.play(model->find_animation("AK_Idle"), 0.0f);
    animator.update(0.0f);
    std::vector<renderer::Vertex> out;
    renderer::skin_primitive(*model, model->primitives[1], animator.world(), out);
    f32 zmin = 1e9f, zmax = -1e9f;
    for (const auto& v : out) {
        zmin = std::min(zmin, v.position.z);
        zmax = std::max(zmax, v.position.z);
        CHECK(std::isfinite(v.position.x));
    }
    CHECK(zmin == doctest::Approx(-0.779f).epsilon(0.02));
    CHECK(zmax == doctest::Approx(1.978f).epsilon(0.02));
}
