#include "rig_viewmodel.hpp"

#include "nova/core/log.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace guerrilla {

using namespace nova;
using math::Mat4;
using math::Quat;
using math::Vec3;
using math::Vec4;

namespace {

// Iron sights and muzzle in the rifle bone's local frame (bone +Y = forward, -Z = up), measured
// from the bind-pose mesh of the Sketchfab AK-74M rig.
constexpr Vec3 kRearSight{0.0f, 0.10f, -0.068f};
constexpr Vec3 kFrontSight{0.0f, 0.59f, -0.068f};
constexpr Vec3 kMuzzle{0.0f, 0.726f, 0.002f};
constexpr Vec3 kRifleUp{0.0f, 0.0f, -1.0f};
// Eye relief when aiming, in rig units (the rig is ~1.46x real scale).
constexpr float kAdsDistance = 0.34f;
// The authoring camera sits behind and above the Head_Cam bone (Camera node of the file), in
// Head_Cam-local units.
constexpr Vec3 kEyeOffset{0.0f, 0.0208f, -0.0792f};

Vec3 xform_dir(const Mat4& m, const Vec3& d) {
    const Vec4 r = m * Vec4{d.x, d.y, d.z, 0.0f};
    return {r.x, r.y, r.z};
}

Mat4 columns(const Vec3& x, const Vec3& y, const Vec3& z) {
    Mat4 m = Mat4::identity();
    const Vec3 c[3] = {x, y, z};
    for (int i = 0; i < 3; ++i) {
        m.m[static_cast<usize>(i)][0] = c[i].x;
        m.m[static_cast<usize>(i)][1] = c[i].y;
        m.m[static_cast<usize>(i)][2] = c[i].z;
    }
    return m;
}

Mat4 transpose3(const Mat4& m) {
    Mat4 r = Mat4::identity();
    for (usize c = 0; c < 3; ++c) {
        for (usize row = 0; row < 3; ++row) {
            r.m[c][row] = m.m[row][c];
        }
    }
    return r;
}

} // namespace

std::unique_ptr<RigViewmodel> RigViewmodel::load(const std::filesystem::path& path, renderer::WorldRenderer& world,
                                                 std::string* error) {
    auto fail = [&](const std::string& msg) -> std::unique_ptr<RigViewmodel> {
        if (error) {
            *error = msg;
        }
        return nullptr;
    };
    auto model = renderer::load_gltf(path, error);
    if (!model) {
        return nullptr;
    }
    std::unique_ptr<RigViewmodel> vm(new RigViewmodel());
    vm->model_ = std::make_unique<renderer::ModelData>(std::move(*model));
    const renderer::ModelData& md = *vm->model_;

    const char* names[kClipCount] = {"AK_Idle",        "AK_Walk",        "AK_Run",         "AK_Shot",
                                     "AK_Reload",      "AK_Reload_var2", "AK_Reload_full", "AK_Draw"};
    for (int i = 0; i < kClipCount; ++i) {
        vm->clips_[i] = md.find_animation(names[i]);
        if (vm->clips_[i] < 0) {
            return fail(std::string("viewmodel has no animation clip ") + names[i]);
        }
    }
    vm->head_cam_ = md.find_node("Head_Cam_014");
    vm->rifle_ = md.find_node("PBody_058");
    if (vm->head_cam_ < 0 || vm->rifle_ < 0) {
        return fail("viewmodel rig lacks the Head_Cam_014 / PBody_058 bones");
    }

    vm->world_ = &world;
    const std::vector<renderer::MaterialId> materials = world.upload_materials(md);
    for (usize i = 0; i < md.primitives.size(); ++i) {
        const renderer::ModelPrimitive& prim = md.primitives[i];
        Part part;
        part.primitive = i;
        part.mesh = world.upload_dynamic(prim.mesh);
        if (prim.material >= 0 && static_cast<usize>(prim.material) < materials.size()) {
            part.material = materials[static_cast<usize>(prim.material)];
            part.two_sided = md.materials[static_cast<usize>(prim.material)].double_sided;
        } else {
            renderer::PbrMaterial m;
            m.metallic = 0.0f;
            part.material = world.add_material(m);
            part.two_sided = false;
        }
        vm->parts_.push_back(part);
        vm->triangles_ += prim.mesh.indices.size() / 3;
    }

    vm->animator_ = std::make_unique<renderer::Animator>(*vm->model_);
    vm->animator_->play(vm->clips_[kDraw], 0.0f, false);
    vm->state_ = State::Draw;
    return vm;
}

void RigViewmodel::update(const Motion& m) {
    ads_ = m.ads;
    int desired = kIdle;
    if (m.sprint && m.speed > 1.0f && m.grounded) {
        desired = kRun;
    } else if (m.speed > 0.4f && m.grounded && m.ads < 0.5f) {
        desired = kWalk;
    }
    if (state_ != State::Locomotion && animator_->finished()) {
        state_ = State::Locomotion;
        locomotion_ = desired;
        animator_->play(clips_[desired], 0.2f, true);
    } else if (state_ == State::Locomotion && desired != locomotion_) {
        locomotion_ = desired;
        animator_->play(clips_[desired], 0.25f, true);
    }
    if (state_ == State::Locomotion) {
        if (locomotion_ == kWalk) {
            animator_->set_speed(std::clamp(m.speed / 3.6f, 0.7f, 1.5f));
        } else if (locomotion_ == kRun) {
            animator_->set_speed(std::clamp(m.speed / 6.2f, 0.8f, 1.3f));
        }
    }
    animator_->update(m.dt);

    const auto& world = animator_->world();
    for (const Part& p : parts_) {
        renderer::skin_primitive(*model_, model_->primitives[p.primitive], world, scratch_);
        world_->update_vertices(p.mesh, scratch_);
    }
}

void RigViewmodel::fire() {
    animator_->play(clips_[kShot], 0.03f, false);
    state_ = State::Action;
}

float RigViewmodel::start_reload(bool empty) {
    const int clip = empty ? kReloadFull : (alternate_reload_ ? kReload2 : kReload);
    alternate_reload_ = !alternate_reload_;
    animator_->play(clips_[clip], 0.12f, false);
    state_ = State::Action;
    return model_->animations[static_cast<usize>(clips_[clip])].duration;
}

Mat4 RigViewmodel::rifle_view() const {
    const auto& world = animator_->world();
    // Rig space has forward +Z / right -X; view space has forward -Z / right +X: a 180 degree turn.
    const Mat4 to_view = math::scale({-1.0f, 1.0f, -1.0f}) * math::inverse_affine(world[static_cast<usize>(head_cam_)] * math::translate(kEyeOffset));
    return to_view * world[static_cast<usize>(rifle_)];
}

// Rotates and moves the rifle so the sight line runs down the view axis, blended by `ads`.
Mat4 RigViewmodel::sight_alignment(float ads) const {
    if (ads <= 0.0f) {
        return Mat4::identity();
    }
    const Mat4 rifle = rifle_view();
    const Vec3 rear = math::transform_point(rifle, kRearSight);
    const Vec3 front = math::transform_point(rifle, kFrontSight);
    const Vec3 fwd = math::normalize(front - rear);
    Vec3 up = xform_dir(rifle, kRifleUp);
    up = math::normalize(up - fwd * math::dot(up, fwd));
    const Vec3 right = math::cross(fwd, up);
    const Quat q = math::to_quat(transpose3(columns(right, up, -fwd)));
    const Vec3 target{0.0f, -0.004f, -kAdsDistance};
    return math::translate(math::lerp(rear, target, ads)) * math::to_mat4(math::slerp(Quat{}, q, ads)) *
           math::translate(-rear);
}

Mat4 RigViewmodel::model_to_view(float ads) const {
    const Mat4 to_view = math::scale({-1.0f, 1.0f, -1.0f}) *
                         math::inverse_affine(animator_->world()[static_cast<usize>(head_cam_)] * math::translate(kEyeOffset));
    return sight_alignment(ads) * to_view;
}

Mat4 RigViewmodel::flash_to_view(float ads, const Vec3& flash_muzzle) const {
    // Flash space: +X right, +Y up, -Z forward. Rifle bone: -X right, -Z up, +Y forward.
    const Mat4 flash_to_rifle = math::translate(kMuzzle) * columns({-1, 0, 0}, {0, 0, -1}, {0, -1, 0}) *
                                math::scale({1.3f, 1.3f, 1.3f}) * math::translate(-flash_muzzle);
    return sight_alignment(ads) * rifle_view() * flash_to_rifle;
}

Vec3 RigViewmodel::muzzle_view(float ads) const {
    return math::transform_point(sight_alignment(ads) * rifle_view(), kMuzzle);
}

void RigViewmodel::push_items(std::vector<renderer::WorldDrawItem>& items, const Mat4& view_to_world) const {
    const Mat4 m = view_to_world * model_to_view(ads_);
    for (const Part& p : parts_) {
        renderer::WorldDrawItem it{p.mesh, m};
        it.material = p.material;
        it.two_sided = p.two_sided;
        it.casts_shadow = false;
        items.push_back(it);
    }
}

} // namespace guerrilla
