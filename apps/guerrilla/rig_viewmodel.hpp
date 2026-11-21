#pragma once

// First-person arms + rifle from a skinned, animated glTF (e.g. the Sketchfab "FPS AK-74m
// animations" rig): animation state machine, CPU skinning into dynamic meshes, aim-down-sights
// alignment from the rifle's iron sights and the muzzle frame for the flash.

#include "nova/math/math.hpp"
#include "nova/renderer/animation.hpp"
#include "nova/renderer/model.hpp"
#include "nova/renderer/world_renderer.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace guerrilla {

class RigViewmodel {
public:
    // Needs clips named *AK_Idle/Walk/Run/Shot/Reload/Reload_var2/Reload_full/Draw and the
    // Head_Cam_014 / PBody_058 bones. Returns nullptr (and sets `error`) otherwise.
    [[nodiscard]] static std::unique_ptr<RigViewmodel> load(const std::filesystem::path& path,
                                                            nova::renderer::WorldRenderer& world, std::string* error);

    struct Motion {
        float dt = 0.0f;
        float speed = 0.0f; // horizontal m/s
        bool grounded = true;
        bool sprint = false;
        float ads = 0.0f;
    };
    void update(const Motion& m);
    void fire();
    // Starts a reload and returns its duration (s); the magazine is full when it ends.
    float start_reload(bool empty);

    // Model -> view space (x right, y up, -z forward, eye at the origin) with `ads` in 0..1.
    [[nodiscard]] nova::math::Mat4 model_to_view(float ads) const;
    // Procedural flash space (muzzle at `flash_muzzle`, firing along -Z, metres) -> view space.
    [[nodiscard]] nova::math::Mat4 flash_to_view(float ads, const nova::math::Vec3& flash_muzzle) const;
    [[nodiscard]] nova::math::Vec3 muzzle_view(float ads) const;
    void push_items(std::vector<nova::renderer::WorldDrawItem>& items, const nova::math::Mat4& view_to_world) const;

    [[nodiscard]] nova::usize triangle_count() const noexcept { return triangles_; }

private:
    enum class State : nova::u8 { Draw, Locomotion, Action };
    enum Clip { kIdle, kWalk, kRun, kShot, kReload, kReload2, kReloadFull, kDraw, kClipCount };

    RigViewmodel() = default;
    [[nodiscard]] nova::math::Mat4 sight_alignment(float ads) const;
    [[nodiscard]] nova::math::Mat4 rifle_view() const; // rifle bone -> view (hip, no ADS)

    std::unique_ptr<nova::renderer::ModelData> model_;
    std::unique_ptr<nova::renderer::Animator> animator_;
    struct Part {
        nova::usize primitive;
        nova::renderer::MeshId mesh;
        nova::renderer::MaterialId material;
        bool two_sided;
    };
    std::vector<Part> parts_;
    std::vector<nova::renderer::Vertex> scratch_;
    nova::renderer::WorldRenderer* world_ = nullptr;
    int clips_[kClipCount]{};
    int head_cam_ = -1;
    int rifle_ = -1;
    State state_ = State::Draw;
    int locomotion_ = kIdle;
    bool alternate_reload_ = false;
    nova::usize triangles_ = 0;
    float ads_ = 0.0f;
};

} // namespace guerrilla
