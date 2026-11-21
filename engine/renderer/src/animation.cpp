#include "nova/renderer/animation.hpp"

#include <algorithm>
#include <cmath>

namespace nova::renderer {

using math::Mat4;
using math::Quat;
using math::Vec3;
using math::Vec4;

namespace {

Vec4 lerp4(const Vec4& a, const Vec4& b, f32 t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
}

Vec4 hermite(const Vec4& p0, const Vec4& m0, const Vec4& p1, const Vec4& m1, f32 t, f32 dt) {
    const f32 t2 = t * t, t3 = t2 * t;
    const f32 h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
    auto c = [&](f32 a, f32 ma, f32 b, f32 mb) { return h00 * a + h10 * dt * ma + h01 * b + h11 * dt * mb; };
    return {c(p0.x, m0.x, p1.x, m1.x), c(p0.y, m0.y, p1.y, m1.y), c(p0.z, m0.z, p1.z, m1.z),
            c(p0.w, m0.w, p1.w, m1.w)};
}

Vec4 sample_channel(const AnimChannel& ch, f32 time) {
    const auto& t = ch.times;
    const bool cubic = ch.interpolation == AnimInterpolation::CubicSpline;
    auto value = [&](usize key) { return ch.values[cubic ? key * 3 + 1 : key]; };
    if (time <= t.front()) {
        return value(0);
    }
    if (time >= t.back()) {
        return value(t.size() - 1);
    }
    const usize k1 = static_cast<usize>(std::upper_bound(t.begin(), t.end(), time) - t.begin());
    const usize k0 = k1 - 1;
    const f32 span = t[k1] - t[k0];
    const f32 u = span > 0.0f ? (time - t[k0]) / span : 0.0f;
    switch (ch.interpolation) {
    case AnimInterpolation::Step: return value(k0);
    case AnimInterpolation::CubicSpline:
        return hermite(value(k0), ch.values[k0 * 3 + 2], value(k1), ch.values[k1 * 3], u, span);
    case AnimInterpolation::Linear: break;
    }
    if (ch.path == AnimPath::Rotation) {
        const Quat q = math::slerp(Quat{value(k0).x, value(k0).y, value(k0).z, value(k0).w},
                                   Quat{value(k1).x, value(k1).y, value(k1).z, value(k1).w}, u);
        return {q.x, q.y, q.z, q.w};
    }
    return lerp4(value(k0), value(k1), u);
}

} // namespace

std::vector<NodeTransform> rest_pose(const ModelData& model) {
    std::vector<NodeTransform> pose(model.nodes.size());
    for (usize i = 0; i < pose.size(); ++i) {
        pose[i] = {model.nodes[i].translation, model.nodes[i].rotation, model.nodes[i].scale};
    }
    return pose;
}

void sample_clip(const AnimationData& clip, f32 time, std::span<NodeTransform> pose) {
    time = std::clamp(time, 0.0f, clip.duration);
    for (const AnimChannel& ch : clip.channels) {
        if (ch.node < 0 || static_cast<usize>(ch.node) >= pose.size() || ch.times.empty()) {
            continue;
        }
        const Vec4 v = sample_channel(ch, time);
        NodeTransform& n = pose[static_cast<usize>(ch.node)];
        switch (ch.path) {
        case AnimPath::Translation: n.translation = {v.x, v.y, v.z}; break;
        case AnimPath::Rotation: n.rotation = math::normalize(Quat{v.x, v.y, v.z, v.w}); break;
        case AnimPath::Scale: n.scale = {v.x, v.y, v.z}; break;
        }
    }
}

void blend_pose(std::span<NodeTransform> pose, std::span<const NodeTransform> other, f32 weight) {
    const usize n = std::min(pose.size(), other.size());
    for (usize i = 0; i < n; ++i) {
        pose[i].translation = math::lerp(pose[i].translation, other[i].translation, weight);
        pose[i].rotation = math::nlerp(pose[i].rotation, other[i].rotation, weight);
        pose[i].scale = math::lerp(pose[i].scale, other[i].scale, weight);
    }
}

void node_world_matrices(const ModelData& model, std::span<const NodeTransform> pose, std::vector<Mat4>& world) {
    const usize n = model.nodes.size();
    world.resize(n);
    // 0 = pending, 1 = done; parents are resolved on demand (glTF does not order nodes).
    std::vector<u8> done(n, 0);
    std::vector<usize> chain;
    for (usize i = 0; i < n; ++i) {
        usize cur = i;
        chain.clear();
        while (!done[cur]) {
            chain.push_back(cur);
            const i32 p = model.nodes[cur].parent;
            if (p < 0) {
                break;
            }
            cur = static_cast<usize>(p);
        }
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            const usize k = *it;
            const NodeTransform& t = k < pose.size() ? pose[k] : NodeTransform{};
            const Mat4 local = math::compose(t.translation, t.rotation, t.scale);
            const i32 p = model.nodes[k].parent;
            world[k] = p >= 0 ? world[static_cast<usize>(p)] * local : local;
            done[k] = 1;
        }
    }
}

void skin_primitive(const ModelData& model, const ModelPrimitive& prim, std::span<const Mat4> world,
                    std::vector<Vertex>& out) {
    const auto& src = prim.mesh.vertices;
    out.resize(src.size());
    if (prim.skin < 0 || static_cast<usize>(prim.skin) >= model.skins.size() || prim.joints.size() != src.size()) {
        const Mat4 m = prim.node >= 0 && static_cast<usize>(prim.node) < world.size()
                           ? world[static_cast<usize>(prim.node)]
                           : prim.transform;
        for (usize v = 0; v < src.size(); ++v) {
            out[v] = src[v];
            out[v].position = math::transform_point(m, src[v].position);
            out[v].normal = math::transform_normal(m, src[v].normal);
            const Vec4 t4 = m * Vec4{src[v].tangent.x, src[v].tangent.y, src[v].tangent.z, 0.0f};
            out[v].tangent = {t4.x, t4.y, t4.z, src[v].tangent.w};
        }
        return;
    }
    const SkinData& skin = model.skins[static_cast<usize>(prim.skin)];
    // Joint matrices as 3x4 rows for a tight inner loop.
    struct Affine {
        f32 r[3][4];
    };
    std::vector<Affine> joints(skin.joints.size());
    for (usize j = 0; j < joints.size(); ++j) {
        const i32 node = skin.joints[j];
        const Mat4 m = (node >= 0 && static_cast<usize>(node) < world.size() ? world[static_cast<usize>(node)]
                                                                            : Mat4::identity()) *
                       skin.inverse_bind[j];
        for (int row = 0; row < 3; ++row) {
            for (int c = 0; c < 4; ++c) {
                joints[j].r[row][c] = m.m[static_cast<usize>(c)][static_cast<usize>(row)];
            }
        }
    }
    for (usize v = 0; v < src.size(); ++v) {
        const auto& ji = prim.joints[v];
        const Vec4& w = prim.weights[v];
        const f32 wk[4] = {w.x, w.y, w.z, w.w};
        Affine m{};
        for (int k = 0; k < 4; ++k) {
            if (wk[k] == 0.0f) {
                continue;
            }
            const Affine& a = joints[ji[static_cast<usize>(k)]];
            for (int row = 0; row < 3; ++row) {
                for (int c = 0; c < 4; ++c) {
                    m.r[row][c] += a.r[row][c] * wk[k];
                }
            }
        }
        const Vertex& s = src[v];
        Vertex& d = out[v];
        d = s;
        auto xf = [&](const Vec3& p, f32 wcoord) -> Vec3 {
            return {m.r[0][0] * p.x + m.r[0][1] * p.y + m.r[0][2] * p.z + m.r[0][3] * wcoord,
                    m.r[1][0] * p.x + m.r[1][1] * p.y + m.r[1][2] * p.z + m.r[1][3] * wcoord,
                    m.r[2][0] * p.x + m.r[2][1] * p.y + m.r[2][2] * p.z + m.r[2][3] * wcoord};
        };
        d.position = xf(s.position, 1.0f);
        // Rig scales are uniform per joint, so the linear part transforms normals up to scale.
        d.normal = math::normalize(xf(s.normal, 0.0f));
        const Vec3 t = math::normalize(xf({s.tangent.x, s.tangent.y, s.tangent.z}, 0.0f));
        d.tangent = {t.x, t.y, t.z, s.tangent.w};
    }
}

Animator::Animator(const ModelData& model) : model_(model), rest_(rest_pose(model)) {
    pose_ = rest_;
    scratch_ = rest_;
    node_world_matrices(model_, pose_, world_);
}

void Animator::play(i32 clip, f32 fade, bool loop, f32 speed, f32 start) {
    if (clip < 0 || static_cast<usize>(clip) >= model_.animations.size()) {
        return;
    }
    Layer l;
    l.clip = clip;
    l.time = start;
    l.speed = speed;
    l.loop = loop;
    l.weight = fade > 0.0f && !layers_.empty() ? 0.0f : 1.0f;
    l.fade_rate = fade > 0.0f ? 1.0f / fade : 0.0f;
    layers_.push_back(l);
    if (layers_.size() > 6) {
        layers_.erase(layers_.begin()); // drop the oldest of a burst of quick transitions
    }
}

void Animator::set_speed(f32 speed) noexcept {
    if (!layers_.empty()) {
        layers_.back().speed = speed;
    }
}

bool Animator::finished() const noexcept {
    if (layers_.empty()) {
        return true;
    }
    const Layer& l = layers_.back();
    return !l.loop && l.time >= model_.animations[static_cast<usize>(l.clip)].duration;
}

void Animator::update(f32 dt) {
    for (Layer& l : layers_) {
        const f32 duration = model_.animations[static_cast<usize>(l.clip)].duration;
        l.time += dt * l.speed;
        if (l.loop && duration > 0.0f) {
            l.time = std::fmod(l.time, duration);
        } else {
            l.time = std::min(l.time, duration);
        }
        l.weight = std::min(1.0f, l.weight + l.fade_rate * dt);
    }
    // A fully opaque layer hides everything below it.
    for (usize i = layers_.size(); i-- > 1;) {
        if (layers_[i].weight >= 1.0f) {
            layers_.erase(layers_.begin(), layers_.begin() + static_cast<std::ptrdiff_t>(i));
            break;
        }
    }
    pose_ = rest_;
    for (usize i = 0; i < layers_.size(); ++i) {
        const Layer& l = layers_[i];
        if (i == 0) {
            sample_clip(model_.animations[static_cast<usize>(l.clip)], l.time, pose_);
            continue;
        }
        scratch_ = rest_;
        sample_clip(model_.animations[static_cast<usize>(l.clip)], l.time, scratch_);
        blend_pose(pose_, scratch_, l.weight);
    }
    node_world_matrices(model_, pose_, world_);
}

} // namespace nova::renderer
