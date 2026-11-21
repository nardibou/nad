#include "viewmodel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace guerrilla {
namespace {

using nova::f32;
using nova::u32;
using nova::u8;
using nova::math::cross;
using nova::math::dot;
using nova::math::kPi;
using nova::math::length;
using nova::math::normalize;
using nova::math::Vec2;
using nova::math::Vec3;
using nova::math::Vec4;
using nova::renderer::ImageData;
using nova::renderer::MeshData;
using nova::renderer::Vertex;

constexpr f32 kUv = 4.0f; // texture repeats per metre
constexpr int kMat = static_cast<int>(ViewMaterial::Count);

Vec3 hadamard(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
f32 clamp01(f32 v) { return std::clamp(v, 0.0f, 1.0f); }
f32 smooth(f32 t) {
    t = clamp01(t);
    return t * t * (3.0f - 2.0f * t);
}
f32 deg(f32 d) { return d * kPi / 180.0f; }

// ---------------------------------------------------------------------------------------------
// Mesh construction primitives
// ---------------------------------------------------------------------------------------------

Vertex make_vertex(const Vec3& p, const Vec3& n, f32 u, f32 v, const Vec3& tdir, const Vec3& vdir) {
    Vertex x;
    x.position = p;
    x.normal = n;
    x.uv = {u, v};
    Vec3 t = tdir - n * dot(n, tdir);
    if (dot(t, t) < 1e-12f) {
        const Vec3 axis = std::fabs(n.x) < 0.9f ? Vec3{1.0f, 0.0f, 0.0f} : Vec3{0.0f, 1.0f, 0.0f};
        t = axis - n * dot(n, axis);
    }
    t = normalize(t);
    const f32 w = dot(cross(n, t), vdir) < 0.0f ? -1.0f : 1.0f;
    x.tangent = {t.x, t.y, t.z, w};
    return x;
}

// Adds a triangle, flipping the winding so the geometric normal agrees with the vertex normals.
void add_tri(MeshData& m, u32 a, u32 b, u32 c) {
    const Vertex &va = m.vertices[a], &vb = m.vertices[b], &vc = m.vertices[c];
    const Vec3 g = cross(vb.position - va.position, vc.position - va.position);
    if (dot(g, g) < 1e-14f) return;
    const Vec3 ns = va.normal + vb.normal + vc.normal;
    if (dot(g, ns) < 0.0f) std::swap(b, c);
    m.indices.push_back(a);
    m.indices.push_back(b);
    m.indices.push_back(c);
}

struct Shape {
    bool rect; // rounded rectangle (corner radius `radius`) or ellipse
    f32 radius;
    int segs; // arc segments per corner (rect) or segments around (ellipse)
};

struct P2 {
    f32 x, y, nx, ny;
};

void build_profile(const Shape& s, f32 hw, f32 hh, std::vector<P2>& out) {
    out.clear();
    if (!s.rect) {
        for (int i = 0; i <= s.segs; ++i) {
            const f32 a = 2.0f * kPi * static_cast<f32>(i) / static_cast<f32>(s.segs);
            const f32 c = std::cos(a), sn = std::sin(a);
            f32 nx = c * hh, ny = sn * hw;
            const f32 l = std::sqrt(nx * nx + ny * ny);
            if (l < 1e-9f) {
                nx = c;
                ny = sn;
            } else {
                nx /= l;
                ny /= l;
            }
            out.push_back({hw * c, hh * sn, nx, ny});
        }
        return;
    }
    const f32 rr = std::max(std::min(s.radius, 0.98f * std::min(hw, hh)), 1e-5f);
    const std::array<f32, 4> sx = {1.0f, -1.0f, -1.0f, 1.0f};
    const std::array<f32, 4> sy = {1.0f, 1.0f, -1.0f, -1.0f};
    for (int k = 0; k < 4; ++k) {
        const f32 cx = sx[static_cast<std::size_t>(k)] * (hw - rr);
        const f32 cy = sy[static_cast<std::size_t>(k)] * (hh - rr);
        for (int j = 0; j <= s.segs; ++j) {
            const f32 a = 0.5f * kPi * (static_cast<f32>(k) + static_cast<f32>(j) / static_cast<f32>(s.segs));
            const f32 c = std::cos(a), sn = std::sin(a);
            out.push_back({cx + rr * c, cy + rr * sn, c, sn});
        }
    }
    out.push_back(out.front());
}

struct Ring {
    Vec3 c, r, u;
    f32 hw, hh;
};

void add_sweep(MeshData& m, const std::vector<Ring>& rings, const Shape& sh, bool cap0, bool cap1) {
    const std::size_t nr = rings.size();
    if (nr < 2) return;
    std::vector<std::vector<P2>> prof(nr);
    std::vector<std::vector<Vec3>> pos(nr);
    for (std::size_t k = 0; k < nr; ++k) {
        const Ring& rg = rings[k];
        build_profile(sh, rg.hw, rg.hh, prof[k]);
        for (const P2& p : prof[k]) pos[k].push_back(rg.c + rg.r * p.x + rg.u * p.y);
    }
    const std::size_t np = prof[0].size();
    std::vector<f32> s(nr, 0.0f);
    for (std::size_t k = 1; k < nr; ++k) s[k] = s[k - 1] + length(rings[k].c - rings[k - 1].c);

    const u32 base = static_cast<u32>(m.vertices.size());
    for (std::size_t k = 0; k < nr; ++k) {
        const Ring& rg = rings[k];
        const std::size_t ka = k > 0 ? k - 1 : k, kb = k + 1 < nr ? k + 1 : k;
        const bool pole = rg.hw < 1e-5f || rg.hh < 1e-5f;
        f32 perim = 0.0f;
        for (std::size_t i = 0; i < np; ++i) {
            if (i > 0) perim += length(pos[k][i] - pos[k][i - 1]);
            const P2& p = prof[k][i];
            const Vec3 tr = normalize(rg.r * (-p.ny) + rg.u * p.nx);
            const Vec3 nref = rg.r * p.nx + rg.u * p.ny;
            const Vec3 dp = pos[kb][i] - pos[ka][i];
            Vec3 n = cross(tr, dp);
            if (pole || dot(n, n) < 1e-16f) {
                if (k == 0) n = rings[0].c - rings[1].c;
                else if (k + 1 == nr) n = rings[k].c - rings[k - 1].c;
                else n = nref;
                if (dot(n, n) < 1e-16f) n = nref;
            } else if (dot(n, nref) < 0.0f) {
                n = -n;
            }
            n = normalize(n);
            const Vec3 td = dot(dp, dp) > 1e-14f ? dp : cross(n, tr);
            m.vertices.push_back(make_vertex(pos[k][i], n, s[k] * kUv, perim * kUv, td, tr));
        }
    }
    for (std::size_t k = 0; k + 1 < nr; ++k) {
        for (std::size_t i = 0; i + 1 < np; ++i) {
            const u32 a = base + static_cast<u32>(k * np + i);
            const u32 b = a + 1;
            const u32 c = base + static_cast<u32>((k + 1) * np + i);
            const u32 d = c + 1;
            add_tri(m, a, b, c);
            add_tri(m, b, d, c);
        }
    }
    auto cap = [&](std::size_t k, const Vec3& n) {
        const Ring& rg = rings[k];
        const u32 centre = static_cast<u32>(m.vertices.size());
        m.vertices.push_back(make_vertex(rg.c, n, 0.0f, 0.0f, rg.r, rg.u));
        const std::size_t cnt = np - 1;
        for (std::size_t i = 0; i < cnt; ++i) {
            const P2& p = prof[k][i];
            m.vertices.push_back(make_vertex(pos[k][i], n, p.x * kUv, p.y * kUv, rg.r, rg.u));
        }
        for (std::size_t i = 0; i < cnt; ++i) {
            add_tri(m, centre, centre + 1 + static_cast<u32>(i), centre + 1 + static_cast<u32>((i + 1) % cnt));
        }
    };
    if (cap0) cap(0, normalize(rings[0].c - rings[1].c));
    if (cap1) cap(nr - 1, normalize(rings[nr - 1].c - rings[nr - 2].c));
}

// Axis-aligned bevelled box. The longest axis becomes the sweep (and texture "grain") direction.
void add_box(MeshData& m, const Vec3& c, const Vec3& h, f32 bevel, int segs = 2) {
    const Shape sh{true, bevel, segs};
    std::vector<Ring> rings;
    if (h.z >= h.x && h.z >= h.y) {
        rings = {{c + Vec3{0, 0, h.z}, {1, 0, 0}, {0, 1, 0}, h.x, h.y}, {c - Vec3{0, 0, h.z}, {1, 0, 0}, {0, 1, 0}, h.x, h.y}};
    } else if (h.x >= h.y) {
        rings = {{c - Vec3{h.x, 0, 0}, {0, 0, 1}, {0, 1, 0}, h.z, h.y}, {c + Vec3{h.x, 0, 0}, {0, 0, 1}, {0, 1, 0}, h.z, h.y}};
    } else {
        rings = {{c - Vec3{0, h.y, 0}, {1, 0, 0}, {0, 0, 1}, h.x, h.z}, {c + Vec3{0, h.y, 0}, {1, 0, 0}, {0, 0, 1}, h.x, h.z}};
    }
    add_sweep(m, rings, sh, true, true);
}

// Control point of a smooth path: position plus the two profile half-extents.
struct CP {
    Vec3 p;
    f32 a, b;
};

f32 catmull_scalar(f32 p0, f32 p1, f32 p2, f32 p3, f32 t) {
    return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t * t +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t * t * t);
}

std::vector<CP> catmull(const std::vector<CP>& pts, int sub) {
    std::vector<CP> out;
    const std::size_t n = pts.size();
    if (n < 2 || sub < 1) return pts;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const CP& p0 = pts[i > 0 ? i - 1 : 0];
        const CP& p1 = pts[i];
        const CP& p2 = pts[i + 1];
        const CP& p3 = pts[std::min(i + 2, n - 1)];
        for (int j = 0; j < sub; ++j) {
            const f32 t = static_cast<f32>(j) / static_cast<f32>(sub);
            out.push_back({{catmull_scalar(p0.p.x, p1.p.x, p2.p.x, p3.p.x, t), catmull_scalar(p0.p.y, p1.p.y, p2.p.y, p3.p.y, t),
                            catmull_scalar(p0.p.z, p1.p.z, p2.p.z, p3.p.z, t)},
                           catmull_scalar(p0.a, p1.a, p2.a, p3.a, t), catmull_scalar(p0.b, p1.b, p2.b, p3.b, t)});
        }
    }
    out.push_back(pts.back());
    return out;
}

// Rings with a fixed reference axis `r_axis` (must not be parallel to the path).
std::vector<Ring> make_rings(const std::vector<CP>& cps, const Vec3& r_axis) {
    std::vector<Ring> rings;
    const std::size_t n = cps.size();
    for (std::size_t k = 0; k < n; ++k) {
        const Vec3 t = normalize(cps[std::min(k + 1, n - 1)].p - cps[k > 0 ? k - 1 : 0].p);
        const Vec3 r = normalize(r_axis - t * dot(r_axis, t));
        rings.push_back({cps[k].p, r, normalize(cross(t, r)), cps[k].a, cps[k].b});
    }
    return rings;
}

void add_path(MeshData& m, const std::vector<CP>& ctrl, int sub, const Vec3& r_axis, const Shape& sh, bool caps = true) {
    add_sweep(m, make_rings(catmull(ctrl, sub), r_axis), sh, caps, caps);
}

// Round-section limb (finger, forearm, sleeve) with parallel-transported frames. With
// `round_ends` the ends are closed by hemispherical ring sequences, otherwise by flat caps.
void add_limb(MeshData& m, const std::vector<CP>& ctrl, int sub, int segs, bool round_ends, const Vec3& up_hint) {
    const std::vector<CP> cps = catmull(ctrl, sub);
    const std::size_t n = cps.size();
    std::vector<Ring> rings;
    Vec3 r_prev{};
    std::vector<Vec3> ts(n);
    for (std::size_t k = 0; k < n; ++k) {
        ts[k] = normalize(cps[std::min(k + 1, n - 1)].p - cps[k > 0 ? k - 1 : 0].p);
        Vec3 r;
        if (k == 0) {
            r = cross(up_hint, ts[0]);
            if (dot(r, r) < 1e-8f) r = cross(Vec3{1, 0, 0}, ts[0]);
        } else {
            r = r_prev - ts[k] * dot(r_prev, ts[k]);
        }
        r = normalize(r);
        r_prev = r;
        rings.push_back({cps[k].p, r, normalize(cross(ts[k], r)), cps[k].a, cps[k].b});
    }
    if (round_ends) {
        const std::array<f32, 5> phis = {90.0f, 72.0f, 54.0f, 36.0f, 18.0f};
        std::vector<Ring> head, tail;
        const Ring first = rings.front(), last = rings.back();
        const f32 r0 = 0.5f * (first.hw + first.hh), r1 = 0.5f * (last.hw + last.hh);
        for (const f32 ph : phis) {
            const f32 sn = std::sin(deg(ph)), cs = std::cos(deg(ph));
            Ring h = first;
            h.c = first.c - ts.front() * (r0 * sn);
            h.hw = first.hw * cs;
            h.hh = first.hh * cs;
            head.push_back(h);
            Ring t = last;
            t.c = last.c + ts.back() * (r1 * sn);
            t.hw = last.hw * cs;
            t.hh = last.hh * cs;
            tail.push_back(t);
        }
        std::vector<Ring> all = head;
        all.insert(all.end(), rings.begin(), rings.end());
        all.insert(all.end(), tail.rbegin(), tail.rend());
        rings = std::move(all);
    }
    add_sweep(m, rings, Shape{false, 0.0f, segs}, !round_ends, !round_ends);
}

struct Group {
    std::array<MeshData, kMat> meshes;
    MeshData& operator()(ViewMaterial k) { return meshes[static_cast<std::size_t>(k)]; }
    [[nodiscard]] std::vector<ViewPiece> pieces() {
        std::vector<ViewPiece> out;
        for (int i = 0; i < kMat; ++i) {
            MeshData& md = meshes[static_cast<std::size_t>(i)];
            if (!md.indices.empty()) out.push_back({std::move(md), static_cast<ViewMaterial>(i)});
        }
        return out;
    }
};

// ---------------------------------------------------------------------------------------------
// Rifle geometry (weapon space: -Z forward, bore axis on y = 0)
// ---------------------------------------------------------------------------------------------

constexpr f32 kSightY = 0.071f;
constexpr f32 kRearSightZ = -0.2085f;
constexpr f32 kFrontSightZ = -0.5485f;
constexpr f32 kMuzzleZ = -0.605f;
constexpr f32 kMagTopY = -0.028f;
constexpr f32 kMagTopZ = -0.1945f;
constexpr f32 kMagR = 0.45f;
constexpr f32 kMagTheta0 = 0.10f;
constexpr f32 kMagLen = 0.222f;
constexpr f32 kGripSlope = 0.35f / 1.10f; // z shift (backwards) per metre of descent

// Centre of the pistol grip at height y.
Vec3 grip_at(f32 y) { return {0.0f, y, -0.075f + (-0.035f - y) * kGripSlope}; }

void build_weapon(Group& g) {
    using M = ViewMaterial;
    MeshData& metal = g(M::Gunmetal);
    MeshData& wood = g(M::Wood);

    // Receiver: stamped body, dust cover, trunnion, tang.
    add_box(metal, {0.0f, -0.0105f, -0.1225f}, {0.0185f, 0.0245f, 0.1225f}, 0.0025f);
    add_path(metal, {{{0, 0.0235f, 0.003f}, 0.0190f, 0.0145f}, {{0, 0.0235f, -0.236f}, 0.0190f, 0.0145f}}, 1, {1, 0, 0},
             Shape{true, 0.0115f, 4});
    add_box(metal, {0.0f, 0.003f, -0.2275f}, {0.0197f, 0.0245f, 0.0285f}, 0.0065f, 3);
    add_box(metal, {0.0f, 0.003f, 0.008f}, {0.0135f, 0.0155f, 0.010f}, 0.004f);
    // Mag well collar and catch, trigger, trigger guard.
    add_box(metal, {0.0f, -0.0405f, -0.1955f}, {0.0200f, 0.0055f, 0.0255f}, 0.0025f);
    add_box(metal, {0.0f, -0.050f, -0.1665f}, {0.0045f, 0.0085f, 0.0025f}, 0.0015f);
    add_path(metal, {{{0, -0.034f, -0.080f}, 0.0065f, 0.0013f}, {{0, -0.054f, -0.088f}, 0.0065f, 0.0013f},
                     {{0, -0.0625f, -0.104f}, 0.0065f, 0.0013f}, {{0, -0.0635f, -0.135f}, 0.0065f, 0.0013f},
                     {{0, -0.058f, -0.155f}, 0.0065f, 0.0013f}, {{0, -0.045f, -0.166f}, 0.0065f, 0.0013f},
                     {{0, -0.034f, -0.172f}, 0.0065f, 0.0013f}},
             4, {1, 0, 0}, Shape{true, 0.001f, 2});
    add_path(metal, {{{0, -0.032f, -0.108f}, 0.0035f, 0.0016f}, {{0, -0.042f, -0.113f}, 0.0035f, 0.0016f},
                     {{0, -0.052f, -0.1175f}, 0.0035f, 0.0016f}, {{0, -0.059f, -0.1195f}, 0.0035f, 0.0016f}},
             4, {1, 0, 0}, Shape{true, 0.0012f, 2});
    // Stamped side rib and rivets on both flanks.
    for (const f32 sx : {-1.0f, 1.0f}) {
        add_box(metal, {sx * 0.0188f, -0.0225f, -0.105f}, {0.0007f, 0.0016f, 0.085f}, 0.0006f);
        add_box(metal, {sx * 0.0188f, 0.0100f, -0.140f}, {0.0007f, 0.0013f, 0.075f}, 0.0006f);
        for (const f32 rz : {-0.020f, -0.060f, -0.185f, -0.225f}) {
            add_path(metal, {{{sx * 0.0183f, -0.012f, rz}, 0.0022f, 0.0022f}, {{sx * 0.0197f, -0.012f, rz}, 0.0022f, 0.0022f}}, 1,
                     {0, 0, 1}, Shape{false, 0.0f, 10});
        }
    }
    // Safety lever and charging handle on the right side.
    add_box(metal, {0.0200f, 0.0f, -0.075f}, {0.0014f, 0.0035f, 0.046f}, 0.001f);
    add_box(metal, {0.0225f, 0.0100f, -0.030f}, {0.0032f, 0.0055f, 0.0075f}, 0.0015f);
    add_box(metal, {0.0200f, 0.0f, -0.1215f}, {0.0030f, 0.0050f, 0.0030f}, 0.0015f);
    add_box(metal, {0.0285f, 0.0115f, -0.092f}, {0.0105f, 0.0035f, 0.0035f}, 0.0015f);
    add_box(metal, {0.0405f, 0.0155f, -0.090f}, {0.0030f, 0.0075f, 0.0050f}, 0.002f);

    // Barrel, chamber collar, gas tube/block, cleaning rod, bayonet lug.
    const Shape circ{false, 0.0f, 20};
    add_path(metal, {{{0, 0, -0.235f}, 0.0088f, 0.0088f}, {{0, 0, kFrontSightZ - 0.016f}, 0.0088f, 0.0088f}}, 1, {1, 0, 0}, circ);
    add_path(metal, {{{0, 0, -0.240f}, 0.0125f, 0.0125f}, {{0, 0, -0.272f}, 0.0125f, 0.0125f}}, 1, {1, 0, 0}, circ);
    add_path(metal, {{{0, 0.0285f, -0.240f}, 0.0095f, 0.0095f}, {{0, 0.0285f, -0.452f}, 0.0095f, 0.0095f}}, 1, {1, 0, 0}, circ);
    add_box(metal, {0.0f, 0.0145f, -0.436f}, {0.0105f, 0.0255f, 0.016f}, 0.004f, 3);
    add_limb(metal, {{{0, -0.0118f, -0.440f}, 0.0028f, 0.0028f}, {{0, -0.0118f, -0.520f}, 0.0028f, 0.0028f}}, 1, 10, true, {1, 0, 0});
    add_box(metal, {0.0f, -0.0125f, -0.522f}, {0.0042f, 0.0055f, 0.0045f}, 0.0015f);

    // Muzzle brake: ringed tube, cut at a slant after generation.
    {
        std::vector<CP> pts;
        for (int i = 0; i <= 15; ++i) {
            const f32 z = -0.559f - 0.004f * static_cast<f32>(i);
            const bool groove = (i == 3 || i == 4 || i == 8 || i == 9);
            const f32 r = groove ? 0.0094f : 0.0106f;
            pts.push_back({{0, 0, z}, r, r});
        }
        MeshData brake;
        add_sweep(brake, make_rings(pts, {1, 0, 0}), Shape{false, 0.0f, 20}, true, true);
        // z = a + b*x plane; everything in front of it is projected onto it.
        const f32 a = kMuzzleZ, b = -0.006f / 0.0106f;
        const Vec3 pn = normalize(Vec3{b, 0.0f, -1.0f});
        for (Vertex& v : brake.vertices) {
            const f32 zc = a + b * v.position.x;
            if (v.position.z < zc) {
                const bool capv = v.normal.z < -0.9f;
                v.position.z = zc;
                if (capv) {
                    v.normal = pn;
                    const Vec3 t = normalize(Vec3{1.0f, 0.0f, b} - pn * dot(pn, Vec3{1.0f, 0.0f, b}));
                    v.tangent = {t.x, t.y, t.z, v.tangent.w};
                }
            }
        }
        const u32 off = static_cast<u32>(metal.vertices.size());
        metal.vertices.insert(metal.vertices.end(), brake.vertices.begin(), brake.vertices.end());
        for (const u32 i : brake.indices) metal.indices.push_back(i + off);
    }

    // Front sight: base block, post and protective ears.
    add_box(metal, {0.0f, 0.012f, kFrontSightZ}, {0.0100f, 0.0160f, 0.0165f}, 0.0035f, 3);
    add_box(metal, {0.0f, 0.0425f, kFrontSightZ}, {0.0017f, 0.0285f, 0.0015f}, 0.0007f);
    add_box(metal, {0.0072f, 0.0425f, kFrontSightZ}, {0.0012f, 0.0160f, 0.0075f}, 0.0008f);
    add_box(metal, {-0.0072f, 0.0425f, kFrontSightZ}, {0.0012f, 0.0160f, 0.0075f}, 0.0008f);

    // Rear sight: block, sloped leaf, notch ears.
    add_box(metal, {0.0f, 0.0235f, -0.2545f}, {0.0125f, 0.0245f, 0.0185f}, 0.0035f, 3);
    add_path(metal, {{{0, 0.0475f, -0.250f}, 0.0090f, 0.0022f}, {{0, 0.0570f, -0.2125f}, 0.0090f, 0.0022f}}, 1, {1, 0, 0},
             Shape{true, 0.0012f, 2});
    add_box(metal, {0.0058f, 0.0635f, kRearSightZ}, {0.0030f, 0.0075f, 0.0024f}, 0.0012f);
    add_box(metal, {-0.0058f, 0.0635f, kRearSightZ}, {0.0030f, 0.0075f, 0.0024f}, 0.0012f);
    add_box(metal, {0.0f, 0.0455f, -0.2300f}, {0.0090f, 0.0090f, 0.0105f}, 0.0015f);

    // Wooden handguards. Lower one has finger grooves.
    {
        std::vector<CP> pts;
        for (int i = 0; i <= 42; ++i) {
            const f32 z = -0.255f - 0.004f * static_cast<f32>(i);
            f32 groove = 0.0f;
            for (const f32 gz : {-0.318f, -0.346f, -0.374f}) {
                const f32 d = (z - gz) / 0.0045f;
                groove += std::exp(-d * d);
            }
            const f32 taper = 0.0020f * smooth((-z - 0.385f) / 0.04f);
            pts.push_back({{0, -0.011f, z}, 0.0205f - taper - 0.0016f * groove, 0.0210f - taper - 0.0016f * groove});
        }
        add_sweep(wood, make_rings(pts, {1, 0, 0}), Shape{true, 0.0105f, 4}, true, true);
        std::vector<CP> up;
        for (int i = 0; i <= 30; ++i) {
            const f32 z = -0.272f - 0.005f * static_cast<f32>(i);
            const f32 taper = 0.0015f * smooth((-z - 0.40f) / 0.02f);
            up.push_back({{0, 0.0225f, z}, 0.0172f - taper, 0.0175f - taper});
        }
        add_sweep(wood, make_rings(up, {1, 0, 0}), Shape{true, 0.0105f, 5}, true, true);
    }

    // Pistol grip (leaning back), and fixed stock with butt plate.
    {
        const Vec3 top = grip_at(-0.030f);
        const Vec3 bot = grip_at(-0.150f);
        std::vector<CP> gp = {{top, 0.0160f, 0.0200f}, {grip_at(-0.060f), 0.0180f, 0.0215f}, {grip_at(-0.100f), 0.0185f, 0.0222f},
                              {grip_at(-0.130f), 0.0182f, 0.0215f}, {bot, 0.0170f, 0.0195f}};
        add_path(wood, gp, 4, {1, 0, 0}, Shape{true, 0.0085f, 4});
        // grip cap
        add_box(metal, grip_at(-0.1505f), {0.0172f, 0.0022f, 0.0205f}, 0.0015f);

        std::vector<CP> st = {{{0, -0.005f, 0.002f}, 0.0175f, 0.0350f}, {{0, -0.012f, 0.060f}, 0.0185f, 0.0400f},
                              {{0, -0.022f, 0.140f}, 0.0185f, 0.0440f}, {{0, -0.0365f, 0.220f}, 0.0200f, 0.0485f},
                              {{0, -0.054f, 0.270f}, 0.0215f, 0.0580f}};
        std::vector<Ring> sr;
        for (const CP& cp : catmull(st, 4)) sr.push_back({cp.p, {1, 0, 0}, {0, 1, 0}, cp.a, cp.b});
        // Butt plate is canted: toe further back than the heel.
        const f32 tilt = deg(8.0f);
        const Vec3 up_t{0.0f, std::cos(tilt), -std::sin(tilt)};
        const Vec3 plate_n{0.0f, std::sin(tilt), std::cos(tilt)};
        sr.back().u = up_t;
        sr.back().hh = 0.0580f / std::cos(tilt);
        add_sweep(wood, sr, Shape{true, 0.0065f, 4}, true, false);
        const Vec3 pc = sr.back().c;
        add_sweep(metal,
                  {{pc, {1, 0, 0}, up_t, 0.0215f, sr.back().hh}, {pc + plate_n * 0.0010f, {1, 0, 0}, up_t, 0.0222f, sr.back().hh + 0.0008f},
                   {pc + plate_n * 0.0065f, {1, 0, 0}, up_t, 0.0222f, sr.back().hh + 0.0008f}},
                  Shape{true, 0.0025f, 3}, false, true);
    }
}

void build_magazine(Group& g) {
    using M = ViewMaterial;
    MeshData& metal = g(M::Gunmetal);
    const f32 th0 = kMagTheta0;
    auto centre = [&](f32 s) {
        const f32 th = th0 + s / kMagR;
        return Vec3{0.0f, kMagTopY - kMagR * (std::sin(th) - std::sin(th0)), kMagTopZ - kMagR * (std::cos(th0) - std::cos(th))};
    };
    auto fore = [&](f32 s) { // unit vector in the mag plane pointing to the rear
        const f32 th = th0 + s / kMagR;
        return Vec3{0.0f, -std::sin(th), std::cos(th)};
    };
    auto body = [&](f32 s0, f32 s1, int n, f32 hw, f32 hh0, f32 hh1, f32 rad, int segs) {
        std::vector<CP> pts;
        for (int i = 0; i <= n; ++i) {
            const f32 s = s0 + (s1 - s0) * static_cast<f32>(i) / static_cast<f32>(n);
            pts.push_back({centre(s), hw, hh0 + (hh1 - hh0) * (s / kMagLen)});
        }
        add_sweep(metal, make_rings(pts, {1, 0, 0}), Shape{true, rad, segs}, true, true);
    };
    body(0.0f, kMagLen - 0.006f, 26, 0.0182f, 0.0225f, 0.0208f, 0.0035f, 3);
    body(kMagLen - 0.007f, kMagLen, 2, 0.0195f, 0.0227f, 0.0212f, 0.003f, 3); // floor plate
    body(0.0f, 0.012f, 2, 0.0187f, 0.0230f, 0.0230f, 0.003f, 3);               // top lip
    body(0.050f, 0.056f, 2, 0.0187f, 0.0225f, 0.0225f, 0.003f, 3);             // reinforcing band
    // Stamped ribs on both sides.
    for (const f32 sx : {-1.0f, 1.0f}) {
        for (const f32 off : {-0.0085f, 0.0085f}) {
            std::vector<CP> pts;
            for (int i = 0; i <= 14; ++i) {
                const f32 s = 0.016f + 0.19f * static_cast<f32>(i) / 14.0f;
                const f32 hh = 0.0225f + (0.0208f - 0.0225f) * (s / kMagLen);
                pts.push_back({centre(s) + Vec3{sx * 0.0183f, 0.0f, 0.0f} + fore(s) * (off * hh / 0.0225f), 0.0016f, 0.0022f});
            }
            add_sweep(metal, make_rings(pts, {1, 0, 0}), Shape{true, 0.0012f, 2}, true, true);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Hands and sleeves
// ---------------------------------------------------------------------------------------------

void sleeve_and_forearm(Group& g, const Vec3& wrist, const Vec3& dir_in, f32 sleeve_len, f32 end_radius) {
    const Vec3 d = normalize(dir_in);
    MeshData& skin = g(ViewMaterial::Skin);
    MeshData& cloth = g(ViewMaterial::Sleeve);
    add_limb(skin, {{wrist, 0.0235f, 0.0235f}, {wrist + d * 0.04f, 0.0245f, 0.0245f}, {wrist + d * 0.12f, 0.0265f, 0.0265f}}, 3, 18,
             true, {0, 1, 0});
    std::vector<CP> sl;
    const Vec3 s0 = wrist + d * 0.050f;
    sl.push_back({s0, 0.0315f, 0.0315f});
    sl.push_back({s0 + d * 0.004f, 0.0335f, 0.0335f});
    sl.push_back({s0 + d * 0.016f, 0.0337f, 0.0337f});
    sl.push_back({s0 + d * 0.022f, 0.0315f, 0.0315f});
    sl.push_back({s0 + d * 0.10f, 0.0345f, 0.0345f});
    sl.push_back({s0 + d * (sleeve_len * 0.5f), 0.0385f, 0.0385f});
    sl.push_back({s0 + d * sleeve_len, end_radius, end_radius});
    add_limb(cloth, sl, 4, 26, false, {0, 1, 0});
}

void build_right_hand(Group& g) {
    MeshData& skin = g(ViewMaterial::Skin);
    const f32 a = 0.0178f, b = 0.0212f; // grip half-extents (x, z)
    const Vec3 grip_axis = normalize(Vec3{0.0f, -0.110f, 0.035f});

    // Palm wrapped around the back of the grip.
    {
        const Vec3 gc = grip_at(-0.083f);
        std::vector<CP> pts;
        const std::array<f32, 6> th = {-150.0f, -125.0f, -100.0f, -75.0f, -50.0f, -30.0f};
        for (std::size_t i = 0; i < th.size(); ++i) {
            const f32 t = deg(th[i]);
            const bool end = i == 0 || i + 1 == th.size();
            pts.push_back({gc + Vec3{std::cos(t) * (a + 0.0112f), 0.0f, -std::sin(t) * (b + 0.0112f)}, end ? 0.0095f : 0.0105f,
                           end ? 0.040f : 0.050f});
        }
        add_limb(skin, pts, 4, 20, true, grip_axis);
    }

    // Three wrapping fingers.
    struct Finger {
        f32 y, r0, r1;
    };
    const std::array<Finger, 3> fingers = {{{-0.0715f, 0.0085f, 0.0068f}, {-0.0875f, 0.0082f, 0.0066f}, {-0.1030f, 0.0072f, 0.0058f}}};
    const std::array<f32, 6> th = {-52.0f, -17.0f, 18.0f, 53.0f, 88.0f, 115.0f};
    for (const Finger& f : fingers) {
        std::vector<CP> pts;
        for (std::size_t i = 0; i < th.size(); ++i) {
            const f32 t = deg(th[i]);
            const f32 r = f.r0 + (f.r1 - f.r0) * static_cast<f32>(i) / 5.0f;
            const Vec3 c = grip_at(f.y - 0.0008f * static_cast<f32>(i));
            pts.push_back({c + Vec3{std::cos(t) * (a + r * 0.95f), 0.0f, -std::sin(t) * (b + r * 0.95f)}, r, r});
        }
        add_limb(skin, pts, 4, 14, true, {0, 1, 0});
    }
    // Index finger resting on the trigger.
    add_limb(skin,
             {{{0.0190f, -0.054f, -0.036f}, 0.0092f, 0.0092f}, {{0.0295f, -0.0545f, -0.071f}, 0.0086f, 0.0086f},
              {{0.0275f, -0.054f, -0.096f}, 0.0080f, 0.0080f}, {{0.0180f, -0.0515f, -0.114f}, 0.0076f, 0.0076f},
              {{0.0060f, -0.0500f, -0.1275f}, 0.0068f, 0.0068f}},
             4, 14, true, {0, 1, 0});
    // Thumb across the back of the grip to its left side.
    add_limb(skin,
             {{{0.026f, -0.056f, -0.028f}, 0.0100f, 0.0100f}, {{0.010f, -0.0485f, -0.037f}, 0.0092f, 0.0092f},
              {{-0.012f, -0.0475f, -0.046f}, 0.0085f, 0.0085f}, {{-0.0255f, -0.0495f, -0.062f}, 0.0075f, 0.0075f}},
             4, 14, true, {0, 1, 0});
    sleeve_and_forearm(g, {0.022f, -0.115f, -0.020f}, {0.30f, -0.30f, 0.90f}, 0.46f, 0.042f);
}

void build_left_hand(Group& g) {
    MeshData& skin = g(ViewMaterial::Skin);
    // Palm cupped under the lower handguard (path along x, ring axes: z extent, y extent).
    add_limb(skin,
             {{{-0.060f, -0.0455f, -0.340f}, 0.032f, 0.0100f}, {{-0.035f, -0.0455f, -0.338f}, 0.043f, 0.0135f},
              {{0.000f, -0.0455f, -0.338f}, 0.045f, 0.0135f}, {{0.030f, -0.0460f, -0.337f}, 0.043f, 0.0125f},
              {{0.040f, -0.0475f, -0.337f}, 0.036f, 0.0105f}},
             4, 18, true, {0, 1, 0});

    struct Finger {
        f32 z, r0, r1, len;
    };
    const std::array<Finger, 4> fingers = {{{-0.368f, 0.0088f, 0.0070f, 0.92f}, {-0.351f, 0.0090f, 0.0070f, 1.0f}, {-0.334f, 0.0084f, 0.0068f, 0.93f}, {-0.317f, 0.0074f, 0.0060f, 0.78f}}};
    for (const Finger& f : fingers) {
        const std::array<Vec3, 6> xy = {{{0.034f, -0.047f, 0}, {0.0325f, -0.036f, 0}, {0.0295f, -0.022f, 0}, {0.0292f, -0.008f, 0}, {0.0285f, 0.006f, 0}, {0.0245f, 0.0185f, 0}}};
        std::vector<CP> pts;
        for (std::size_t i = 0; i < xy.size(); ++i) {
            const f32 r = f.r0 + (f.r1 - f.r0) * static_cast<f32>(i) / 5.0f;
            pts.push_back({{xy[i].x, -0.047f + (xy[i].y + 0.047f) * f.len, f.z}, r, r});
        }
        add_limb(skin, pts, 4, 14, true, {0, 0, 1});
    }
    // Thumb along the left side of the handguard.
    add_limb(skin,
             {{{-0.040f, -0.047f, -0.312f}, 0.0098f, 0.0098f}, {{-0.0325f, -0.035f, -0.338f}, 0.0092f, 0.0092f},
              {{-0.0305f, -0.022f, -0.362f}, 0.0086f, 0.0086f}, {{-0.0300f, -0.010f, -0.385f}, 0.0078f, 0.0078f}},
             4, 14, true, {0, 1, 0});
    sleeve_and_forearm(g, {-0.058f, -0.050f, -0.325f}, {-0.40f, -0.30f, 0.866f}, 0.56f, 0.043f);
}

// ---------------------------------------------------------------------------------------------
// Muzzle flash
// ---------------------------------------------------------------------------------------------

void add_quad(MeshData& m, const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    const Vec3 n = normalize(cross(p1 - p0, p3 - p0));
    const u32 base = static_cast<u32>(m.vertices.size());
    const Vec3 tdir = p1 - p0, vdir = p3 - p0;
    m.vertices.push_back(make_vertex(p0, n, 0.0f, 1.0f, tdir, vdir));
    m.vertices.push_back(make_vertex(p1, n, 1.0f, 1.0f, tdir, vdir));
    m.vertices.push_back(make_vertex(p2, n, 1.0f, 0.0f, tdir, vdir));
    m.vertices.push_back(make_vertex(p3, n, 0.0f, 0.0f, tdir, vdir));
    add_tri(m, base, base + 1, base + 2);
    add_tri(m, base, base + 2, base + 3);
}

void build_flash(Group& g, const Vec3& muzzle) {
    MeshData& m = g(ViewMaterial::Flash);
    // Four crossed jets pointing along -Z.
    for (int k = 0; k < 4; ++k) {
        const f32 an = deg(45.0f * static_cast<f32>(k));
        const Vec3 d{std::cos(an), std::sin(an), 0.0f};
        const f32 hw = 0.040f;
        const f32 z0 = muzzle.z + 0.015f, z1 = muzzle.z - 0.125f;
        add_quad(m, muzzle + d * (-hw) + Vec3{0, 0, z0 - muzzle.z}, muzzle + d * hw + Vec3{0, 0, z0 - muzzle.z},
                 muzzle + d * hw + Vec3{0, 0, z1 - muzzle.z}, muzzle + d * (-hw) + Vec3{0, 0, z1 - muzzle.z});
    }
    // Spiky star disc at the muzzle, facing forward.
    const u32 centre = static_cast<u32>(m.vertices.size());
    const Vec3 n{0, 0, -1};
    const Vec3 cpos = muzzle + Vec3{0, 0, -0.012f};
    m.vertices.push_back(make_vertex(cpos, n, 0.5f, 0.5f, {1, 0, 0}, {0, 1, 0}));
    const int spikes = 8;
    const int cnt = spikes * 2;
    for (int i = 0; i < cnt; ++i) {
        const f32 an = kPi * static_cast<f32>(i) / static_cast<f32>(spikes);
        const f32 rad = (i % 2 == 0) ? 0.070f : 0.026f;
        const f32 ur = rad / 0.075f * 0.5f;
        m.vertices.push_back(make_vertex(cpos + Vec3{std::cos(an) * rad, std::sin(an) * rad, 0.0f}, n, 0.5f + std::cos(an) * ur,
                                         0.5f - std::sin(an) * ur, {1, 0, 0}, {0, 1, 0}));
    }
    for (int i = 0; i < cnt; ++i) add_tri(m, centre, centre + 1 + static_cast<u32>(i), centre + 1 + static_cast<u32>((i + 1) % cnt));
}

// ---------------------------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------------------------

u32 hash_u(u32 x, u32 y, u32 seed) {
    u32 h = x * 374761393u + y * 668265263u + seed * 2246822519u + 0x9e3779b9u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = (h ^ (h >> 16)) * 2246822519u;
    return h ^ (h >> 15);
}
f32 hash_f(u32 x, u32 y, u32 seed) { return static_cast<f32>(hash_u(x, y, seed) & 0xFFFFFFu) / 16777216.0f; }

// Periodic value noise: lattice of px x py cells wraps seamlessly.
f32 vnoise(f32 x, f32 y, int px, int py, u32 seed) {
    const f32 fx = std::floor(x), fy = std::floor(y);
    const f32 tx = smooth(x - fx), ty = smooth(y - fy);
    const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
    auto wrap = [](int v, int p) { return static_cast<u32>(((v % p) + p) % p); };
    const u32 x0 = wrap(ix, px), x1 = wrap(ix + 1, px), y0 = wrap(iy, py), y1 = wrap(iy + 1, py);
    const f32 a = hash_f(x0, y0, seed), b = hash_f(x1, y0, seed), c = hash_f(x0, y1, seed), d = hash_f(x1, y1, seed);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
}

f32 fbm(f32 u, f32 v, int fx, int fy, int octaves, u32 seed) {
    f32 sum = 0.0f, amp = 0.5f, norm = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * vnoise(u * static_cast<f32>(fx), v * static_cast<f32>(fy), fx, fy, seed + static_cast<u32>(o) * 131u);
        norm += amp;
        fx *= 2;
        fy *= 2;
        amp *= 0.5f;
    }
    return sum / norm;
}

struct Rng {
    u32 s;
    f32 next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<f32>((s >> 8) & 0xFFFFFFu) / 16777216.0f;
    }
};

u8 to8(f32 v) { return static_cast<u8>(std::lround(clamp01(v) * 255.0f)); }
f32 lin2srgb(f32 c) {
    c = clamp01(c);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

struct Canvas {
    int w, h;
    std::vector<f32> height;
    std::vector<Vec3> albedo; // linear
    std::vector<f32> rough, metal, ao, alpha;
    Canvas(int w_, int h_)
        : w(w_), h(h_), height(static_cast<std::size_t>(w_ * h_)), albedo(static_cast<std::size_t>(w_ * h_)),
          rough(static_cast<std::size_t>(w_ * h_), 0.5f), metal(static_cast<std::size_t>(w_ * h_), 0.0f),
          ao(static_cast<std::size_t>(w_ * h_), 1.0f), alpha(static_cast<std::size_t>(w_ * h_), 1.0f) {}
    [[nodiscard]] std::size_t idx(int x, int y) const {
        return static_cast<std::size_t>(((y % h) + h) % h) * static_cast<std::size_t>(w) + static_cast<std::size_t>(((x % w) + w) % w);
    }
};

template <class F> void for_pixels(Canvas& c, F&& fn) {
    for (int y = 0; y < c.h; ++y) {
        for (int x = 0; x < c.w; ++x) {
            fn(x, y, (static_cast<f32>(x) + 0.5f) / static_cast<f32>(c.w), (static_cast<f32>(y) + 0.5f) / static_cast<f32>(c.h));
        }
    }
}

void draw_scratch(std::vector<f32>& map, int w, int h, f32 x0, f32 y0, f32 dx, f32 dy, int len, f32 val) {
    for (int i = 0; i < len; ++i) {
        const int x = static_cast<int>(x0 + dx * static_cast<f32>(i));
        const int y = static_cast<int>(y0 + dy * static_cast<f32>(i));
        const std::size_t k = static_cast<std::size_t>(((y % h) + h) % h) * static_cast<std::size_t>(w) + static_cast<std::size_t>(((x % w) + w) % w);
        map[k] = std::max(map[k], val);
    }
}


ViewmodelAssets::Textures finish(const Canvas& c, f32 normal_strength) {
    ViewmodelAssets::Textures t;
    const std::size_t px = static_cast<std::size_t>(c.w * c.h);
    for (ImageData* img : {&t.base_color, &t.metallic_roughness, &t.normal}) {
        img->width = static_cast<u32>(c.w);
        img->height = static_cast<u32>(c.h);
        img->rgba.assign(px * 4, 255);
    }
    for (int y = 0; y < c.h; ++y) {
        for (int x = 0; x < c.w; ++x) {
            const std::size_t i = c.idx(x, y);
            const std::size_t o = i * 4;
            t.base_color.rgba[o + 0] = to8(lin2srgb(c.albedo[i].x));
            t.base_color.rgba[o + 1] = to8(lin2srgb(c.albedo[i].y));
            t.base_color.rgba[o + 2] = to8(lin2srgb(c.albedo[i].z));
            t.base_color.rgba[o + 3] = to8(c.alpha[i]);
            t.metallic_roughness.rgba[o + 0] = to8(c.ao[i]);
            t.metallic_roughness.rgba[o + 1] = to8(c.rough[i]);
            t.metallic_roughness.rgba[o + 2] = to8(c.metal[i]);
            const f32 dx = (c.height[c.idx(x + 1, y)] - c.height[c.idx(x - 1, y)]) * 0.5f * normal_strength;
            const f32 dy = (c.height[c.idx(x, y + 1)] - c.height[c.idx(x, y - 1)]) * 0.5f * normal_strength;
            const Vec3 n = normalize(Vec3{-dx, -dy, 1.0f});
            t.normal.rgba[o + 0] = to8(n.x * 0.5f + 0.5f);
            t.normal.rgba[o + 1] = to8(n.y * 0.5f + 0.5f);
            t.normal.rgba[o + 2] = to8(n.z * 0.5f + 0.5f);
        }
    }
    return t;
}

Vec3 mix(const Vec3& a, const Vec3& b, f32 t) { return a + (b - a) * t; }

ViewmodelAssets::Textures make_gunmetal() {
    Canvas c(512, 512);
    std::vector<f32> scratch(static_cast<std::size_t>(c.w * c.h), 0.0f);
    Rng rng{1234u};
    for (int i = 0; i < 90; ++i) {
        const f32 x0 = rng.next() * 512.0f, y0 = rng.next() * 512.0f;
        const f32 an = (rng.next() - 0.5f) * 0.5f + (rng.next() < 0.25f ? 1.57f : 0.0f);
        draw_scratch(scratch, c.w, c.h, x0, y0, std::cos(an), std::sin(an), 20 + static_cast<int>(rng.next() * 160.0f), 0.4f + 0.6f * rng.next());
    }
    for_pixels(c, [&](int x, int y, f32 u, f32 v) {
        const std::size_t i = c.idx(x, y);
        const f32 base = fbm(u, v, 8, 8, 4, 11u);
        const f32 streak = fbm(u, v, 2, 160, 3, 23u);
        const f32 blot = fbm(u, v, 3, 3, 3, 37u);
        const f32 sc = scratch[i];
        const f32 speckle = hash_f(static_cast<u32>(x), static_cast<u32>(y), 77u) > 0.996f ? 1.0f : 0.0f;
        const f32 grey = 0.050f + 0.030f * streak + 0.012f * base + 0.020f * blot + 0.09f * sc * sc + 0.06f * speckle;
        c.albedo[i] = Vec3{grey * 0.93f, grey * 0.97f, grey * 1.10f};
        c.height[i] = 0.35f * base + 0.35f * streak - 0.7f * sc;
        c.rough[i] = clamp01(0.40f + 0.14f * blot - 0.10f * streak + 0.1f * base - 0.20f * sc);
        c.metal[i] = clamp01(0.90f + 0.08f * streak - 0.15f * speckle);
    });
    return finish(c, 3.0f);
}

ViewmodelAssets::Textures make_wood() {
    Canvas c(512, 512);
    const Vec3 dark{0.085f, 0.022f, 0.007f}, light{0.270f, 0.095f, 0.030f};
    for_pixels(c, [&](int x, int y, f32 u, f32 v) {
        const std::size_t i = c.idx(x, y);
        const f32 warp = fbm(u, v, 2, 6, 3, 5u);
        const f32 warp2 = fbm(u, v, 1, 3, 2, 9u);
        const f32 ring = 0.5f + 0.5f * std::sin(2.0f * kPi * (v * 14.0f + warp * 2.4f + warp2 * 1.6f));
        const f32 fibre = fbm(u, v, 4, 128, 2, 17u);
        const f32 fine = vnoise(u * 8.0f, v * 256.0f, 8, 256, 29u);
        const f32 pore = smooth((vnoise(u * 24.0f, v * 512.0f, 24, 512, 41u) - 0.86f) / 0.07f); // thin streaks along the grain
        const f32 tone = clamp01(0.50f * ring * ring + 0.30f * fibre + 0.20f * fine);
        const f32 lacquer = fbm(u, v, 3, 3, 3, 53u);
        Vec3 col = mix(dark, light, tone);
        col = col * (0.85f + 0.3f * lacquer) * (1.0f - 0.22f * pore);
        c.albedo[i] = col;
        c.height[i] = 0.30f * fibre + 0.25f * ring + 0.1f * fine - 0.25f * pore;
        c.rough[i] = clamp01(0.40f + 0.10f * (1.0f - tone) + 0.12f * lacquer + 0.15f * pore);
        c.metal[i] = 0.0f;
        c.ao[i] = 1.0f - 0.15f * pore;
    });
    return finish(c, 1.6f);
}

ViewmodelAssets::Textures make_skin() {
    Canvas c(256, 256);
    const Vec3 tone{0.46f, 0.205f, 0.13f};
    for_pixels(c, [&](int x, int y, f32 u, f32 v) {
        const std::size_t i = c.idx(x, y);
        const f32 blotch = fbm(u, v, 4, 4, 3, 61u);
        const f32 red = fbm(u, v, 6, 6, 2, 67u);
        const f32 pore = vnoise(u * 64.0f, v * 64.0f, 64, 64, 71u);
        const f32 fine = fbm(u, v, 32, 32, 2, 73u);
        Vec3 col = tone * (0.88f + 0.24f * blotch);
        col = mix(col, Vec3{0.52f, 0.17f, 0.11f}, 0.35f * smooth((red - 0.5f) * 2.0f + 0.5f));
        col = col * (1.0f - 0.10f * (pore > 0.85f ? 1.0f : 0.0f));
        c.albedo[i] = col;
        c.height[i] = 0.6f * fine - 0.8f * (pore > 0.85f ? 1.0f : 0.0f);
        c.rough[i] = clamp01(0.58f + 0.06f * fine - 0.04f * blotch);
        c.metal[i] = 0.0f;
    });
    return finish(c, 1.2f);
}

ViewmodelAssets::Textures make_sleeve() {
    Canvas c(512, 512);
    const Vec3 olive{0.092f, 0.100f, 0.036f};
    const f32 threads = 128.0f;
    for_pixels(c, [&](int x, int y, f32 u, f32 v) {
        const std::size_t i = c.idx(x, y);
        const f32 su = u * threads, sv = v * threads;
        const int iu = static_cast<int>(su), iv = static_cast<int>(sv);
        const f32 a = std::sin(kPi * (su - static_cast<f32>(iu)));
        const f32 b = std::sin(kPi * (sv - static_cast<f32>(iv)));
        const f32 weave = ((iu + iv) & 1) != 0 ? a : b;
        const f32 lump = fbm(u, v, 64, 64, 2, 83u);
        const f32 dirt = smooth((fbm(u, v, 4, 4, 4, 89u) - 0.42f) * 3.0f);
        const f32 fade = fbm(u, v, 3, 5, 3, 97u);
        Vec3 col = olive * (0.70f + 0.30f * weave + 0.25f * lump);
        col = mix(col, hadamard(col, Vec3{1.25f, 1.20f, 1.30f}), 0.5f * fade);
        col = mix(col, Vec3{0.05f, 0.040f, 0.026f}, 0.55f * dirt);
        c.albedo[i] = col;
        c.height[i] = 0.7f * weave + 0.3f * lump;
        c.rough[i] = clamp01(0.88f + 0.07f * lump - 0.04f * weave);
        c.metal[i] = 0.0f;
        c.ao[i] = 0.75f + 0.25f * weave;
    });
    return finish(c, 2.2f);
}

ViewmodelAssets::Textures make_flash() {
    Canvas c(64, 64);
    for_pixels(c, [&](int x, int y, f32 u, f32 v) {
        const std::size_t i = c.idx(x, y);
        const f32 d = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 2.0f;
        const f32 inten = std::pow(clamp01(1.0f - d), 1.6f);
        const Vec3 hot{1.0f, 0.95f, 0.75f}, mid{1.0f, 0.55f, 0.12f}, edge{0.9f, 0.22f, 0.03f};
        const f32 t = clamp01(inten * 1.4f);
        const Vec3 col = (t > 0.5f ? mix(mid, hot, (t - 0.5f) * 2.0f) : mix(edge, mid, t * 2.0f)) * clamp01(inten * 2.0f);
        c.albedo[i] = col;
        c.alpha[i] = clamp01(inten * 1.15f);
        c.rough[i] = 1.0f;
    });
    ViewmodelAssets::Textures t = finish(c, 0.0f);
    // The renderer treats this base texture as the emissive source too, so keep it in sRGB-encoded colour.
    return t;
}

} // namespace

ViewmodelAssets build_ak_viewmodel() {
    ViewmodelAssets out;
    {
        Group g;
        build_weapon(g);
        out.weapon = g.pieces();
    }
    {
        Group g;
        build_magazine(g);
        out.magazine = g.pieces();
    }
    {
        Group g;
        build_right_hand(g);
        out.right_hand = g.pieces();
    }
    {
        Group g;
        build_left_hand(g);
        out.left_hand = g.pieces();
    }
    out.muzzle = {0.0f, 0.0f, kMuzzleZ};
    out.rear_sight = {0.0f, kSightY, kRearSightZ};
    out.front_sight = {0.0f, kSightY, kFrontSightZ};
    out.magazine_pivot = {0.0f, kMagTopY - 0.007f, kMagTopZ - 0.0225f};
    out.grip = grip_at(-0.090f);
    {
        Group g;
        build_flash(g, out.muzzle);
        out.flash = g.pieces();
    }

    out.textures[static_cast<int>(ViewMaterial::Gunmetal)] = make_gunmetal();
    out.textures[static_cast<int>(ViewMaterial::Wood)] = make_wood();
    out.textures[static_cast<int>(ViewMaterial::Skin)] = make_skin();
    out.textures[static_cast<int>(ViewMaterial::Sleeve)] = make_sleeve();
    out.textures[static_cast<int>(ViewMaterial::Flash)] = make_flash();

    const Vec3 none{0.0f, 0.0f, 0.0f};
    out.factors[static_cast<int>(ViewMaterial::Gunmetal)] = {1.0f, 1.0f, none, false};
    out.factors[static_cast<int>(ViewMaterial::Wood)] = {1.0f, 1.0f, none, false};
    out.factors[static_cast<int>(ViewMaterial::Skin)] = {1.0f, 1.0f, none, false};
    out.factors[static_cast<int>(ViewMaterial::Sleeve)] = {1.0f, 1.0f, none, false};
    out.factors[static_cast<int>(ViewMaterial::Flash)] = {1.0f, 1.0f, {6.0f, 3.2f, 1.2f}, true};
    return out;
}

} // namespace guerrilla
