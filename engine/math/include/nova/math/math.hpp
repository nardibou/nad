#pragma once

// Minimal linear algebra for the engine. Conventions:
// - right-handed world space, +Y up, camera looks down -Z
// - Mat4 is column-major (m[column][row]); vectors are columns: v' = M * v
// - perspective() targets Vulkan clip space: depth 0..1, Y down (flip baked in)

#include "nova/core/types.hpp"

#include <array>
#include <cmath>
#include <numbers>

namespace nova::math {

inline constexpr f32 kPi = std::numbers::pi_v<f32>;
[[nodiscard]] constexpr f32 radians(f32 degrees) noexcept { return degrees * kPi / 180.0f; }

struct Vec2 {
    f32 x = 0, y = 0;
    constexpr bool operator==(const Vec2&) const noexcept = default;
};

struct Vec3 {
    f32 x = 0, y = 0, z = 0;

    constexpr Vec3 operator+(const Vec3& o) const noexcept { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const noexcept { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(f32 s) const noexcept { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator-() const noexcept { return {-x, -y, -z}; }
    constexpr Vec3& operator+=(const Vec3& o) noexcept { return *this = *this + o; }
    constexpr Vec3& operator-=(const Vec3& o) noexcept { return *this = *this - o; }
    constexpr bool operator==(const Vec3&) const noexcept = default;
};

struct Vec4 {
    f32 x = 0, y = 0, z = 0, w = 0;
    constexpr bool operator==(const Vec4&) const noexcept = default;
};

[[nodiscard]] constexpr f32 dot(const Vec3& a, const Vec3& b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
[[nodiscard]] constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
[[nodiscard]] inline f32 length(const Vec3& v) noexcept { return std::sqrt(dot(v, v)); }
[[nodiscard]] inline Vec3 normalize(const Vec3& v) noexcept {
    const f32 len = length(v);
    return len > 0.0f ? v * (1.0f / len) : Vec3{};
}

struct Mat4 {
    std::array<std::array<f32, 4>, 4> m{}; // m[column][row]

    [[nodiscard]] static constexpr Mat4 identity() noexcept {
        Mat4 r;
        for (int i = 0; i < 4; ++i) {
            r.m[i][i] = 1.0f;
        }
        return r;
    }

    [[nodiscard]] constexpr Mat4 operator*(const Mat4& b) const noexcept {
        Mat4 r;
        for (int c = 0; c < 4; ++c) {
            for (int row = 0; row < 4; ++row) {
                f32 s = 0;
                for (int k = 0; k < 4; ++k) {
                    s += m[k][row] * b.m[c][k];
                }
                r.m[c][row] = s;
            }
        }
        return r;
    }

    [[nodiscard]] constexpr Vec4 operator*(const Vec4& v) const noexcept {
        const f32 in[4] = {v.x, v.y, v.z, v.w};
        f32 out[4] = {};
        for (int row = 0; row < 4; ++row) {
            for (int c = 0; c < 4; ++c) {
                out[row] += m[c][row] * in[c];
            }
        }
        return {out[0], out[1], out[2], out[3]};
    }

    [[nodiscard]] const f32* data() const noexcept { return &m[0][0]; }
};

[[nodiscard]] constexpr Mat4 translate(const Vec3& t) noexcept {
    Mat4 r = Mat4::identity();
    r.m[3][0] = t.x;
    r.m[3][1] = t.y;
    r.m[3][2] = t.z;
    return r;
}

[[nodiscard]] constexpr Mat4 scale(const Vec3& s) noexcept {
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}

// Rotation by `angle` radians around normalized `axis` (right-hand rule).
[[nodiscard]] inline Mat4 rotate(const Vec3& axis, f32 angle) noexcept {
    const Vec3 a = normalize(axis);
    const f32 c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    Mat4 r = Mat4::identity();
    r.m[0][0] = t * a.x * a.x + c;
    r.m[0][1] = t * a.x * a.y + s * a.z;
    r.m[0][2] = t * a.x * a.z - s * a.y;
    r.m[1][0] = t * a.x * a.y - s * a.z;
    r.m[1][1] = t * a.y * a.y + c;
    r.m[1][2] = t * a.y * a.z + s * a.x;
    r.m[2][0] = t * a.x * a.z + s * a.y;
    r.m[2][1] = t * a.y * a.z - s * a.x;
    r.m[2][2] = t * a.z * a.z + c;
    return r;
}

[[nodiscard]] inline Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up) noexcept {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x; r.m[1][0] = s.y; r.m[2][0] = s.z;
    r.m[0][1] = u.x; r.m[1][1] = u.y; r.m[2][1] = u.z;
    r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z;
    r.m[3][0] = -dot(s, eye);
    r.m[3][1] = -dot(u, eye);
    r.m[3][2] = dot(f, eye);
    return r;
}

// Vulkan clip space: x right, y down, depth in [0, 1] with near -> 0.
[[nodiscard]] inline Mat4 perspective(f32 fov_y, f32 aspect, f32 z_near, f32 z_far) noexcept {
    const f32 f = 1.0f / std::tan(fov_y * 0.5f);
    Mat4 r;
    r.m[0][0] = f / aspect;
    r.m[1][1] = -f;
    r.m[2][2] = z_far / (z_near - z_far);
    r.m[2][3] = -1.0f;
    r.m[3][2] = (z_near * z_far) / (z_near - z_far);
    return r;
}

// Orthographic projection into Vulkan clip space (same conventions as perspective()).
[[nodiscard]] constexpr Mat4 orthographic(f32 left, f32 right, f32 bottom, f32 top, f32 z_near, f32 z_far) noexcept {
    Mat4 r = Mat4::identity();
    r.m[0][0] = 2.0f / (right - left);
    r.m[1][1] = -2.0f / (top - bottom);
    r.m[2][2] = -1.0f / (z_far - z_near);
    r.m[3][0] = -(right + left) / (right - left);
    r.m[3][1] = (top + bottom) / (top - bottom);
    r.m[3][2] = -z_near / (z_far - z_near);
    return r;
}

[[nodiscard]] constexpr Vec3 transform_point(const Mat4& m, const Vec3& p) noexcept {
    const Vec4 r = m * Vec4{p.x, p.y, p.z, 1.0f};
    return {r.x, r.y, r.z};
}

// Transforms a normal by the inverse transpose of m's upper 3x3 (correct for non-uniform scale).
[[nodiscard]] inline Vec3 transform_normal(const Mat4& m, const Vec3& n) noexcept {
    const Vec3 c0{m.m[0][0], m.m[0][1], m.m[0][2]};
    const Vec3 c1{m.m[1][0], m.m[1][1], m.m[1][2]};
    const Vec3 c2{m.m[2][0], m.m[2][1], m.m[2][2]};
    // The cofactor matrix equals det * inverse-transpose; the scale vanishes after normalizing.
    const Vec3 r0 = cross(c1, c2), r1 = cross(c2, c0), r2 = cross(c0, c1);
    return normalize(r0 * n.x + r1 * n.y + r2 * n.z);
}

[[nodiscard]] constexpr f32 lerp(f32 a, f32 b, f32 t) noexcept { return a + (b - a) * t; }
[[nodiscard]] constexpr Vec3 lerp(const Vec3& a, const Vec3& b, f32 t) noexcept { return a + (b - a) * t; }

// Unit quaternion (x, y, z, w), glTF convention; identity = (0, 0, 0, 1).
struct Quat {
    f32 x = 0, y = 0, z = 0, w = 1;
    constexpr bool operator==(const Quat&) const noexcept = default;
};

[[nodiscard]] inline Quat normalize(const Quat& q) noexcept {
    const f32 len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return len > 0.0f ? Quat{q.x / len, q.y / len, q.z / len, q.w / len} : Quat{};
}

[[nodiscard]] constexpr Quat operator*(const Quat& a, const Quat& b) noexcept {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

[[nodiscard]] inline Quat axis_angle(const Vec3& axis, f32 angle) noexcept {
    const Vec3 a = normalize(axis);
    const f32 s = std::sin(angle * 0.5f);
    return {a.x * s, a.y * s, a.z * s, std::cos(angle * 0.5f)};
}

// Normalized lerp along the shorter arc; accurate enough for animation blending.
[[nodiscard]] inline Quat nlerp(const Quat& a, Quat b, f32 t) noexcept {
    if (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0.0f) {
        b = {-b.x, -b.y, -b.z, -b.w};
    }
    return normalize(Quat{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t});
}

[[nodiscard]] inline Quat slerp(const Quat& a, Quat b, f32 t) noexcept {
    f32 d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (d < 0.0f) {
        b = {-b.x, -b.y, -b.z, -b.w};
        d = -d;
    }
    if (d > 0.9995f) {
        return nlerp(a, b, t);
    }
    const f32 theta = std::acos(d);
    const f32 sa = std::sin((1.0f - t) * theta) / std::sin(theta), sb = std::sin(t * theta) / std::sin(theta);
    return {a.x * sa + b.x * sb, a.y * sa + b.y * sb, a.z * sa + b.z * sb, a.w * sa + b.w * sb};
}

[[nodiscard]] inline Mat4 to_mat4(const Quat& q) noexcept {
    const f32 x = q.x, y = q.y, z = q.z, w = q.w;
    Mat4 r = Mat4::identity();
    r.m[0][0] = 1 - 2 * (y * y + z * z);
    r.m[0][1] = 2 * (x * y + z * w);
    r.m[0][2] = 2 * (x * z - y * w);
    r.m[1][0] = 2 * (x * y - z * w);
    r.m[1][1] = 1 - 2 * (x * x + z * z);
    r.m[1][2] = 2 * (y * z + x * w);
    r.m[2][0] = 2 * (x * z + y * w);
    r.m[2][1] = 2 * (y * z - x * w);
    r.m[2][2] = 1 - 2 * (x * x + y * y);
    return r;
}

// Rotation of an orthonormal, right-handed basis given as matrix columns (scale must be removed).
[[nodiscard]] inline Quat to_quat(const Mat4& m) noexcept {
    const f32 m00 = m.m[0][0], m11 = m.m[1][1], m22 = m.m[2][2];
    const f32 trace = m00 + m11 + m22;
    Quat q;
    if (trace > 0.0f) {
        const f32 s = std::sqrt(trace + 1.0f) * 2.0f;
        q = {(m.m[1][2] - m.m[2][1]) / s, (m.m[2][0] - m.m[0][2]) / s, (m.m[0][1] - m.m[1][0]) / s, 0.25f * s};
    } else if (m00 > m11 && m00 > m22) {
        const f32 s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q = {0.25f * s, (m.m[1][0] + m.m[0][1]) / s, (m.m[2][0] + m.m[0][2]) / s, (m.m[1][2] - m.m[2][1]) / s};
    } else if (m11 > m22) {
        const f32 s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q = {(m.m[1][0] + m.m[0][1]) / s, 0.25f * s, (m.m[2][1] + m.m[1][2]) / s, (m.m[2][0] - m.m[0][2]) / s};
    } else {
        const f32 s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q = {(m.m[2][0] + m.m[0][2]) / s, (m.m[2][1] + m.m[1][2]) / s, 0.25f * s, (m.m[0][1] - m.m[1][0]) / s};
    }
    return normalize(q);
}

// translate(t) * rotate(r) * scale(s), the glTF node transform.
[[nodiscard]] inline Mat4 compose(const Vec3& t, const Quat& r, const Vec3& s) noexcept {
    Mat4 m = to_mat4(r);
    for (int row = 0; row < 3; ++row) {
        m.m[0][row] *= s.x;
        m.m[1][row] *= s.y;
        m.m[2][row] *= s.z;
    }
    m.m[3][0] = t.x;
    m.m[3][1] = t.y;
    m.m[3][2] = t.z;
    return m;
}

// Inverse of an affine matrix (last row 0 0 0 1); returns identity if singular.
[[nodiscard]] inline Mat4 inverse_affine(const Mat4& m) noexcept {
    const Vec3 c0{m.m[0][0], m.m[0][1], m.m[0][2]};
    const Vec3 c1{m.m[1][0], m.m[1][1], m.m[1][2]};
    const Vec3 c2{m.m[2][0], m.m[2][1], m.m[2][2]};
    const Vec3 t{m.m[3][0], m.m[3][1], m.m[3][2]};
    // Rows of the inverse 3x3 are the cofactor columns divided by the determinant.
    const Vec3 r0 = cross(c1, c2), r1 = cross(c2, c0), r2 = cross(c0, c1);
    const f32 det = dot(c0, r0);
    if (std::abs(det) < 1e-20f) {
        return Mat4::identity();
    }
    const f32 inv = 1.0f / det;
    Mat4 r = Mat4::identity();
    const Vec3 rows[3] = {r0 * inv, r1 * inv, r2 * inv};
    for (int row = 0; row < 3; ++row) {
        r.m[0][row] = rows[row].x;
        r.m[1][row] = rows[row].y;
        r.m[2][row] = rows[row].z;
        r.m[3][row] = -dot(rows[row], t);
    }
    return r;
}

} // namespace nova::math
