#pragma once

#include "nova/core/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace nova::platform {

// Backend-independent key codes (physical layout, US names).
enum class Key : u16 {
    Unknown = 0,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Escape, Enter, Tab, Backspace, Delete, Space, Minus, Equal,
    Left, Right, Up, Down,
    LeftShift, RightShift, LeftCtrl, RightCtrl, LeftAlt, RightAlt,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Count
};

enum class MouseButton : u8 { Left, Right, Middle, X1, X2, Count };

// Per-frame input snapshot. Backends feed events between begin_frame() calls;
// gameplay code only reads it.
class Input {
public:
    // Promotes current state to "previous" and clears per-frame deltas. Call once per frame
    // before pumping events.
    void begin_frame() noexcept;

    void on_key(Key key, bool down) noexcept;
    void on_mouse_button(MouseButton button, bool down) noexcept;
    void on_mouse_move(f32 x, f32 y, f32 dx, f32 dy) noexcept;
    void on_mouse_wheel(f32 dy) noexcept;
    void on_file_drop(std::string path);
    void reset() noexcept; // e.g. on focus loss, so keys do not stick

    [[nodiscard]] bool is_down(Key key) const noexcept { return keys_[index(key)]; }
    [[nodiscard]] bool was_pressed(Key key) const noexcept { return keys_[index(key)] && !prev_keys_[index(key)]; }
    [[nodiscard]] bool was_released(Key key) const noexcept { return !keys_[index(key)] && prev_keys_[index(key)]; }

    [[nodiscard]] bool is_down(MouseButton b) const noexcept { return buttons_[index(b)]; }
    [[nodiscard]] bool was_pressed(MouseButton b) const noexcept { return buttons_[index(b)] && !prev_buttons_[index(b)]; }
    [[nodiscard]] bool was_released(MouseButton b) const noexcept { return !buttons_[index(b)] && prev_buttons_[index(b)]; }

    [[nodiscard]] f32 mouse_x() const noexcept { return mouse_x_; }
    [[nodiscard]] f32 mouse_y() const noexcept { return mouse_y_; }
    [[nodiscard]] f32 mouse_dx() const noexcept { return mouse_dx_; }
    [[nodiscard]] f32 mouse_dy() const noexcept { return mouse_dy_; }
    [[nodiscard]] f32 wheel_delta() const noexcept { return wheel_; }
    [[nodiscard]] const std::vector<std::string>& dropped_files() const noexcept { return dropped_files_; }

private:
    static constexpr usize kKeyCount = static_cast<usize>(Key::Count);
    static constexpr usize kButtonCount = static_cast<usize>(MouseButton::Count);
    static constexpr usize index(Key k) noexcept { return static_cast<usize>(k) < kKeyCount ? static_cast<usize>(k) : 0; }
    static constexpr usize index(MouseButton b) noexcept {
        return static_cast<usize>(b) < kButtonCount ? static_cast<usize>(b) : 0;
    }

    std::array<bool, kKeyCount> keys_{};
    std::array<bool, kKeyCount> prev_keys_{};
    std::array<bool, kButtonCount> buttons_{};
    std::array<bool, kButtonCount> prev_buttons_{};
    f32 mouse_x_ = 0, mouse_y_ = 0, mouse_dx_ = 0, mouse_dy_ = 0, wheel_ = 0;
    std::vector<std::string> dropped_files_;
};

} // namespace nova::platform
