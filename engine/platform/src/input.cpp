#include "nova/platform/input.hpp"

namespace nova::platform {

void Input::begin_frame() noexcept {
    prev_keys_ = keys_;
    prev_buttons_ = buttons_;
    mouse_dx_ = mouse_dy_ = wheel_ = 0;
    dropped_files_.clear();
}

void Input::on_file_drop(std::string path) {
    if (!path.empty()) {
        dropped_files_.push_back(std::move(path));
    }
}

void Input::on_key(Key key, bool down) noexcept {
    if (key != Key::Unknown) {
        keys_[index(key)] = down;
    }
}

void Input::on_mouse_button(MouseButton button, bool down) noexcept { buttons_[index(button)] = down; }

void Input::on_mouse_move(f32 x, f32 y, f32 dx, f32 dy) noexcept {
    mouse_x_ = x;
    mouse_y_ = y;
    mouse_dx_ += dx;
    mouse_dy_ += dy;
}

void Input::on_mouse_wheel(f32 dy) noexcept { wheel_ += dy; }

void Input::reset() noexcept {
    keys_.fill(false);
    buttons_.fill(false);
}

} // namespace nova::platform
