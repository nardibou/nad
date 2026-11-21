#include "nova/platform/input.hpp"
#include "nova/platform/window.hpp"

#include <doctest/doctest.h>

using namespace nova;
using namespace nova::platform;

TEST_CASE("input: pressed and released are edge-triggered per frame") {
    Input in;
    in.begin_frame();
    in.on_key(Key::W, true);
    CHECK(in.is_down(Key::W));
    CHECK(in.was_pressed(Key::W));

    in.begin_frame(); // still held
    CHECK(in.is_down(Key::W));
    CHECK_FALSE(in.was_pressed(Key::W));

    in.begin_frame();
    in.on_key(Key::W, false);
    CHECK_FALSE(in.is_down(Key::W));
    CHECK(in.was_released(Key::W));

    in.begin_frame();
    CHECK_FALSE(in.was_released(Key::W));
}

TEST_CASE("input: mouse deltas accumulate within a frame and clear on the next") {
    Input in;
    in.begin_frame();
    in.on_mouse_move(10, 20, 3, -1);
    in.on_mouse_move(12, 21, 2, 1);
    in.on_mouse_wheel(1.0f);
    in.on_mouse_button(MouseButton::Right, true);
    CHECK(in.mouse_x() == 12);
    CHECK(in.mouse_dx() == 5);
    CHECK(in.mouse_dy() == 0);
    CHECK(in.wheel_delta() == 1.0f);
    CHECK(in.was_pressed(MouseButton::Right));

    in.begin_frame();
    CHECK(in.mouse_dx() == 0);
    CHECK(in.wheel_delta() == 0);
    CHECK(in.mouse_x() == 12);
    CHECK(in.is_down(MouseButton::Right));
}

TEST_CASE("input: reset releases everything and Unknown is ignored") {
    Input in;
    in.on_key(Key::Space, true);
    in.on_mouse_button(MouseButton::Left, true);
    in.on_key(Key::Unknown, true);
    CHECK_FALSE(in.is_down(Key::Unknown));
    in.reset();
    CHECK_FALSE(in.is_down(Key::Space));
    CHECK_FALSE(in.is_down(MouseButton::Left));
}

TEST_CASE("window: headless backend runs until close is requested") {
    WindowDesc desc;
    desc.backend = Backend::Headless;
    desc.width = 320;
    desc.height = 200;
    auto window = create_window(desc);
    REQUIRE(window);
    CHECK(window->backend() == Backend::Headless);
    CHECK(window->size().width == 320);
    CHECK(window->native_handle() == nullptr);

    Input in;
    CHECK(window->poll_events(in));
    window->request_close();
    CHECK(window->should_close());
    CHECK_FALSE(window->poll_events(in));
}

TEST_CASE("window: unavailable backends fail cleanly") {
    if (!backend_available(Backend::SDL3)) {
        WindowDesc desc;
        desc.backend = Backend::SDL3;
        CHECK(create_window(desc) == nullptr);
    }
    CHECK(backend_available(Backend::Headless));
}
