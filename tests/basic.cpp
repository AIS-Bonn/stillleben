// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>

#include <stillleben/phong_pass.h>

#include <iostream>

// Could be so much nicer with concepts...
template<std::size_t size, class T>
std::ostream& operator<<(std::ostream& stream, const Magnum::Math::Vector<size, T>& value)
{
    Corrade::Utility::Debug{&stream, Corrade::Utility::Debug::Flag::NoNewlineAtTheEnd} << value;
}

#include "catch.hpp"

TEST_CASE("basic")
{
    auto context = sl::Context::Create();
    REQUIRE(context);

    auto mesh = std::make_shared<sl::Mesh>(context);
    mesh->load(PATH_TO_SOURCES "/tests/stanford_bunny/scene.gltf");

    sl::Scene scene(context, sl::ViewportSize(640, 480));

    auto object = sl::Object::instantiate(mesh);
    REQUIRE(object);

    sl::PhongPass phong;
    auto buffer = phong.render(scene);

    REQUIRE(buffer);

    CHECK(buffer->imageSize() == Magnum::Vector2i(640, 480));
}
