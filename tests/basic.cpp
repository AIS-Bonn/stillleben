// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>

#include <stillleben/phong_pass.h>

#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Image.h>

#include <iostream>

using namespace Corrade;
using namespace Magnum;

// Could be so much nicer with concepts...
template<std::size_t size, class T>
std::ostream& operator<<(std::ostream& stream, const Magnum::Math::Vector<size, T>& value)
{
    Corrade::Utility::Debug{&stream, Corrade::Utility::Debug::Flag::NoNewlineAtTheEnd} << value;
}

#include "catch.hpp"

TEST_CASE("basic")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    // Load a mesh file
    auto mesh = std::make_shared<sl::Mesh>(context);
    mesh->load(PATH_TO_SOURCES "/tests/stanford_bunny/scene.gltf");

    // Create a scene
    sl::Scene scene(context, sl::ViewportSize(640, 480));

    // Instantiate the mesh to create a movable scene object
    auto object = sl::Object::instantiate(mesh);
    REQUIRE(object);

    // Render everything using a Phong shader
    sl::PhongPass phong;
    auto buffer = phong.render(scene);

    REQUIRE(buffer);

    CHECK(buffer->imageSize() == Magnum::Vector2i(640, 480));

    {
        PluginManager::Manager<Trade::AbstractImageConverter> manager;
        std::unique_ptr<Trade::AbstractImageConverter> converter =
            manager.loadAndInstantiate("PngImageConverter");
        if(!converter) Fatal{} << "Cannot load the PngImageConverter plugin";

        Image2D image = buffer->image({PixelFormat::RGBA8Unorm});
        converter->exportToFile(image, "/tmp/stillleben.png");
    }
}
