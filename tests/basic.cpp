// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>

#include <stillleben/phong_pass.h>
#include <stillleben/render_pass.h>

#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Image.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/Math/Range.h>

#include <iostream>

using namespace Corrade;
using namespace Magnum;

// Could be so much nicer with concepts...
template<std::size_t size, class T>
std::ostream& operator<<(std::ostream& stream, const Magnum::Math::Vector<size, T>& value)
{
    Corrade::Utility::Debug{&stream, Corrade::Utility::Debug::Flag::NoNewlineAtTheEnd} << value;
}
template<class T>
std::ostream& operator<<(std::ostream& stream, const Magnum::Math::Range3D<T>& value)
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

    // Check mesh file details
    {
        CAPTURE(mesh->bbox());
        CHECK(std::isfinite(mesh->bbox().sizeX()));
        CHECK(std::isfinite(mesh->bbox().sizeY()));
        CHECK(std::isfinite(mesh->bbox().sizeZ()));

        mesh->centerBBox();
        CHECK(mesh->bbox().center().x() == Approx(0.0f));
        CHECK(mesh->bbox().center().y() == Approx(0.0f));
        CHECK(mesh->bbox().center().z() == Approx(0.0f));

        mesh->scaleToBBoxDiagonal(0.5);
        CHECK(mesh->bbox().size().length() == Approx(0.5f));
    }

    // Create a scene
    sl::Scene scene(context, sl::ViewportSize(640, 480));

    // Instantiate the mesh to create a movable scene object
    auto object = sl::Object::instantiate(mesh);
    REQUIRE(object);

    float distance = scene.minimumDistanceForObjectDiameter(mesh->bbox().size().length());

    object->setPose(Matrix4::translation(Vector3(0.0, 0.0, -distance)));

    // Add it to the scene
    scene.addObject(object);

    // Render everything using a Phong shader
    sl::PhongPass phong;
    auto buffer = phong.render(scene);

    REQUIRE(buffer);
    CHECK(buffer->imageSize() == Magnum::Vector2i(640, 480));

    Image2D image = buffer->image({PixelFormat::RGBA8Unorm});
    {
        PluginManager::Manager<Trade::AbstractImageConverter> manager;
        std::unique_ptr<Trade::AbstractImageConverter> converter =
            manager.loadAndInstantiate("PngImageConverter");
        if(!converter) Fatal{} << "Cannot load the PngImageConverter plugin";

        CHECK(converter->exportToFile(image, "/tmp/stillleben.png"));
    }

    {
        unsigned int nonTransparent = 0;
        REQUIRE(image.pixelSize() == 4);

        const auto data = image.data();

        for(int i = 0; i < image.size().product(); ++i)
        {
            uint8_t alpha = data[i*4 + 3];
            if(alpha != 0)
                nonTransparent++;
        }

        CHECK(nonTransparent > 10);
    }
}


TEST_CASE("render")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    // Load a mesh file
    auto mesh = std::make_shared<sl::Mesh>(context);
    mesh->load(PATH_TO_SOURCES "/tests/stanford_bunny/scene.gltf");

    // Check mesh file details
    {
        CAPTURE(mesh->bbox());
        CHECK(std::isfinite(mesh->bbox().sizeX()));
        CHECK(std::isfinite(mesh->bbox().sizeY()));
        CHECK(std::isfinite(mesh->bbox().sizeZ()));

        mesh->centerBBox();
        CHECK(mesh->bbox().center().x() == Approx(0.0f));
        CHECK(mesh->bbox().center().y() == Approx(0.0f));
        CHECK(mesh->bbox().center().z() == Approx(0.0f));

        mesh->scaleToBBoxDiagonal(0.5);
        CHECK(mesh->bbox().size().length() == Approx(0.5f));
    }

    // Create a scene
    sl::Scene scene(context, sl::ViewportSize(640, 480));

    // Instantiate the mesh to create a movable scene object
    auto object = sl::Object::instantiate(mesh);
    REQUIRE(object);

    float distance = scene.minimumDistanceForObjectDiameter(mesh->bbox().size().length());

    object->setPose(Matrix4::translation(Vector3(0.0, 0.0, -distance)));

    // Add it to the scene
    scene.addObject(object);

    // Render everything using a Phong shader
    sl::RenderPass pass;
    auto ret = pass.render(scene);

    auto buffer = &ret->rgb;

    REQUIRE(buffer);
    CHECK(buffer->imageSize() == Magnum::Vector2i(640, 480));

    PluginManager::Manager<Trade::AbstractImageConverter> manager;
    std::unique_ptr<Trade::AbstractImageConverter> converter =
        manager.loadAndInstantiate("PngImageConverter");
    if(!converter) Fatal{} << "Cannot load the PngImageConverter plugin";

    Image2D image = buffer->image({PixelFormat::RGBA8Unorm});
    CHECK(converter->exportToFile(image, "/tmp/stillleben_render.png"));

    {
        unsigned int nonTransparent = 0;
        REQUIRE(image.pixelSize() == 4);

        const auto data = image.data();

        for(int i = 0; i < image.size().product(); ++i)
        {
            uint8_t alpha = data[i*4 + 3];
            if(alpha != 0)
                nonTransparent++;
        }

        CHECK(nonTransparent > 10);
    }

    Image2D coordImage = ret->objectCoordinates.image({PixelFormat::RGBA8Unorm});
    CHECK(converter->exportToFile(coordImage, "/tmp/stillleben_coords.png"));

    Image2D validImage = ret->validMask.image({PixelFormat::R8Unorm});
    CHECK(converter->exportToFile(validImage, "/tmp/stillleben_valid.png"));

    {
        unsigned int nonTransparent = 0;
        REQUIRE(validImage.pixelSize() == 1);

        const auto data = validImage.data();

        for(int i = 0; i < 100; ++i)
            printf("%02X ", (uint8_t)data[i]);
        printf("\n");

//         for(int i = 0; i < image.size().product(); ++i)
//         {
//             uint8_t alpha = data[i*4 + 3];
//             if(alpha != 0)
//                 nonTransparent++;
//         }
    }
}
