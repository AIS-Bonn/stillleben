// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/pose.h>
#include <stillleben/mesh_cache.h>

#include <stillleben/phong_pass.h>
#include <stillleben/render_pass.h>

#include <Corrade/Utility/Configuration.h>

#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/Math/Range.h>
#include <Magnum/PixelFormat.h>

#include <iostream>
#include <iomanip>

#include "doctest.h"

using namespace Corrade;
using namespace Magnum;

using Approx = doctest::Approx;

#define DEF_TO_STRING(TYPE) doctest::String toString(const TYPE& t) { return toStringImpl<TYPE>(t); }

namespace Magnum
{
    template<class T>
    doctest::String toStringImpl(const T& value)
    {
        std::stringstream ss;
        Corrade::Utility::Debug{&ss, Corrade::Utility::Debug::Flag::NoNewlineAtTheEnd} << value;
        return ss.str().c_str();
    }

    DEF_TO_STRING(Matrix4)
    DEF_TO_STRING(Vector3)
}

constexpr const char* BUNNY = PATH_TO_SOURCES "/tests/stanford_bunny/scene.gltf";

TEST_CASE("basic")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    // Load a mesh file
    auto mesh = std::make_shared<sl::Mesh>(BUNNY, context);
    mesh->load();

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

        auto pretransform = mesh->pretransform();
        mesh->setPretransform(pretransform);
        auto newPretransform = mesh->pretransform();

        CAPTURE(pretransform);
        CAPTURE(newPretransform);

        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                CHECK(newPretransform[i][j] == Approx(pretransform[i][j]));
    }

    // Create a scene
    sl::Scene scene(context, sl::ViewportSize(640, 480));

    // Instantiate the mesh to create a movable scene object
    auto object = std::make_shared<sl::Object>();
    object->setMesh(mesh);
    object->loadVisual();

    float distance = sl::pose::minimumDistanceForObjectDiameter(
        mesh->bbox().size().length(),
        scene.projectionMatrix()
    );

    object->setPose(Matrix4::translation(Vector3(0.0, 0.0, distance)));

    // Add it to the scene
    scene.addObject(object);

    // Render everything using a Phong shader
    sl::PhongPass phong;
    auto buffer = phong.render(scene);

    REQUIRE(buffer);
    CHECK(buffer->imageSize() == Magnum::Vector2i(640, 480));

    Image2D image = buffer->image({PixelFormat::RGBA8Unorm});
    {
        Corrade::PluginManager::Manager<Magnum::Trade::AbstractImageConverter> manager(context->imageConverterPluginPath());
        auto converter = manager.loadAndInstantiate("PngImageConverter");
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
    auto mesh = std::make_shared<sl::Mesh>(BUNNY, context);
    mesh->load();

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
    auto object = std::make_shared<sl::Object>();
    object->setMesh(mesh);
    object->loadVisual();

    float distance = sl::pose::minimumDistanceForObjectDiameter(
        mesh->bbox().size().length(),
        scene.projectionMatrix()
    );

    object->setPose(Matrix4::translation(Vector3(0.0, 0.0, distance)));

    // Add it to the scene
    scene.addObject(object);

    // Check that we have got a valid instance ID
    CHECK(object->instanceIndex() == 1);

    object->setInstanceIndex(0xFFFF);

    // Render everything using a Phong shader
    sl::RenderPass pass;
    auto ret = pass.render(scene);

    auto buffer = &ret->rgb;

    REQUIRE(buffer);
    CHECK(buffer->imageSize() == Magnum::Vector2i(640, 480));

    Corrade::PluginManager::Manager<Magnum::Trade::AbstractImageConverter> manager(context->imageConverterPluginPath());
    auto converter = manager.loadAndInstantiate("PngImageConverter");
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

    Image2D classImage = ret->classIndex.image({PixelFormat::R16UI});
    {
        unsigned int classCount = 0;
        REQUIRE(classImage.pixelSize() == 2);

        const auto data = reinterpret_cast<uint16_t*>(classImage.data().data());

        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for(int i = 0; i < 100; ++i)
            ss << std::setw(2) << data[i] << ' ';

        INFO("First pixels: " << ss.str());

        for(int i = 0; i < image.size().product(); ++i)
        {
            if(data[i] != 0)
                classCount++;
        }

        CHECK(classCount > 10);
        CHECK(classCount < 0.5 * image.size().product());
    }

    Image2D instanceImage = ret->instanceIndex.image({PixelFormat::R16UI});
    {
        unsigned int instanceCount = 0;
        REQUIRE(instanceImage.pixelSize() == 2);

        const auto data = reinterpret_cast<uint16_t*>(instanceImage.data().data());

        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for(int i = 0; i < 100; ++i)
            ss << std::setw(4) << data[i] << ' ';

        INFO("First pixels: " << ss.str());

        for(int i = 0; i < image.size().product(); ++i)
        {
            if(data[i] == 65535)
                instanceCount++;
            else if(data[i] != 0)
            {
                std::cout << "got: " << std::hex << data[i] << "\n";
            }
        }

        CHECK(instanceCount > 10);
        CHECK(instanceCount < 0.5 * image.size().product());
    }
}

TEST_CASE("physics")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    // Load a mesh file
    auto mesh = std::make_shared<sl::Mesh>(BUNNY, context);
    mesh->load(100);

    mesh->centerBBox();
    mesh->scaleToBBoxDiagonal(0.5);

    // Create a scene
    sl::Scene scene(context, sl::ViewportSize(640, 480));

    std::vector<std::shared_ptr<sl::Object>> objects;

    for(int i = 0; i < 2; ++i)
    {
        // Instantiate the mesh to create a movable scene object
        auto object = std::make_shared<sl::Object>();
        object->setMesh(mesh);
        object->loadPhysics();

        float distance = sl::pose::minimumDistanceForObjectDiameter(
            mesh->bbox().size().length(),
            scene.projectionMatrix()
        );

        object->setPose(Matrix4::translation(Vector3(0.0, 0.0, distance)));

        // Add it to the scene
        scene.addObject(object);

        sl::pose::RandomPositionSampler posSampler{
            scene.projectionMatrix(),
            object->mesh()->bbox().size().length()
        };
        sl::pose::RandomPoseSampler sampler{posSampler};
        CHECK(scene.findNonCollidingPose(*object, sampler));

        objects.push_back(std::move(object));
    }
}

TEST_CASE("serialization")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    // Load a mesh file
    auto mesh = std::make_shared<sl::Mesh>(BUNNY, context);
    mesh->load(100);

    // Create a scene
    sl::Scene scene(context, sl::ViewportSize(640, 480));

    // Instantiate the mesh to create a movable scene object
    auto object = std::make_shared<sl::Object>();
    object->setMesh(mesh);

    float distance = sl::pose::minimumDistanceForObjectDiameter(
        mesh->bbox().size().length(),
        scene.projectionMatrix()
    );

    object->setPose(Matrix4::translation(Vector3(0.0, 0.0, distance)));

    object->setInstanceIndex(15);

    // Add it to the scene
    scene.addObject(object);

    Corrade::Utility::Configuration config;
    scene.serialize(config);

    std::ostringstream ss;
    config.save(ss);

    CAPTURE(ss.str());

    std::istringstream ssInput{ss.str()};

    Corrade::Utility::Configuration config2{ssInput};

    sl::MeshCache cache(context);

    sl::Scene nScene0(context, sl::ViewportSize(640, 480));
    nScene0.deserialize(config2, &cache);

    REQUIRE(nScene0.objects().size() == 1);

    auto nScene0Obj = nScene0.objects()[0];
    CHECK(nScene0Obj->mesh()->pretransformScale() == mesh->pretransformScale());

    CHECK(nScene0Obj->pose().translation().x() == 0);
    CHECK(nScene0Obj->pose().translation().y() == 0);
    CHECK(nScene0Obj->pose().translation().z() == Approx(distance));

    CHECK(nScene0Obj->instanceIndex() == 15);

    // Check cache functionality
    sl::Scene nScene1(context, sl::ViewportSize(640, 480));
    nScene1.deserialize(config2, &cache);

    REQUIRE(nScene1.objects().size() == 1);
    CHECK(nScene1.objects()[0]->mesh() == nScene0Obj->mesh());
}
