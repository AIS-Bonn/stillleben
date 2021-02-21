// Basic image saver test
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/image_saver.h>

#include <Magnum/Image.h>
#include <Magnum/PixelFormat.h>

#include <iostream>
#include <iomanip>

#include "doctest.h"

using namespace Corrade;
using namespace Magnum;

TEST_CASE("save jpeg")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    Containers::Array<char> data{256*256};
    Magnum::Image2D image(PixelFormat::R8Unorm, {256,256}, std::move(data));

    bool deleted = false;
    {
        sl::ImageSaver saver(context);

        sl::ImageSaver::Job job;
        job.image = image;
        job.path = "/tmp/test.jpg";
        job.deleter = [&](){ deleted = true; };
        saver.save(std::move(job));
    }

    CHECK(deleted);
}

TEST_CASE("save png")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    Containers::Array<char> data{256*256};
    Magnum::Image2D image(PixelFormat::R8Unorm, {256,256}, std::move(data));

    bool deleted = false;
    {
        sl::ImageSaver saver(context);

        sl::ImageSaver::Job job;
        job.image = image;
        job.path = "/tmp/test.png";
        job.deleter = [&](){ deleted = true; };

        saver.save(std::move(job));
    }

    CHECK(deleted);
}
