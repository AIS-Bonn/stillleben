// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/image_loader.h>

#include <iostream>
#include <iomanip>

using namespace Corrade;
using namespace Magnum;

#include "catch.hpp"

constexpr const char* BUNNY_TEXTURES = PATH_TO_SOURCES "/tests/stanford_bunny/textures";

TEST_CASE("load image", "[image_loader]")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    sl::ImageLoader loader{BUNNY_TEXTURES, context};

    Magnum::GL::RectangleTexture texture = loader.next();
    CHECK(texture.imageSize().dot() > 0);
}
