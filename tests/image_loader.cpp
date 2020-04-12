// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/image_loader.h>

#include <iostream>
#include <iomanip>

#include "doctest.h"

using namespace Corrade;
using namespace Magnum;

constexpr const char* BUNNY_TEXTURES = PATH_TO_SOURCES "/tests/stanford_bunny/textures";

TEST_CASE("load image")
{
    // Create our stillleben Context
    auto context = sl::Context::Create();
    REQUIRE(context);

    sl::ImageLoader loader{BUNNY_TEXTURES, context};

    Magnum::GL::RectangleTexture texture = loader.nextRectangleTexture();
    CHECK(texture.imageSize().dot() > 0);
}
