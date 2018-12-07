// Basic high-level unit tests
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "catch.hpp"

#include <stillleben/context.h>

TEST_CASE("basic")
{
    auto context = sl::Context::Create();
    REQUIRE(context);


}
