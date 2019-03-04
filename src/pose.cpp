// Functions for working with 6D poses
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/pose.h>

namespace sl
{
namespace pose
{

float minimumDistanceForObjectDiameter(float diameter, const Magnum::Matrix4& P)
{
    // for perspective projection:
    // P[0][0] = 1.0 / std::tan(alpha)
    // NOTE: alpha is the half horizontal view angle.

    return std::max(
        P[0][0] * diameter / 2.0,
        P[1][1] * diameter / 2.0
    );
}

}
}
