// Functions for working with 6D poses
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/pose.h>

namespace sl
{
namespace pose
{

namespace detail
{
    Magnum::Matrix3 crossMatrix(const Magnum::Vector3& v)
    {
        // col-major!
        return {
            {0.0, v.z(), -v.y()},
            {-v.z(), 0.0, v.x()},
            {v.y(), -v.x(), 0.0}
        };
    }
}

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

Magnum::Matrix3 rotationCorrectionForTranslation(const Magnum::Vector3& pos)
{
    using namespace Magnum;

    Vector3 a = pos.normalized();

    // We want to rotate a onto the z axis
    const Vector3 z = Vector3::zAxis();

    // axis of rotation
    Vector3 v = Math::cross(a, z);

    float s = v.length();
    float c = Math::dot(a, z);

    if(std::abs(s) < 1e-5)
        return {}; // identity

    auto vx = detail::crossMatrix(v);

    auto R = Matrix3{} + vx + (1.0f-c)/(s*s) * (vx*vx);

    return R.transposed();
}

}
}
