// Animate a 6D pose
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/animator.h>

#include <Corrade/Containers/Array.h>

#include <Magnum/Math/Quaternion.h>

using namespace Magnum;

namespace sl
{

Animator::Animator(const std::vector<Matrix4>& poses, unsigned int ticks)
 : m_totalTicks{ticks}
{
    if(poses.size() < 2)
        throw std::invalid_argument("Need at least two poses to animate...");

    Containers::Array<std::pair<unsigned int, Vector3>> positions(poses.size());
    Containers::Array<std::pair<unsigned int, Quaternion>> orientations(poses.size());

    std::size_t idx = 0;
    unsigned int time = 0;

    for(auto& pose : poses)
    {
        time = idx * ticks / (poses.size()-1);

        positions[idx] = {time, pose.translation()};
        orientations[idx] = {time, Quaternion::fromMatrix(pose.rotationScaling())};

        idx++;
    }

    m_positionTrack = Animation::Track<unsigned int, Vector3>{std::move(positions), Animation::Interpolation::Linear};
    m_orientationTrack = Animation::Track<unsigned int, Quaternion>{std::move(orientations), Animation::Interpolation::Linear};
}

Matrix4 Animator::operator()()
{
    auto position = m_positionTrack.at(m_index);
    auto orientation = m_orientationTrack.at(m_index);
    m_index++;

    return Matrix4::from(orientation.toMatrix(), position);
}

}
