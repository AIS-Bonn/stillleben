// Animate a 6D pose
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_ANIMATOR_H
#define STILLLEBEN_ANIMATOR_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Quaternion.h>

#include <Magnum/Animation/Track.h>

#include <vector>

namespace sl
{

class Animator
{
public:
    explicit Animator(const std::vector<Magnum::Matrix4>& poses, unsigned int ticks);

    [[nodiscard]] unsigned int totalTicks() const
    { return m_totalTicks; }

    [[nodiscard]] unsigned int currentTick() const
    { return m_index; }

    Magnum::Matrix4 operator()();
private:
    Magnum::Animation::Track<unsigned int, Magnum::Vector3> m_positionTrack;
    Magnum::Animation::Track<unsigned int, Magnum::Quaternion> m_orientationTrack;
    unsigned int m_index = 0;
    unsigned int m_totalTicks = 0;
};

}

#endif
