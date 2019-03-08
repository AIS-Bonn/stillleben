// Functions for working with 6D poses
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_POSE_H
#define STILLLEBEN_POSE_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Quaternion.h>

#include <random>

namespace sl
{

namespace pose
{

namespace detail
{
    Magnum::Matrix3 crossMatrix(const Magnum::Vector3& v);
}

template<class Generator>
Magnum::Quaternion randomQuaternion(Generator& g)
{
    std::normal_distribution<float> normalDist;
    Magnum::Quaternion q{
        Magnum::Vector3{normalDist(g), normalDist(g), normalDist(g)},
        normalDist(g)
    };

    return q.normalized();
}

template<class Generator>
Magnum::Matrix3 randomRotation(Generator& g)
{
    return randomQuaternion(g).toMatrix();
}

float minimumDistanceForObjectDiameter(float diameter, const Magnum::Matrix4& P);

/**
 * This counters the apparent rotation of an object that is just translated
 * in the camera FOV.
 **/
Magnum::Matrix3 rotationCorrectionForTranslation(const Magnum::Vector3& pos);


// Sampler

constexpr float DEFAULT_MIN_SIZE_FACTOR = 0.2f;

class RandomPositionSampler
{
public:
    RandomPositionSampler(const Magnum::Matrix4& P, float diameter)
     : m_P{P}
     , m_diameter{diameter}
     , m_fullyVisible{minimumDistanceForObjectDiameter(diameter, P)}
    {}

    void setMinSizeFactor(float f)
    { m_minSizeFactor = f; }

    template<class Generator>
    Magnum::Vector3 operator()(Generator& g)
    {
        // Step 1: Produce a suitable z coordinate
        std::uniform_real_distribution<float> zDist(
            1.2 * m_fullyVisible,
            (1.0 / m_minSizeFactor) * m_fullyVisible
        );

        const float z = zDist(g);

        // Step 2: Choose x,y
        // P[0][0] = 1.0 / std::tan(alpha)

        const float x_range = 0.8 * z / m_P[0][0];
        const float y_range = 0.8 * z / m_P[1][1];

        std::uniform_real_distribution<float> xDist(-x_range, x_range);
        std::uniform_real_distribution<float> yDist(-y_range, y_range);

        return {
            xDist(g),
            yDist(g),
            z
        };
    }
private:
    Magnum::Matrix4 m_P;
    float m_diameter;
    float m_fullyVisible;
    float m_minSizeFactor = DEFAULT_MIN_SIZE_FACTOR;
};

class RandomPoseSampler
{
public:
    RandomPoseSampler(RandomPositionSampler& positionSampler)
     : m_positionSampler{positionSampler}
    {}

    template<class Generator>
    Magnum::Matrix4 operator()(Generator& g)
    {
        return Magnum::Matrix4::from(
            randomRotation(g),
            m_positionSampler(g)
        );
    }
private:
    RandomPositionSampler& m_positionSampler;
};

inline Magnum::Vector3 perpendicularVector(const Magnum::Vector3& x)
{
    using Magnum::Math::cross;

    if(std::abs(x.x()) > 0.8)
        return cross(x, Magnum::Vector3::yAxis()).normalized();
    else
        return cross(x, Magnum::Vector3::xAxis()).normalized();
}

class ViewPointPoseSampler
{
public:
    ViewPointPoseSampler(RandomPositionSampler& positionSampler)
     : m_positionSampler{positionSampler}
    {}

    void setViewPoint(const Magnum::Vector3& viewPoint)
    {
        m_viewPoint = viewPoint;
    }

    template<class Generator>
    Magnum::Matrix4 operator()(Generator& g)
    {
        Magnum::Vector3 pos = m_positionSampler(g);

        // Make the local object coordinate system point towards the
        // camera (origin) with its X axis
        Magnum::Matrix3 xFacingCamera;
        xFacingCamera[0] = -pos.normalized();
        xFacingCamera[1] = perpendicularVector(xFacingCamera[0]);
        xFacingCamera[2] = Magnum::Math::cross(
            xFacingCamera[0], xFacingCamera[1]
        );

        // Rotate around the X axis with a uniformly random angle
        std::uniform_real_distribution<float> rotDist(-M_PI, M_PI);
        Magnum::Matrix3 xRot = Magnum::Matrix4::rotation(
            Magnum::Rad{rotDist(g)}, Magnum::Vector3::xAxis()
        ).rotationScaling();

        // Make sure that our view point maps to the X axis
        Magnum::Matrix3 viewPointX;
        viewPointX.setRow(0, m_viewPoint);
        viewPointX.setRow(1, perpendicularVector(viewPointX.row(0)));
        viewPointX.setRow(2, Magnum::Math::cross(
            viewPointX.row(0), viewPointX.row(1)
        ));

        if(false)
        {
            Corrade::Utility::Debug{} << "viewPoint";
            Corrade::Utility::Debug{} << m_viewPoint;

            Corrade::Utility::Debug{} << "xFacingCamera";
            Corrade::Utility::Debug{} << xFacingCamera;
            Corrade::Utility::Debug{} << "xRot";
            Corrade::Utility::Debug{} << xRot;
            Corrade::Utility::Debug{} << "viewPointX";
            Corrade::Utility::Debug{} << viewPointX;
        }

        return Magnum::Matrix4::from(
            xFacingCamera * xRot * viewPointX,
            pos
        );
    }

private:
    RandomPositionSampler& m_positionSampler;
    Magnum::Vector3 m_viewPoint{1.0f, 0.0f, 0.0f};
};

class ViewCorrectedPoseSampler
{
public:
    ViewCorrectedPoseSampler(RandomPositionSampler& positionSampler, const Magnum::Matrix3& orientation)
     : m_positionSampler{positionSampler}
     , m_orientation{orientation}
    {}

    template<class Generator>
    Magnum::Matrix4 operator()(Generator& g)
    {
        Magnum::Vector3 pos = m_positionSampler(g);

        Magnum::Matrix3 correction = rotationCorrectionForTranslation(pos);

        return Magnum::Matrix4::from(
            correction * m_orientation,
            pos
        );
    }

private:
    RandomPositionSampler& m_positionSampler;
    Magnum::Matrix3 m_orientation;
};

}
}

#endif
