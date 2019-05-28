// PhysX integration: implementation utils
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_PHYSX_IMPL_H
#define STILLLEBEN_PHYSX_IMPL_H

#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Matrix4.h>

#include <PxPhysicsAPI.h>
#include <foundation/PxSimpleTypes.h>

#include <utility>

namespace sl
{

class PhysXOutputBuffer : public physx::PxOutputStream
{
public:
    PhysXOutputBuffer() {}
    PhysXOutputBuffer(const PhysXOutputBuffer&) = delete;
    PhysXOutputBuffer(PhysXOutputBuffer&& other)
    {
        m_data = std::move(other.m_data);
    }
    PhysXOutputBuffer(std::vector<uint8_t>&& data)
     : m_data{std::move(data)}
    {}

    PhysXOutputBuffer& operator=(PhysXOutputBuffer&& other)
    {
        std::swap(m_data, other.m_data);
        return *this;
    }

    physx::PxU32 write(const void* src, physx::PxU32 count) override
    {
        std::size_t off = m_data.size();
        m_data.resize(off + count);
        memcpy(m_data.data() + off, src, count);

        return count;
    }

    uint8_t* data() { return m_data.data(); }
    std::size_t size() { return m_data.size(); }
private:
    std::vector<uint8_t> m_data;
};

}

// Magnum <-> PhysX conversions
namespace Magnum { namespace Math { namespace Implementation {

template<> struct VectorConverter<3, float, physx::PxVec3>
{
    static Vector<3, Float> from(const physx::PxVec3& other)
    {
        return {other.x, other.y, other.z};
    }

    static physx::PxVec3 to(const Vector<3, Float>& other)
    {
        return {other[0], other[1], other[2]};
    }
};

template<> struct QuaternionConverter<float, physx::PxQuat>
{
    static Quaternion<Float> from(const physx::PxQuat& other)
    {
        return {{other.x, other.y, other.z}, other.w};
    }

    static physx::PxQuat to(const Quaternion<Float>& other)
    {
        return {other.vector().x(), other.vector().y(), other.vector().z(), other.scalar()};
    }
};

template<> struct RectangularMatrixConverter<4, 4, float, physx::PxMat44>
{
    static RectangularMatrix<4, 4, Float> from(const physx::PxMat44& other)
    {
        return RectangularMatrix<4, 4, Float>::from(other.front());
    }

    static physx::PxMat44 to(const RectangularMatrix<4, 4, Float>& other)
    {
        // physx::PxMat44(float[]) makes a copy, but does not take a const
        // pointer *sigh*
        return physx::PxMat44(const_cast<float*>(other.data()));
    }
};

template<> struct RectangularMatrixConverter<4, 4, float, physx::PxTransform>
{
    static RectangularMatrix<4, 4, Float> from(const physx::PxTransform& other)
    {
        return Magnum::Matrix4::from(Magnum::Quaternion{other.q}.toMatrix(), Magnum::Vector3{other.p});
    }

    static physx::PxTransform to(const RectangularMatrix<4, 4, Float>& other)
    {
        return physx::PxTransform{physx::PxMat44(other)};
    }
};

}}}

#endif
