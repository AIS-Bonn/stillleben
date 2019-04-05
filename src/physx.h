// PhysX integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_PHYSX_H
#define STILLLEBEN_PHYSX_H

#include <PxPhysicsAPI.h>
#include <foundation/PxSimpleTypes.h>

#include <utility>

namespace sl
{

template<class T>
class PhysXHolder
{
public:
    PhysXHolder() {}

    explicit PhysXHolder(T* ptr)
     : m_ptr{ptr}
    {}

    PhysXHolder(const PhysXHolder<T>&) = delete;

    PhysXHolder(PhysXHolder<T>&& other)
     : m_ptr{other.m_ptr}
    {
        other.m_ptr = nullptr;
    }

    ~PhysXHolder()
    {
        doDelete();
    }

    PhysXHolder<T>& operator=(const PhysXHolder<T>&) = delete;

    PhysXHolder<T>& operator=(PhysXHolder<T>&& other)
    {
        std::swap(m_ptr, other.m_ptr);
        return *this;
    }

    explicit operator bool() const { return m_ptr; }

    T* get() { return m_ptr; }
    const T* get() const { return m_ptr; }

    T* operator->() { return m_ptr; }
    const T* operator->() const { return m_ptr; }

    T& operator*() { return *m_ptr; }
    const T& operator*() const { return *m_ptr; }

    void reset(T* ptr = nullptr)
    {
        doDelete();
        m_ptr = ptr;
    }

    T* release()
    {
        T* const out = m_ptr;
        m_ptr = nullptr;
        return out;
    }
private:
    void doDelete()
    {
        if(m_ptr)
        {
            m_ptr->release();
            m_ptr = 0;
        }
    }

    T* m_ptr = nullptr;
};

}

#endif
