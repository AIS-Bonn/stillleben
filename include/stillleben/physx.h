// PhysX integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_PHYSX_H
#define STILLLEBEN_PHYSX_H

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

    PhysXHolder(const PhysXHolder<T>& other)
     : m_ptr{other.m_ptr}
    {
        m_ptr->acquireReference();
    }

    PhysXHolder(PhysXHolder<T>&& other)
     : m_ptr{other.m_ptr}
    {
        other.m_ptr = nullptr;
    }

    ~PhysXHolder()
    {
        doDelete();
    }

    PhysXHolder<T>& operator=(const PhysXHolder<T>& other)
    {
        m_ptr = other.m_ptr;
        m_ptr->acquireReference();
        return *this;
    }

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

template<class T>
class PhysXUnique
{
public:
    PhysXUnique() {}

    explicit PhysXUnique(T* ptr)
     : m_ptr{ptr}
    {}

    PhysXUnique(const PhysXUnique<T>&) = delete;

    PhysXUnique(PhysXUnique<T>&& other)
     : m_ptr{other.m_ptr}
    {
        other.m_ptr = nullptr;
    }

    ~PhysXUnique()
    {
        doDelete();
    }

    PhysXUnique<T>& operator=(const PhysXUnique<T>&) = delete;

    PhysXUnique<T>& operator=(PhysXUnique<T>&& other)
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
