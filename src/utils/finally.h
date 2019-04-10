// RAII finally() method
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_UTILS_FINALLY_H
#define STILLLEBEN_UTILS_FINALLY_H

#include <utility>

namespace sl
{

namespace
{
    template<class F>
    struct FinallyCaller
    {
        FinallyCaller(F&& f)
            : m_cb(f)
        {
        }

        FinallyCaller(FinallyCaller<F>&& other)
            : m_cb(std::move(other.m_cb))
        {}

        ~FinallyCaller()
        {
            m_cb();
        }
    private:
        F m_cb;
    };
}

template<class F>
[[nodiscard]] FinallyCaller<F> finally(F&& f)
{
    return {std::move(f)};
}

}

#endif
