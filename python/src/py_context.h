// sl::Context binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_CONTEXT_H
#define SL_PY_CONTEXT_H

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

namespace sl
{
class Context;

namespace python
{

class ContextSharedPtrBase
{
public:
    ContextSharedPtrBase();
    ~ContextSharedPtrBase();

    void check();
private:
    std::shared_ptr<Context> m_context;
};

template<class T>
class ContextSharedPtr : public std::shared_ptr<T>, private ContextSharedPtrBase
{
public:
    ContextSharedPtr()
    {
        check();
    }

    explicit ContextSharedPtr(T* ptr)
     : std::shared_ptr<T>(ptr)
    {
        check();
    }

    ContextSharedPtr(const std::shared_ptr<T>& ptr)
     : std::shared_ptr<T>(ptr)
    {
        check();
    }

    ~ContextSharedPtr()
    {
        // Delete our pointee before we release the context.
        this->reset();
    }
};

namespace Context
{
    void init(py::module& m);
    const std::shared_ptr<sl::Context>& instance();
    bool cudaEnabled();
    unsigned int cudaDevice();
}

}
}

PYBIND11_DECLARE_HOLDER_TYPE(T, sl::python::ContextSharedPtr<T>);

#endif
