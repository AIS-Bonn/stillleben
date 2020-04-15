// sl::Context binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_context.h"

#include <torch/extension.h>

#include <stillleben/context.h>

namespace
{
    std::shared_ptr<sl::Context> g_context;
    bool g_cudaEnabled = false;
    unsigned int g_cudaIndex = 0;
    std::string g_installPrefix;

    void init()
    {
        if(g_context)
        {
            if(g_cudaEnabled)
            {
                Corrade::Utility::Warning{}
                    << "stillleben context was already created with different CUDA settings";
            }
            return;
        }

        g_context = sl::Context::Create(g_installPrefix);
        if(!g_context)
            throw std::runtime_error("Could not create stillleben context");
    }

    static void initCUDA(unsigned int cudaIndex, bool useCUDA=true)
    {
        if(g_context)
        {
            if(g_cudaEnabled != useCUDA || g_cudaIndex != cudaIndex)
            {
                Corrade::Utility::Warning{}
                    << "stillleben context was already created with different CUDA settings";
            }
            return;
        }

        g_context = sl::Context::CreateCUDA(cudaIndex, g_installPrefix);
        if(!g_context)
            throw std::runtime_error("Could not create stillleben context");

        g_cudaIndex = cudaIndex;
        g_cudaEnabled = useCUDA;
    }

    static void setInstallPrefix(const std::string& path)
    {
        g_installPrefix = path;
    }
}

namespace sl
{
namespace python
{

// shared pointer with reference to the context
ContextSharedPtrBase::ContextSharedPtrBase() = default;
ContextSharedPtrBase::~ContextSharedPtrBase() = default;

void ContextSharedPtrBase::check()
{
    if(!g_context)
        throw std::logic_error("Call sl::init() first");

    m_context = g_context;
}

namespace Context
{

void init(py::module& m)
{
    m.def("init", &::init, "Init without CUDA support");
    m.def("init_cuda", &initCUDA, R"EOS(
        Init with CUDA support.

        Args:
            device_index (int): Index of CUDA device to use for rendering
            use_cuda (bool): If false, return results on CPU
    )EOS", py::arg("device_index") = 0, py::arg("use_cuda")=true);

    m.def("_set_install_prefix", &setInstallPrefix, "set Magnum install prefix");

    // We need to release our context pointer when the python module is
    // unloaded. Otherwise, we release it very late (basically, when the
    // atexit handlers are called. MESA also has atexit handlers, and if they
    // get called before our cleanup code, bad things happen.
    auto cleanup_callback = []() {
        g_context.reset();
    };

    m.add_object("_cleanup", py::capsule(cleanup_callback));
}

bool cudaEnabled()
{
    return g_cudaEnabled;
}

unsigned int cudaDevice()
{
    return g_cudaIndex;
}

const std::shared_ptr<sl::Context>& instance()
{
    return g_context;
}

}
}
}
