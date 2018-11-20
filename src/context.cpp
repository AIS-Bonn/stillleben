// stillleben context
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>

#include <algorithm>

#include <Magnum/GL/Context.h>
#include <Magnum/Platform/GLContext.h>

#if HAVE_EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>

template<class T>
T getExtension(const char* name)
{
    return reinterpret_cast<T>(eglGetProcAddress(name));
}

#endif // HAVE_EGL

#include <Magnum/Platform/WindowlessGlxApplication.h>

using namespace Magnum;

namespace sl
{

class Context::Private
{
public:
    Private()
    {
        int argc = 3;
        std::vector<const char*> argv{
            "dummy",
            "--magnum-log", "quiet"
        };
        gl_context.reset(new Platform::GLContext{NoCreate, argc, argv.data()});
    }

    void* egl_display = nullptr;
    void* egl_context = nullptr;

    std::unique_ptr<Platform::WindowlessGlxContext> glx_context;

    std::unique_ptr<Platform::GLContext> gl_context;
};

Context::Context()
 : m_d{std::make_unique<Private>()}
{
}

Context::Ptr Context::Create()
{
    Context::Ptr context{new Context};

#if HAVE_EGL
    // Load required extensions
    auto eglQueryDevicesEXT = getExtension<PFNEGLQUERYDEVICESEXTPROC>("eglQueryDevicesEXT");
    if(!eglQueryDevicesEXT)
    {
        Error() << "Could not find required eglQueryDevicesEXT";
        return {};
    }

    auto eglQueryDeviceAttribEXT = getExtension<PFNEGLQUERYDEVICEATTRIBEXTPROC>("eglQueryDeviceAttribEXT");
    if(!eglQueryDeviceAttribEXT)
    {
        Error() << "Could not find required eglQueryDeviceAttribEXT";
        return {};
    }

    auto eglGetPlatformDisplayEXT = getExtension<PFNEGLGETPLATFORMDISPLAYEXTPROC>("eglGetPlatformDisplayEXT");
    if(!eglGetPlatformDisplayEXT)
    {
        Error() << "Could not find required eglGetPlatformDisplayEXT";
        return {};
    }

    // Enumerate devices
    std::vector<EGLDeviceEXT> devices(64);
    EGLint num_devices;

    if(!eglQueryDevicesEXT(devices.size(), devices.data(), &num_devices))
    {
        Error() << "Could not enumerate EGL devices";
        return {};
    }

    if(num_devices < 1)
    {
        Error() << "Could not find an EGL device";
        return {};
    }

    context->m_d->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices.front(), nullptr);
    if(!context->m_d->display)
    {
        Error() << "Could not get display for CUDA device";
        return {};
    }

    EGLint major, minor;
    if(!eglInitialize(context->m_d->display, &major, &minor))
    {
        Error() << "Could not initialize EGL display";
        return {};
    }

    Debug{} << "Display initialized for EGL " << major << "." << minor;

    if(!eglBindAPI(EGL_OPENGL_API))
    {
        Error() << "Could not bind OpenGL API";
        return {};
    }

    EGLint numberConfigs;
    EGLConfig eglConfig;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    if(!eglChooseConfig(context->m_d->display, configAttribs, &eglConfig, 1, &numberConfigs))
    {
        Error() << "Could not call eglChooseConfig";
        return {};
    }

    if(!numberConfigs)
    {
        Error() << "Could not find any matching EGL config :-(";
        return {};
    }

    context->m_d->egl_context = eglCreateContext(context->m_d->display, eglConfig, EGL_NO_CONTEXT, nullptr);
    if(!context->m_d->egl_context)
    {
        Error() << "Could not create EGL context";
        return {};
    }

    if(!context->makeCurrent())
    {
        Error() << "Cannot make context current";
        return {};
    }
#else
    context->m_d->glx_context = std::make_unique<Magnum::Platform::WindowlessGlxContext>(
        Magnum::Platform::WindowlessGlxContext::Configuration{}
    );

    if(!context->m_d->glx_context->isCreated() || !context->m_d->glx_context->makeCurrent())
    {
        Error() << "Could not create windowless glx context";
        return {};
    }
#endif

    if(!context->m_d->gl_context->tryCreate())
    {
        Error() << "Could not create Platform::GLContext";
        return {};
    }

    return context;
}

Context::Ptr Context::CreateCUDA(unsigned int device)
{
#if HAVE_EGL
    auto context = std::make_shared<Context>();

    // Load required extensions
    auto eglQueryDevicesEXT = getExtension<PFNEGLQUERYDEVICESEXTPROC>("eglQueryDevicesEXT");
    if(!eglQueryDevicesEXT)
    {
        Error() << "Could not find required eglQueryDevicesEXT";
        return {};
    }

    auto eglQueryDeviceAttribEXT = getExtension<PFNEGLQUERYDEVICEATTRIBEXTPROC>("eglQueryDeviceAttribEXT");
    if(!eglQueryDeviceAttribEXT)
    {
        Error() << "Could not find required eglQueryDeviceAttribEXT";
        return {};
    }

    auto eglGetPlatformDisplayEXT = getExtension<PFNEGLGETPLATFORMDISPLAYEXTPROC>("eglGetPlatformDisplayEXT");
    if(!eglGetPlatformDisplayEXT)
    {
        Error() << "Could not find required eglGetPlatformDisplayEXT";
        return {};
    }

    // Enumerate devices
    std::vector<EGLDeviceEXT> devices(64);
    EGLint num_devices;

    if(!eglQueryDevicesEXT(devices.size(), devices.data(), &num_devices))
    {
        Error() << "Could not enumerate EGL devices";
        return {};
    }

    if(num_devices < 1)
    {
        Error() << "Could not find an EGL device";
        return {};
    }

    devices.resize(num_devices);

    auto it = std::find_if(devices.begin(), devices.end(), [&](EGLDeviceEXT& dev){
        EGLAttrib devCudaIndex;
        if(!eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, &devCudaIndex))
        {
            return false;
        }

        return devCudaIndex == device;
    });

    if(it == devices.end())
    {
        Error() << "Could not find matching CUDA device with index" << cudaIndex;
        return {};
    }

    context->m_d->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, *it, nullptr);
    if(!context->m_d->display)
    {
        Error() << "Could not get display for CUDA device";
        return {};
    }

    EGLint major, minor;
    if(!eglInitialize(context->m_d->display, &major, &minor))
    {
        Error() << "Could not initialize EGL display";
        return {};
    }

    Debug{} << "Display initialized for EGL " << major << "." << minor;

    if(!eglBindAPI(EGL_OPENGL_API))
    {
        Error() << "Could not bind OpenGL API";
        return {};
    }

    EGLint numberConfigs;
    EGLConfig eglConfig;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    if(!eglChooseConfig(context->m_d->display, configAttribs, &eglConfig, 1, &numberConfigs))
    {
        Error() << "Could not call eglChooseConfig";
        return {};
    }

    if(!numberConfigs)
    {
        Error() << "Could not find any matching EGL config :-(";
        return {};
    }

    context->m_d->egl_context = eglCreateContext(context->m_d->display, eglConfig, EGL_NO_CONTEXT, nullptr);
    if(!context->m_d->egl_context)
    {
        Error() << "Could not create EGL context";
        return {};
    }

    if(!context->makeCurrent())
    {
        Error() << "Cannot make context current";
        return {};
    }

    return context;
#else
    Error() << "Called Context::createCUDA() without EGL support";
    return {};
#endif
}

bool Context::makeCurrent()
{
#if HAVE_EGL
    if(!eglMakeCurrent(m_d->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, m_d->egl_context))
    {
        Error() << "Cannot make context current";
        return false;
    }
#else
    return m_d->glx_context->makeCurrent();
#endif
}

}
