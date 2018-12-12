// stillleben context
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>

#include <algorithm>

#include <Corrade/PluginManager/PluginManager.h>

#include <Magnum/GL/Context.h>
#include <Magnum/Platform/GLContext.h>
#include <Magnum/Trade/AbstractImporter.h>

#include <cstring>

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

#if !defined(HAVE_EGL) || !HAVE_EGL
    std::unique_ptr<Platform::WindowlessGlxContext> glx_context;
#endif

    std::unique_ptr<Platform::GLContext> gl_context;

    std::shared_ptr<Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter>> importerManager{
        std::make_shared<PluginManager::Manager<Trade::AbstractImporter>>()
    };
};

Context::Context()
 : m_d{std::make_unique<Private>()}
{
}

Context::Ptr Context::Create()
{
    Context::Ptr context{new Context};

#if HAVE_EGL
    const char* extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if(!extensions)
    {
        Error() << "Could not query EGL extensions";
        return {};
    }

    Debug() << "Supported EGL extensions:" << extensions;

    // Load required extensions
    auto eglQueryDevicesEXT = getExtension<PFNEGLQUERYDEVICESEXTPROC>("eglQueryDevicesEXT");
    if(!eglQueryDevicesEXT)
    {
        Error() << "Could not find required eglQueryDevicesEXT";
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

    EGLDisplay display = nullptr;

    if(num_devices > 0)
    {
        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices.front(), nullptr);
    }
    else
    {
        Debug() << "Could not enumerate EGL devices, trying MESA targets with EGL_DEFAULT_DISPLAY...";

        if(strstr(extensions, "EGL_MESA_platform_surfaceless"))
        {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
            if(!display)
                Debug() << "surfaceless failed";
        }
        if(!display && strstr(extensions, "EGL_EXT_platform_x11"))
        {
            Debug() << "Trying X11";
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_X11_EXT, EGL_DEFAULT_DISPLAY, nullptr);
            if(!display)
                Debug() << "X11 failed";
        }
        if(!display && strstr(extensions, "EGL_MESA_platform_gbm"))
        {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_MESA, EGL_DEFAULT_DISPLAY, nullptr);
            if(!display)
                Debug() << "gpm failed";
        }
    }

    if(!display)
    {
        Error() << "Could not create EGL display";
        return {};
    }

    context->m_d->egl_display = display;

    EGLint major, minor;
    if(!eglInitialize(context->m_d->egl_display, &major, &minor))
    {
        Error() << "Could not initialize EGL display";
        return {};
    }

    Debug{} << "Display initialized for EGL " << major << "." << minor;

    const char* vendor = eglQueryString(display, EGL_VENDOR);
    if(vendor)
        Debug{} << "EGL vendor:" << vendor;

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
    if(!eglChooseConfig(context->m_d->egl_display, configAttribs, &eglConfig, 1, &numberConfigs))
    {
        Error() << "Could not call eglChooseConfig";
        return {};
    }

    if(!numberConfigs)
    {
        Error() << "Could not find any matching EGL config :-(";
        return {};
    }

    EGLint contextAttribs[]{
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,

        EGL_NONE
    };

    context->m_d->egl_context = eglCreateContext(context->m_d->egl_display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    if(!context->m_d->egl_context)
    {
        fprintf(stderr, "Could not create EGL context: 0x%X\n", eglGetError());
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

    {
        GLint max_samples;
        glGetIntegerv(GL_MAX_INTEGER_SAMPLES, &max_samples);
        Debug{} << "GL_MAX_INTEGER_SAMPLES:" << max_samples;
    }

    return context;
}

Context::Ptr Context::CreateCUDA(unsigned int device)
{
#if HAVE_EGL
    std::shared_ptr<Context> context(new Context);

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
        Error() << "Could not find matching CUDA device with index" << device;
        return {};
    }

    context->m_d->egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, *it, nullptr);
    if(!context->m_d->egl_display)
    {
        Error() << "Could not get display for CUDA device";
        return {};
    }

    EGLint major, minor;
    if(!eglInitialize(context->m_d->egl_display, &major, &minor))
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
    if(!eglChooseConfig(context->m_d->egl_display, configAttribs, &eglConfig, 1, &numberConfigs))
    {
        Error() << "Could not call eglChooseConfig";
        return {};
    }

    if(!numberConfigs)
    {
        Error() << "Could not find any matching EGL config :-(";
        return {};
    }

    context->m_d->egl_context = eglCreateContext(context->m_d->egl_display, eglConfig, EGL_NO_CONTEXT, nullptr);
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

    return true;
#else
    return m_d->glx_context->makeCurrent();
#endif
}

std::shared_ptr<Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter>> Context::importerPluginManager()
{
    return m_d->importerManager;
}

}
