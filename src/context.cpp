// stillleben context
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>

#include <algorithm>

#include <Corrade/PluginManager/PluginManager.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Platform/GLContext.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>

#include <Magnum/DebugTools/ResourceManager.h>

#include <cstring>

#include <mutex>

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

using ImporterManager = PluginManager::Manager<Trade::AbstractImporter>;

class Context::Private
{
public:
    Private(const std::string& installPrefix = {})
    {
        int argc = 3;
        std::vector<const char*> argv{
            "dummy",
            "--magnum-log", "quiet"
        };
        gl_context.reset(new Platform::GLContext{NoCreate, argc, argv.data()});

        if(!installPrefix.empty())
            importerManager = std::make_unique<ImporterManager>(installPrefix + "/lib/magnum/importers");
        else
            importerManager = std::make_unique<ImporterManager>();

        auto loadState = importerManager->load("AssimpImporter");
        if(loadState != Corrade::PluginManager::LoadState::Loaded)
        {
            throw std::runtime_error("Could not load AssimpImporter plugin");
        }
    }

    void* egl_display = nullptr;
    void* egl_context = nullptr;

#if !defined(HAVE_EGL) || !HAVE_EGL
    std::unique_ptr<Platform::WindowlessGlxContext> glx_context;
#endif

    std::unique_ptr<Platform::GLContext> gl_context;

    std::unique_ptr<ImporterManager> importerManager;
    std::mutex importerManagerMutex;

    DebugTools::ResourceManager resourceManager;
};

Context::Context(const std::string& installPrefix)
 : m_d{std::make_unique<Private>(installPrefix)}
{
}

Context::Ptr Context::Create(const std::string& installPrefix)
{
    Context::Ptr context{new Context(installPrefix)};

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
        Debug{} << "Found EGL device, trying to create display...";
        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices.front(), nullptr);
    }

    if(!display)
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

Context::Ptr Context::CreateCUDA(unsigned int device, const std::string& installPrefix)
{
#if HAVE_EGL
    std::shared_ptr<Context> context(new Context(installPrefix));

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

    if(!context->m_d->gl_context->tryCreate())
    {
        Error() << "Could not create Platform::GLContext";
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

Corrade::Containers::Pointer<Context::Importer> Context::instantiateImporter()
{
    std::unique_lock<std::mutex> lock(m_d->importerManagerMutex);
    return m_d->importerManager->instantiate("AssimpImporter");
}

Magnum::GL::RectangleTexture Context::loadTexture(const std::string& path)
{
    std::unique_ptr<Trade::AbstractImporter> importer{
        m_d->importerManager->loadAndInstantiate("AnyImageImporter")
    };

    if(!importer)
        throw std::logic_error("Could not load AnyImageImporter plugin");

    if(!importer->openFile(path))
        throw std::runtime_error("Could not open image file");

    auto image = importer->image2D(0);
    if(!image)
        throw std::runtime_error("Could not load image");

    GL::TextureFormat format;
    if(image->format() == PixelFormat::RGB8Unorm)
        format = GL::TextureFormat::RGB8;
    else if(image->format() == PixelFormat::RGBA8Unorm)
        format = GL::TextureFormat::RGBA8;
    else
        throw std::runtime_error("Unsupported texture format");

    GL::RectangleTexture texture;
    texture.setStorage(format, image->size());
    texture.setSubImage({}, *image);

    return texture;
}

}
