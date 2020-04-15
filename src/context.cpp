// stillleben context
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/context.h>
#include <stillleben/physx.h>

#include <algorithm>

#include <Corrade/PluginManager/PluginManager.h>
#include <Corrade/Utility/FormatStl.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/GLContext.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Trade/ImageData.h>

#include <Magnum/DebugTools/ResourceManager.h>

#include <cstring>
#include <sstream>
#include <mutex>
#include <iostream>

#include <EGL/egl.h>
#include <EGL/eglext.h>

template<class T>
T getExtension(const char* name)
{
    return reinterpret_cast<T>(eglGetProcAddress(name));
}

#include "physx_impl.h"

using namespace Magnum;

namespace sl
{

namespace
{
    bool envFlag(const char* name)
    {
        const char* env = getenv(name);
        if(!env)
            return false;

        return strcmp(env, "1") == 0;
    }

    std::string eglErrorString(EGLint error)
    {
        switch(error)
        {
            case EGL_SUCCESS: return "No error";
            case EGL_NOT_INITIALIZED: return "EGL not initialized or failed to initialize";
            case EGL_BAD_ACCESS: return "Resource inaccessible";
            case EGL_BAD_ALLOC: return "Cannot allocate resources";
            case EGL_BAD_ATTRIBUTE: return "Unrecognized attribute or attribute value";
            case EGL_BAD_CONTEXT: return "Invalid EGL context";
            case EGL_BAD_CONFIG: return "Invalid EGL frame buffer configuration";
            case EGL_BAD_CURRENT_SURFACE: return "Current surface is no longer valid";
            case EGL_BAD_DISPLAY: return "Invalid EGL display";
            case EGL_BAD_SURFACE: return "Invalid surface";
            case EGL_BAD_MATCH: return "Inconsistent arguments";
            case EGL_BAD_PARAMETER: return "Invalid argument";
            case EGL_BAD_NATIVE_PIXMAP: return "Invalid native pixmap";
            case EGL_BAD_NATIVE_WINDOW: return "Invalid native window";
            case EGL_CONTEXT_LOST: return "Context lost";
        }

        return Corrade::Utility::formatString("Unknown error 0x%04X", error);
    }

    struct DisplayConfig
    {
        EGLDisplay display{};
        bool useX11 = false;
    };

    DisplayConfig getEglDisplay()
    {
        EGLDisplay display = nullptr;

        const char* extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
        if(!extensions)
        {
            Error() << "Could not query EGL extensions";
            return {};
        }

        auto eglGetPlatformDisplayEXT = getExtension<PFNEGLGETPLATFORMDISPLAYEXTPROC>("eglGetPlatformDisplayEXT");
        if(!eglGetPlatformDisplayEXT)
        {
            Error() << "Could not find required eglGetPlatformDisplayEXT";
            return {};
        }

        // Try X11 first, if available. The user will likely want to use
        // the same device as X11 is running on...
        if(getenv("DISPLAY") && strstr(extensions, "EGL_EXT_platform_x11"))
        {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_X11_EXT, EGL_DEFAULT_DISPLAY, nullptr);
            if(display)
                return {display, true};
            else
                Debug() << "X11 failed";
        }

        // Load required extensions for enumeration
        auto eglQueryDevicesEXT = getExtension<PFNEGLQUERYDEVICESEXTPROC>("eglQueryDevicesEXT");
        if(!eglQueryDevicesEXT)
        {
            Error() << "Could not find required eglQueryDevicesEXT";
            return {};
        }

        auto eglQueryDeviceStringEXT = getExtension<PFNEGLQUERYDEVICESTRINGEXTPROC>("eglQueryDeviceStringEXT");

        // Enumerate devices
        std::vector<EGLDeviceEXT> devices(64);
        EGLint num_devices;

        if(!eglQueryDevicesEXT(devices.size(), devices.data(), &num_devices))
        {
            Error() << "Could not enumerate EGL devices";
            return {};
        }

        if(num_devices > 0)
        {
            Debug{} << "Found EGL device(s) (count:" << num_devices << "), trying to create display...";

            if(eglQueryDeviceStringEXT)
            {
                for(EGLint i = 0; i < num_devices; ++i)
                {
                    const char* extension = eglQueryDeviceStringEXT(devices[i], EGL_EXTENSIONS);
                    if(strcmp(extension, "EGL_EXT_device_drm") == 0)
                    {
                        const char* device = eglQueryDeviceStringEXT(devices[i], EGL_DRM_DEVICE_FILE_EXT);
                        Debug{} << " - device" << i << ":" << device;
                    }
                    else
                        Debug{} << " - device" << i << ":" << extension;
                }
            }

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
            if(!display && strstr(extensions, "EGL_MESA_platform_gbm"))
            {
                display = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_MESA, EGL_DEFAULT_DISPLAY, nullptr);
                if(!display)
                    Debug() << "gpm failed";
            }
        }

        return {display, false};
    }
}

using ImporterManager = PluginManager::Manager<Trade::AbstractImporter>;
using ImageConverterManager = PluginManager::Manager<Trade::AbstractImageConverter>;

class Context::Private
{
public:
    Private(const std::string& installPrefix = {})
    {
        cudaDebug = envFlag("STILLLEBEN_CUDA_DEBUG");

        int argc = 3;
        std::vector<const char*> argv{
            "dummy",
            "--magnum-log", "quiet"
        };
        gl_context.reset(new Platform::GLContext{NoCreate, argc, argv.data()});

        if(!installPrefix.empty())
        {
#ifndef NDEBUG
            importerPath = installPrefix + "/lib/magnum-d/importers";
            imageConverterPath = installPrefix + "/lib/magnum-d/imageconverters";
#else
            importerPath = installPrefix + "/lib/magnum/importers";
            imageConverterPath = installPrefix + "/lib/magnum/imageconverters";
#endif
        }

        // Setup PhysX stuff
        pxFoundation.reset(
            PxCreateFoundation(PX_PHYSICS_VERSION, pxAllocator, pxErrorCallback)
        );
        pxPvd.reset(
            PxCreatePvd(*pxFoundation)
        );
        pxPvdTransport.reset(
//             physx::PxDefaultPvdFileTransportCreate("/tmp/test.pvd")
            physx::PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10)
        );
        pxPvd->connect(
            *pxPvdTransport,
            physx::PxPvdInstrumentationFlag::eALL
        );

        physx::PxTolerancesScale scale;

        pxPhysics.reset(PxCreatePhysics(
            PX_PHYSICS_VERSION, *pxFoundation, scale, true, pxPvd.get()
        ));

        pxCooking.reset(PxCreateCooking(
            PX_PHYSICS_VERSION, *pxFoundation, physx::PxCookingParams(scale)
        ));
    }

    ~Private()
    {
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(egl_display, egl_context);
        eglTerminate(egl_display);
    }

    void* egl_display = nullptr;
    void* egl_context = nullptr;
    EGLConfig egl_config{};

    std::unique_ptr<Platform::GLContext> gl_context;

    DebugTools::ResourceManager resourceManager;

    physx::PxDefaultAllocator pxAllocator;
    physx::PxDefaultErrorCallback pxErrorCallback;
    PhysXHolder<physx::PxFoundation> pxFoundation;
    PhysXHolder<physx::PxPvdTransport> pxPvdTransport;
    PhysXHolder<physx::PxPvd> pxPvd;
    PhysXHolder<physx::PxPhysics> pxPhysics;
    PhysXHolder<physx::PxCooking> pxCooking;

    bool cudaDebug = false;

    std::string importerPath;
    std::string imageConverterPath;
};

Context::Context(const std::string& installPrefix)
 : m_d{std::make_unique<Private>(installPrefix)}
{
}

Context::Ptr Context::Create(const std::string& installPrefix)
{
    Context::Ptr context{new Context(installPrefix)};

    auto displayConfig = getEglDisplay();
    EGLDisplay display = displayConfig.display;

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

    if constexpr(false)
    {
        Debug{} << "Display initialized for EGL " << major << "." << minor;

        const char* vendor = eglQueryString(display, EGL_VENDOR);
        if(vendor)
            Debug{} << "EGL vendor:" << vendor;
    }

    if(!eglBindAPI(EGL_OPENGL_API))
    {
        Error() << "Could not bind OpenGL API";
        return {};
    }

    EGLint surfaceType = EGL_PBUFFER_BIT;
    if(displayConfig.useX11)
        surfaceType |= EGL_WINDOW_BIT;

    EGLint numberConfigs;
    EGLConfig eglConfig;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT | EGL_WINDOW_BIT,
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
//         EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,

        EGL_NONE
    };

    context->m_d->egl_context = eglCreateContext(context->m_d->egl_display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    if(!context->m_d->egl_context)
    {
        Error() << "Could not create EGL context:" << eglErrorString(eglGetError());
        return {};
    }

    context->m_d->egl_config = eglConfig;

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
}

Context::Ptr Context::CreateCUDA(unsigned int device, const std::string& installPrefix)
{
    std::shared_ptr<Context> context(new Context(installPrefix));

    if(context->m_d->cudaDebug)
        Debug{} << "stillleben CUDA initialization";

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

    if(context->m_d->cudaDebug)
    {
        Debug{} << "Found EGL devices:";
        for(auto& dev : devices)
        {
            EGLAttrib devCudaIndex;
            if(!eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, &devCudaIndex))
                Debug{} << " - non-CUDA device";
            else
                Debug{} << " - CUDA device" << devCudaIndex;
        }
    }

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

    context->m_d->egl_config = eglConfig;

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
}

bool Context::makeCurrent()
{
    Debug{} << "Context::makeCurrent()";
    if(!eglMakeCurrent(m_d->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, m_d->egl_context))
    {
        Error() << "Cannot make context current";
        return false;
    }

    return true;
}

Magnum::GL::RectangleTexture Context::loadTexture(const std::string& path)
{
    Corrade::PluginManager::Manager<Trade::AbstractImporter> manager(importerPluginPath());
    auto importer = manager.loadAndInstantiate("AnyImageImporter");

    std::ostringstream ss;
    Error redirectTo{&ss};

    if(!importer)
        throw std::logic_error("Could not load AnyImageImporter plugin");

    if(!importer->openFile(path))
        throw std::runtime_error("Could not open image file: " + ss.str());

    auto image = importer->image2D(0);
    if(!image)
        throw std::runtime_error("Could not load image: " + ss.str());

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

    // Needed for sticker textures - this is ugly.
    texture.setWrapping(Magnum::SamplerWrapping::ClampToBorder);
    texture.setBorderColor(Magnum::Color4{0.0, 0.0, 0.0, 0.0});

    std::string messages = ss.str();
    if(!messages.empty())
        std::cerr << messages << std::flush;

    return texture;
}

Magnum::GL::Texture2D Context::loadTexture2D(const std::string& path)
{
    Corrade::PluginManager::Manager<Trade::AbstractImporter> manager(importerPluginPath());
    auto importer = manager.loadAndInstantiate("AnyImageImporter");

    std::ostringstream ss;
    Error redirectTo{&ss};

    if(!importer)
        throw std::logic_error("Could not load AnyImageImporter plugin");

    if(!importer->openFile(path))
        throw std::runtime_error("Could not open image file: " + ss.str());

    auto image = importer->image2D(0);
    if(!image)
        throw std::runtime_error("Could not load image: " + ss.str());

    GL::TextureFormat format;
    if(image->format() == PixelFormat::RGB8Unorm)
        format = GL::TextureFormat::RGB8;
    else if(image->format() == PixelFormat::RGBA8Unorm)
        format = GL::TextureFormat::RGBA8;
    else
        throw std::runtime_error("Unsupported texture format");

    GL::Texture2D texture;
    texture.setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy());
    texture.setStorage(Math::log2(image->size().max())+1, format, image->size());
    texture.setSubImage(0, {}, *image);
    texture.generateMipmap();

    // Needed for sticker textures - this is ugly.
    texture.setWrapping(Magnum::SamplerWrapping::ClampToBorder);
    texture.setBorderColor(Magnum::Color4{0.0, 0.0, 0.0, 0.0});

    std::string messages = ss.str();
    if(!messages.empty())
        std::cerr << messages << std::flush;

    return texture;
}

physx::PxPhysics& Context::physxPhysics()
{
    return *m_d->pxPhysics;
}

physx::PxCooking& Context::physxCooking()
{
    return *m_d->pxCooking;
}

std::string Context::importerPluginPath() const
{
    return m_d->importerPath;
}

std::string Context::imageConverterPluginPath() const
{
    return m_d->imageConverterPath;
}

Magnum::DebugTools::ResourceManager& Context::debugResourceManager()
{
    return m_d->resourceManager;
}

int Context::visualID() const
{
    EGLint id;
    if(!eglGetConfigAttrib(m_d->egl_display, m_d->egl_config, EGL_NATIVE_VISUAL_ID, &id))
        throw std::runtime_error{"Could not query visual ID"};

    Debug{} << "Context: got EGL visual ID" << id;

    return id;
}

void* Context::eglConfig() const
{
    return m_d->egl_config;
}

}
