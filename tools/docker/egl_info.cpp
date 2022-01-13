// EGL info tool
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <cstring>
#include <sstream>
#include <iostream>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#if HAVE_CUDA
#include <cuda_gl_interop.h>
#endif

template<class T>
T getExtension(const char* name)
{
    return reinterpret_cast<T>(eglGetProcAddress(name));
}

namespace
{
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

        std::stringstream ss;
        ss << "Unknown error " << error;
        return ss.str();
    }

    struct DisplayConfig
    {
        DisplayConfig(DisplayConfig&) = delete;
        DisplayConfig& operator=(const DisplayConfig&) = delete;

        EGLDisplay display{};
        Display* x11 = nullptr;
    };

    DisplayConfig getEglDisplay()
    {
        EGLDisplay display = nullptr;

        const char* extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
        if(!extensions)
        {
            std::cerr << "Could not query EGL extensions: " << eglErrorString(eglGetError()) << "\n";
            return {};
        }

        auto eglGetPlatformDisplayEXT = getExtension<PFNEGLGETPLATFORMDISPLAYEXTPROC>("eglGetPlatformDisplayEXT");
        if(!eglGetPlatformDisplayEXT)
        {
            std::cerr << "Could not find required eglGetPlatformDisplayEXT: " << eglErrorString(eglGetError()) << "\n";
            return {};
        }

        // Try X11 first, if available. The user will likely want to use
        // the same device as X11 is running on...
        if(getenv("DISPLAY") && strstr(extensions, "EGL_EXT_platform_x11"))
        {
            auto x11Display = XOpenDisplay(nullptr);

            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_X11_EXT, x11Display, nullptr);
            if(display)
            {
                std::cout << "Display opened via EGL_PLATFORM_X11_EXT\n";
                return {display, x11Display};
            }
            else
                std::cerr << "X11 failed\n";
        }

        // Load required extensions for enumeration
        auto eglQueryDevicesEXT = getExtension<PFNEGLQUERYDEVICESEXTPROC>("eglQueryDevicesEXT");
        if(!eglQueryDevicesEXT)
        {
            std::cerr << "Could not find required eglQueryDevicesEXT: " << eglErrorString(eglGetError()) << "\n";
            return {};
        }

        auto eglQueryDeviceStringEXT = getExtension<PFNEGLQUERYDEVICESTRINGEXTPROC>("eglQueryDeviceStringEXT");

        // Enumerate devices
        std::vector<EGLDeviceEXT> devices(64);
        EGLint num_devices;

        if(!eglQueryDevicesEXT(devices.size(), devices.data(), &num_devices))
        {
            std::cerr << "Could not enumerate EGL devices: " << eglErrorString(eglGetError()) << "\n";
            return {};
        }

        if(num_devices > 0)
        {
            std::cerr << "Found EGL device(s) (count:" << num_devices << "), trying to create display...\n";

            if(eglQueryDeviceStringEXT)
            {
                for(EGLint i = 0; i < num_devices; ++i)
                {
                    const char* extension = eglQueryDeviceStringEXT(devices[i], EGL_EXTENSIONS);
                    if(strcmp(extension, "EGL_EXT_device_drm") == 0)
                    {
                        const char* device = eglQueryDeviceStringEXT(devices[i], EGL_DRM_DEVICE_FILE_EXT);
                        std::cerr << " - device" << i << ":" << device << "\n";
                    }
                    else
                        std::cerr << " - device" << i << ":" << extension << "\n";
                }
            }

            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices.front(), nullptr);
            if(display)
            {
                std::cout << "Display opened via EGL_PLATFORM_DEVICE_EXT\n";
            }
        }

        if(!display)
        {
            std::cerr << "Could not enumerate EGL devices, trying MESA targets with EGL_DEFAULT_DISPLAY...\n";

            if(strstr(extensions, "EGL_MESA_platform_surfaceless"))
            {
                display = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
                if(!display)
                    std::cerr << "surfaceless failed: " << eglErrorString(eglGetError()) << "\n";
                else
                    std::cout << "surfaceless\n";
            }
            if(!display && strstr(extensions, "EGL_MESA_platform_gbm"))
            {
                display = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_MESA, EGL_DEFAULT_DISPLAY, nullptr);
                if(!display)
                    std::cerr << "gpm failed: " << eglErrorString(eglGetError()) << "\n";
                else
                    std::cout << "gpm\n";
            }
        }

        return {display, nullptr};
    }
}

int main(int argc, char** argv)
{
    DisplayConfig displayConfig = getEglDisplay();
    EGLDisplay egl_display = displayConfig.display;

    EGLint major, minor;
    if(!eglInitialize(egl_display, &major, &minor))
    {
        std::cerr << "Could not initialize EGL display: " << eglErrorString(eglGetError()) << "\n";
        return false;
    }

    {
        std::cout << "Display initialized for EGL " << major << "." << minor << "\n";

        const char* vendor = eglQueryString(egl_display, EGL_VENDOR);
        if(vendor)
            std::cout << "EGL vendor:" << vendor << "\n";
    }

    if(!eglBindAPI(EGL_OPENGL_API))
    {
        std::cout << "Could not bind OpenGL API: " << eglErrorString(eglGetError()) << "\n";
        return false;
    }

    EGLint surfaceType = EGL_PBUFFER_BIT;
    if(displayConfig.x11)
        surfaceType |= EGL_WINDOW_BIT;

    EGLint numberConfigs;
    EGLConfig eglConfig;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, surfaceType,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,

        EGL_NONE
    };
    if(!eglChooseConfig(egl_display, configAttribs, &eglConfig, 1, &numberConfigs))
    {
        std::cerr << "Could not call eglChooseConfig: " << eglErrorString(eglGetError()) << "\n";
        return false;
    }

    if(!numberConfigs)
    {
        std::cerr << "Could not find any matching EGL config :-(\n";
        return false;
    }

    EGLint contextAttribs[]{
//         EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,

        EGL_NONE
    };

    EGLContext egl_context = eglCreateContext(egl_display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    if(!egl_context)
    {
        std::cerr << "Could not create EGL context:" << eglErrorString(eglGetError()) << "\n";
        return false;
    }

    return 0;
}
