
# Magnum options
set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE BOOL "With WindowlessEglApplication" FORCE)
set(WITH_XEGLAPPLICATION ON CACHE BOOL "With XEglApplication" FORCE)
set(WITH_ANYIMAGEIMPORTER ON CACHE BOOL "With AnyImageImporter" FORCE)
set(WITH_EGLCONTEXT ON)

# magnum-plugins options
set(WITH_ASSIMPIMPORTER ON CACHE BOOL "With AssimpImporter" FORCE)
set(WITH_JPEGIMPORTER ON CACHE BOOL "With JPEGImporter" FORCE)
set(WITH_PNGIMPORTER ON CACHE BOOL "With PNGImporter" FORCE)
set(WITH_PNGIMAGECONVERTER ON CACHE BOOL "With PNGConverter" FORCE)
set(WITH_STBIMAGEIMPORTER ON CACHE BOOL "With StbImageImporter" FORCE)
set(WITH_TINYGLTFIMPORTER ON CACHE BOOL "With TinyGltfImporter" FORCE)

if(WITH_ALIGN_MESH)
    set(WITH_SDL2APPLICATION ON CACHE BOOL "With Sdl2Application" FORCE)
    set(WITH_IMGUI ON CACHE BOOL "With IMGUI" FORCE)
endif()

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

add_subdirectory(corrade)
add_subdirectory(magnum)
add_subdirectory(magnum-plugins)
add_subdirectory(magnum-integration)
add_subdirectory(physx)

# hackery.
add_library(MagnumPlatformPrivate INTERFACE)
target_include_directories(MagnumPlatformPrivate
    INTERFACE
    magnum/src/Magnum/Platform/Implementation
)
