
# Magnum options
set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE BOOL "With WindowlessEglApplication" FORCE)
set(WITH_XEGLAPPLICATION ON CACHE BOOL "With XEglApplication" FORCE)
set(WITH_ANYIMAGEIMPORTER ON CACHE BOOL "With AnyImageImporter" FORCE)
set(WITH_ANYIMAGECONVERTER ON CACHE BOOL "With AnyImageConverter" FORCE)
set(WITH_EGLCONTEXT ON CACHE BOOL "With EglContext" FORCE)

if(WITH_ALIGN_MESH)
    set(WITH_SDL2APPLICATION ON CACHE BOOL "With Sdl2Application" FORCE)
endif()

# magnum-plugins options
set(WITH_ASSIMPIMPORTER ON CACHE BOOL "With AssimpImporter" FORCE)
set(WITH_JPEGIMPORTER ON CACHE BOOL "With JPEGImporter" FORCE)
set(WITH_PNGIMPORTER ON CACHE BOOL "With PNGImporter" FORCE)
set(WITH_PNGIMAGECONVERTER ON CACHE BOOL "With PNGImageConverter" FORCE)
set(WITH_JPEGIMAGECONVERTER ON CACHE BOOL "With JPEGImageConverter" FORCE)
set(WITH_STBIMAGEIMPORTER ON CACHE BOOL "With StbImageImporter" FORCE)
set(WITH_CGLTFIMPORTER ON CACHE BOOL "With CGltfImporter" FORCE)
set(WITH_STANFORDSCENECONVERTER ON CACHE BOOL "With StanfordSceneConverter" FORCE)
set(WITH_ANYSCENECONVERTER ON CACHE BOOL "With AnySceneConverter" FORCE)

# magnum-integration
set(WITH_IMGUI ON CACHE BOOL "With IMGUI" FORCE)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

add_subdirectory(corrade)
add_subdirectory(magnum)
add_subdirectory(magnum-plugins)
add_subdirectory(magnum-integration)
add_subdirectory(physx)
add_subdirectory(v-hacd)

# hackery.
add_library(MagnumPlatformPrivate INTERFACE)
target_include_directories(MagnumPlatformPrivate
    INTERFACE
    magnum/src/Magnum/Platform/Implementation
)
