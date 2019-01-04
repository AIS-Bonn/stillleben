
# Magnum options
set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE BOOL "With WindowlessEglApplication")
set(WITH_ANYIMAGEIMPORTER ON CACHE BOOL "With AnyImageImporter")
set(WITH_EGLCONTEXT ON)

# magnum-plugins options
set(WITH_ASSIMPIMPORTER ON CACHE BOOL "With AssimpImporter")
set(WITH_JPEGIMPORTER ON CACHE BOOL "With JPEGImporter")
set(WITH_PNGIMPORTER ON CACHE BOOL "With PNGImporter")
set(WITH_PNGIMAGECONVERTER ON CACHE BOOL "With PNGConverter")

# magnum-integration options
set(WITH_BULLET ON CACHE BOOL "Build BulletIntegration library")

add_subdirectory(corrade)
add_subdirectory(magnum)
add_subdirectory(magnum-plugins)
add_subdirectory(magnum-integration)
