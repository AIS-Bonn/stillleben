
# Magnum options
set(WITH_WINDOWLESSEGLAPPLICATION ON)
set(WITH_ANYIMAGEIMPORTER ON)

# magnum-plugins options
set(WITH_ASSIMPIMPORTER ON)
set(WITH_JPEGIMPORTER ON)
set(WITH_PNGIMPORTER ON)

add_subdirectory(corrade)
add_subdirectory(magnum)
add_subdirectory(magnum-plugins)

