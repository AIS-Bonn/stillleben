
cmake_minimum_required(VERSION 3.0)
project(stillleben)

# register CMake modules in modules/
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/")

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL)

if(OpenGL_EGL_FOUND)
    message(STATUS "Found EGL, using that one...")
    add_definitions(-DHAVE_EGL=1)
    set(MAGNUM_GL_APP WindowlessEglApplication)
else()
    message(STATUS "Could not find EGL/GLVND, falling back to GLX")
    set(MAGNUM_GL_APP WindowlessGlxApplication)
endif()

# Viewer wants Xcursor
# TODO: move into own file
add_library(Xcursor INTERFACE)
if(X11_Xcursor_INCLUDE_PATH AND X11_Xcursor_LIB)
    target_include_directories(Xcursor INTERFACE ${X11_Xcursor_INCLUDE_PATH})
    target_link_libraries(Xcursor INTERFACE ${X11_Xcursor_LIB})
    target_compile_definitions(Xcursor INTERFACE HAVE_XCURSOR=1)
endif()

option(WITH_ALIGN_MESH "Build the align_mesh tool" OFF)
option(USE_RELATIVE_RPATH "Set rpath for standalone python installation" OFF)

if(USE_RELATIVE_RPATH)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

set(EXTRA_RPATH "" CACHE STRING "Extra RPATH")
if(EXTRA_RPATH)
    list(APPEND CMAKE_INSTALL_RPATH "${EXTRA_RPATH}")
endif()

list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# ninja is cool, but suppresses color output
# https://medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options(-fdiagnostics-color=always)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options(-fcolor-diagnostics)
    endif()
endif()

# Build Corrade & Magnum
add_subdirectory(contrib)

find_package(Magnum REQUIRED COMPONENTS
    GL
    MeshTools
    Primitives
    Shaders
    SceneGraph
    Trade
    DebugTools
    ${MAGNUM_GL_APP}
)

include_directories(${OPENGL_INCLUDE_DIR} ${OPENGL_EGL_INCLUDE_DIRS})

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DHAVE_CUDA=1)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

corrade_add_resource(stillleben_RESOURCES src/shaders/resources.conf)

add_library(stillleben SHARED
    src/animator.cpp
    src/context.cpp
    src/cuda_interop.cpp
    src/debug.cpp
    src/exception.cpp
    src/image_loader.cpp
    src/light_map.cpp
    src/mesh.cpp
    src/mesh_cache.cpp
    src/object.cpp
    src/phong_pass.cpp
    src/pose.cpp
    src/render_pass.cpp
    src/scene.cpp
    src/stillleben.cpp
    src/viewer.cpp
    src/shaders/background_shader.cpp
    src/shaders/background_cube_shader.cpp
    src/shaders/brdf_shader.cpp
    src/shaders/check_attributes.cpp
    src/shaders/cubemap_shader.cpp
    src/shaders/render_shader.cpp
    src/shaders/ssao_shader.cpp
    src/shaders/ssao_apply_shader.cpp
    src/shaders/viewer/viewer_shader.cpp
    src/mesh_tools/pbr_material_data.cpp
    src/mesh_tools/tangents.cpp
    src/utils/arc_ball.cpp
    src/utils/os.cpp
    ${stillleben_RESOURCES}
)
target_include_directories(stillleben PUBLIC
    include
)
target_include_directories(stillleben PRIVATE
    contrib/magnum-plugins/src
)
target_link_libraries(stillleben PUBLIC
    Magnum::GL
    Magnum::Magnum
    Magnum::SceneGraph
    Magnum::Shaders
    Magnum::Trade
    Magnum::MeshTools
    Magnum::Primitives
    Magnum::DebugTools
    Magnum::XEglApplication
    MagnumIntegration::ImGui
    MagnumPlatformPrivate
    PhysX
    Xcursor
)

if(CUDA_FOUND)
    target_link_libraries(stillleben PUBLIC
        ${CUDA_LIBRARIES}
    )
endif()

install(TARGETS stillleben
    LIBRARY DESTINATION lib
)
install(DIRECTORY
    include/stillleben/
    DESTINATION include/stillleben
)

add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(python)
