
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

option(WITH_ALIGN_MESH "Build the align_mesh tool" OFF)

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

include_directories(include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

corrade_add_resource(stillleben_RESOURCES src/shaders/resources.conf)

add_library(stillleben SHARED
    src/animator.cpp
    src/context.cpp
    src/cuda_interop.cpp
    src/debug.cpp
    src/exception.cpp
    src/mesh.cpp
    src/mesh_cache.cpp
    src/object.cpp
    src/phong_pass.cpp
    src/pose.cpp
    src/render_pass.cpp
    src/scene.cpp
    src/stillleben.cpp
    src/shaders/background_shader.cpp
    src/shaders/render_shader.cpp
    src/shaders/resolve_shader.cpp
    ${stillleben_RESOURCES}
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
    MagnumIntegration::Bullet
    Magnum::${MAGNUM_GL_APP}
    PhysX
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
