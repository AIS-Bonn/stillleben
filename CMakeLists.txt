
cmake_minimum_required(VERSION 3.0)
project(stillleben)

# register CMake modules in modules/
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/")

# set(OpenGL_GL_PREFERENCE GLVND)
# find_package(OpenGL)

if(OpenGL_EGL_FOUND)
    message(STATUS "Found EGL, using that one...")
    add_definitions(-DHAVE_EGL=1)
    set(MAGNUM_GL_APP WindowlessEglApplication)
else()
    message(STATUS "Could not find EGL/GLVND, falling back to GLX")
    set(MAGNUM_GL_APP GlfwApplication)
endif()

find_package(Magnum REQUIRED COMPONENTS
    GL
    MeshTools
    Shaders
    SceneGraph
    Trade
    GlfwApplication
)

include_directories(${OPENGL_INCLUDE_DIR} ${OPENGL_EGL_INCLUDE_DIRS})

find_package(CUDA)
# include_directories(${CUDA_INCLUDE_DIRS})

# Package shaders
corrade_add_resource(render_RESOURCES resources.conf)

include_directories(include)

set(CMAKE_CXX_STANDARD 14)

add_library(stillleben SHARED
    src/context.cpp
    src/mesh.cpp
    src/object.cpp
    src/phong_pass.cpp
    src/scene.cpp
    src/stillleben.cpp
    ${render_RESOURCES}
)
target_link_libraries(stillleben PRIVATE
    Magnum::GL
    Magnum::Magnum
    Magnum::SceneGraph
    Magnum::Shaders
    Magnum::Trade
    Magnum::MeshTools
    Magnum::${MAGNUM_GL_APP}
#     OpenGL::EGL
#     ${CUDA_LIBRARIES}
)
