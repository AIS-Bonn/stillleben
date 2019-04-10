
find_package(Magnum REQUIRED
    GL
    MeshTools
    Shaders
    SceneGraph
    Trade
    Sdl2Application
    DebugTools
)

find_package(MagnumIntegration REQUIRED ImGui)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(align_mesh align_mesh.cpp)
target_link_libraries(align_mesh PRIVATE
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::MeshTools
    Magnum::SceneGraph
    Magnum::Shaders
    Magnum::Trade
    Magnum::DebugTools
    MagnumIntegration::ImGui
    stdc++fs
)
