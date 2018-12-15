// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/torch.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>

static thread_local std::shared_ptr<sl::Context> g_context;

static void init()
{
    g_context = sl::Context::Create();
    if(!g_context)
        throw std::runtime_error("Could not create stillleben context");
}

static void initCUDA(unsigned int cudaIndex)
{
    g_context = sl::Context::CreateCUDA(cudaIndex);
    if(!g_context)
        throw std::runtime_error("Could not create stillleben context");
}

static std::shared_ptr<Mesh> Mesh_factory()
{
    if(!g_context)
        throw std::logic_error("You need to call init() first!");
    return {new Mesh(g_context)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "Init without CUDA support");
    m.def("initCUDA", &init, "Init with CUDA support", py::arg("device_index") = 0);

    py::class_<sl::Mesh, std::shared_ptr<sl::Mesh>>(m, "Mesh")
        .def(py::init(&Mesh_factory))
        .def("load", &sl::Mesh::load, py::arg("filename"))
    ;
}
