// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>

static thread_local std::shared_ptr<sl::Context> g_context;

// Conversion functions
static at::Tensor magnumToTorch(const Magnum::Matrix4& mat)
{
    auto tensor = torch::CPU(at::kFloat).tensorFromBlob(
        const_cast<float*>(mat.data()),
        {4,4}
    );

    // NOTE: Magnum matrices are column-major
    return tensor.t().clone();
}


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

static std::shared_ptr<sl::Mesh> Mesh_factory(const std::string& filename)
{
    if(!g_context)
        throw std::logic_error("You need to call init() first!");

    auto mesh = std::make_shared<sl::Mesh>(g_context);

    mesh->load(filename);

    return mesh;
}

static at::Tensor Mesh_bbox(const std::shared_ptr<sl::Mesh>& mesh)
{
    auto bbox = mesh->bbox();

    auto min = bbox.min();
    auto max = bbox.max();

    float data[]{
        min.x(), max.x(),
        min.y(), max.y(),
        min.z(), max.z()
    };

    return torch::from_blob(data, {3,2}).clone();
}

static at::Tensor Mesh_pretransform(const std::shared_ptr<sl::Mesh>& mesh)
{
    return magnumToTorch(mesh->pretransform());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "Init without CUDA support");
    m.def("initCUDA", &initCUDA, "Init with CUDA support", py::arg("device_index") = 0);

    // sl::Mesh
    py::class_<sl::Mesh, std::shared_ptr<sl::Mesh>>(m, "Mesh", R"EOS(
            Represents a loaded mesh file. A Mesh can be seen as an object template.
            In order to be rendered, you need to instantiate it
            (see :func:`Object.instantiate`).
        )EOS")

        .def(py::init(&Mesh_factory), R"EOS(
            Constructor

            Args:
                filename (str): Mesh filename

            Examples:
                >>> m = Mesh("path/to/my/mesh.gltf")
        )EOS", py::arg("filename"))

        .def_property_readonly("bbox", &Mesh_bbox, R"EOS(
            Mesh bounding box (3 x 2 tensor with min/max in each dimension)
        )EOS")

        .def("center_bbox", &sl::Mesh::centerBBox, R"EOS(
            Modifies the pretransform such that the bounding box
            (see `bbox`) is centered at the origin.
        )EOS")

        .def("scale_to_bbox_diagonal", &sl::Mesh::scaleToBBoxDiagonal, R"EOS(
            Modifies the pretransform such that the bounding box diagonal
            (see `bbox`) is equal to :attr:`target_diagonal`.

            Args:
                target_diagonal (float): Target diagonal
        )EOS", py::arg("target_diagonal"))

        .def_property_readonly("pretransform", &Mesh_pretransform, R"EOS(
            The current pretransform matrix. Initialized to identity and
            modified by :func:`center_bbox` and :func:`scale_to_bbox_diagonal`.
        )EOS")

        .def_property("class_index",
            &sl::Mesh::classIndex, &sl::Mesh::setClassIndex, R"EOS(
            Class index for training semantic segmentation.
        )EOS")
    ;
}
