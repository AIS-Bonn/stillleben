// sl::Mesh binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_mesh.h"
#include "py_context.h"
#include "py_magnum.h"

#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>

using namespace sl::python::magnum;

namespace
{
    std::shared_ptr<sl::Mesh> Mesh_factory(
        const std::string& filename,
        std::size_t maxPhysicsTriangles,
        bool visual, bool physics)
    {
        if(!sl::python::Context::instance())
            throw std::logic_error("You need to call init() first!");

        auto mesh = std::make_shared<sl::Mesh>(filename, sl::python::Context::instance());
        mesh->load(maxPhysicsTriangles, visual, physics);

        return mesh;
    }

    std::vector<std::shared_ptr<sl::Mesh>> Mesh_loadThreaded(
        const std::vector<std::string>& filenames,
        bool visual, bool physics,
        std::size_t maxPhysicsTriangles)
    {
        if(!sl::python::Context::instance())
            throw std::logic_error("You need to call init() first!");

        return sl::Mesh::loadThreaded(sl::python::Context::instance(), filenames, visual, physics, maxPhysicsTriangles);
    }

    void Mesh_scaleToBBoxDiagonal(const std::shared_ptr<sl::Mesh>& mesh, float diagonal, const std::string& modeStr)
    {
        sl::Mesh::Scale mode;
        if(modeStr == "exact")
            mode = sl::Mesh::Scale::Exact;
        else if(modeStr == "order_of_magnitude")
            mode = sl::Mesh::Scale::OrderOfMagnitude;
        else
            throw std::invalid_argument("invalid value '" + modeStr + "' for mode argument");

        mesh->scaleToBBoxDiagonal(diagonal, mode);
    }

    void Mesh_updatePositions(const std::shared_ptr<sl::Mesh>& mesh,
        torch::Tensor& verticesIndex,
        torch::Tensor& positionsUpdate)
    {
        if(verticesIndex.dim() != 1)
            throw std::invalid_argument{"vertex_indices (1st argument) should be one dimensional"};
        if(positionsUpdate.dim() != 2)
            throw std::invalid_argument{"position_update (2nd argument) should be two dimensional"};
        if(verticesIndex.size(0) != positionsUpdate.size(0))
            throw std::invalid_argument{"vertex_indices and position_update should be of same size"};
        if(positionsUpdate.size(1) != 3)
            throw std::invalid_argument{"position_update should be of shape (N,3)"};
        if(verticesIndex.device().type() == torch::kCUDA || positionsUpdate.device().type() == torch::kCUDA)
            throw std::invalid_argument{"vertex_indices and position_update should be CPU tensors"};

        if(!verticesIndex.is_contiguous())
        {
            throw std::invalid_argument{"vertex_indices should be contiguous tensor\n"
                "(use vertex_indices.contiguous() before passing vertex_indices as an argument)"};
        }

        if(!positionsUpdate.is_contiguous())
        {
            throw std::invalid_argument{"position_update should be contiguous tensor\n"
                "(use position_update.contiguous() before passing position_update as an argument)"};
        }

        Corrade::Containers::ArrayView<int> vertexIndexView{
            verticesIndex.data_ptr<int>(),
            static_cast<std::size_t>(verticesIndex.numel())
        };

        auto positionUpdateView = Corrade::Containers::arrayCast<Magnum::Vector3>(
            Corrade::Containers::arrayView(
                positionsUpdate.data_ptr<float>(), positionsUpdate.numel()
            )
        );
        mesh->updateVertexPositions(vertexIndexView, positionUpdateView);
    }

    void Mesh_updateColors(const std::shared_ptr<sl::Mesh>& mesh,
        torch::Tensor& verticesIndex,
        torch::Tensor& ColorsUpdate)
    {
        if(verticesIndex.dim() != 1)
            throw std::invalid_argument{"vertex_indices (1st argument) should be one dimensional"};

        if(ColorsUpdate.dim() != 2)
            throw std::invalid_argument{"color_update (2nd argument) should be two dimensional"};

        if(verticesIndex.size(0) != ColorsUpdate.size(0))
            throw std::invalid_argument{"vertex_indices and colors_update should be of same size"};

        if(ColorsUpdate.size(1) != 4)
            throw std::invalid_argument{"color_update should be of shape (N,4)"};

        if(verticesIndex.device().type() == torch::kCUDA || ColorsUpdate.device().type() == torch::kCUDA)
            throw std::invalid_argument{"vertex_indices and colors_update should be CPU tensors"};

        if(!verticesIndex.is_contiguous())
        {
            throw std::invalid_argument{"vertex_indices should be contiguous tensor\n"
                        "(use vertex_indices.contiguous() before passing vertex_indices as an argument)"};
        }

        if(!ColorsUpdate.is_contiguous())
        {
            throw std::invalid_argument{"colors_update should be contiguous tensor\n"
                        "(use colors_update.contiguous() before passing colors_update as an argument)"};
        }

        Corrade::Containers::ArrayView<int> vertexIndexView{
            verticesIndex.data_ptr<int>(),
            static_cast<std::size_t>(verticesIndex.numel())
        };

        auto colorUpdateView = Corrade::Containers::arrayCast<Magnum::Color4>(
            Corrade::Containers::arrayView(
                ColorsUpdate.data_ptr<float>(), ColorsUpdate.numel()
            )
        );

        mesh->updateVertexColors(vertexIndexView, colorUpdateView);
    }

    void Mesh_updatePositionsAndColors(const std::shared_ptr<sl::Mesh>& mesh,
        torch::Tensor& verticesIndex,
        torch::Tensor& positionsUpdate,
        torch::Tensor& ColorsUpdate)
    {
        if(verticesIndex.dim() != 1)
            throw std::invalid_argument{"vertex_indices (1st argument) should be one dimensional"};
        if(positionsUpdate.dim() != 2)
            throw std::invalid_argument{"position_update (2nd argument) should be two dimensional"};
        if(ColorsUpdate.dim() != 2)
            throw std::invalid_argument{"color_update (3rd argument) should be two dimensional"};
        if(verticesIndex.size(0) != positionsUpdate.size(0))
            throw std::invalid_argument{"vertices_index  and vertices_update should be of same size"};
        if(verticesIndex.size(0) != ColorsUpdate.size(0))
            throw std::invalid_argument{"vertices_index  and color_update should be of same size"};
        if(positionsUpdate.size(1) != 3)
            throw std::invalid_argument{"position_update should be of shape (N,3)"};
        if(ColorsUpdate.size(1) != 4)
            throw std::invalid_argument{"color_update should be of shape (N,4)"};
        if(verticesIndex.device().type() != torch::kCPU
            || positionsUpdate.device().type() != torch::kCPU
            || ColorsUpdate.device().type() != torch::kCPU )
            throw std::invalid_argument{"vertex_indices, position_update, and color_update should be CPU tensors"};

        if(!verticesIndex.is_contiguous())
        {
            throw std::invalid_argument{"vertex_indices should be contiguous tensor\n"
                "(use vertex_indices.contiguous() before passing vertex_indices as an argument)"};
        }

        if(!positionsUpdate.is_contiguous())
        {
            throw std::invalid_argument{"position_update should be contiguous tensor\n"
                "(use position_update.contiguous() before passing position_update as an argument)"};
        }

        if(!ColorsUpdate.is_contiguous())
        {
            throw std::invalid_argument{"color_update should be contiguous tensor\n"
                "(use color_update.contiguous() before passing color_update as an argument)"};
        }

        Corrade::Containers::ArrayView<int> vertexIndexView{
            verticesIndex.data_ptr<int>(),
            static_cast<std::size_t>(verticesIndex.numel())
        };

        auto positionUpdateView = Corrade::Containers::arrayCast<Magnum::Vector3>(
            Corrade::Containers::arrayView(
                positionsUpdate.data_ptr<float>(), positionsUpdate.numel()
            )
        );
        auto colorUpdateView = Corrade::Containers::arrayCast<Magnum::Color4>(
            Corrade::Containers::arrayView(
                ColorsUpdate.data_ptr<float>(), ColorsUpdate.numel()
            )
        );

        mesh->updateVertexPositionsAndColors(
            vertexIndexView, positionUpdateView, colorUpdateView
        );
    }

    void Mesh_setNewPositions(const std::shared_ptr<sl::Mesh>& mesh,
        torch::Tensor& newVertices)
    {
        if(!newVertices.is_contiguous())
        {
            throw std::invalid_argument{"new_positions should be contiguous tensor\n"
                "(use new_positions.contiguous() before passing new_positions as an argument)"};
        }

        auto newVertexView = Corrade::Containers::arrayCast<Magnum::Vector3>(
            Corrade::Containers::arrayView(newVertices.data_ptr<float>(), newVertices.numel())
        );
        mesh->setVertexPositions(newVertexView);
    }

    void Mesh_setNewColors(const std::shared_ptr<sl::Mesh>& mesh,
        torch::Tensor& newColors)
    {
        if(!newColors.is_contiguous())
            throw std::invalid_argument{"new_colors should be contiguous tensor \n"
                        "(use new_colors.contiguous() before passing new_colors as an argument)"};

        auto newColorView = Corrade::Containers::arrayCast<Magnum::Color4>(
            Corrade::Containers::arrayView(
                newColors.data_ptr<float>(),
                newColors.numel()
            )
        );
        mesh->setVertexColors(newColorView);
    }
}

namespace sl
{
namespace python
{
namespace Mesh
{

void init(py::module& m)
{
    py::class_<sl::Mesh, std::shared_ptr<sl::Mesh>>(m, "Mesh", R"EOS(
            Represents a loaded mesh file. A Mesh can be seen as an object template.
            In order to be rendered, you need to instantiate it
            (see :func:`Object.instantiate`).
        )EOS")

        .def(py::init(&Mesh_factory), R"EOS(
            Constructor

            Args:
                filename (str): Mesh filename
                max_physics_triangles (int): Maximum number of triangles for
                     collision shape. If the mesh is more complex than this,
                     it is simplified using quadric edge decimation.
                     You can view the collision mesh using
                     :func:`render_physics_debug_image`.
                visual (bool): Should we load visual components?
                physics (bool): Should we load collision meshes?

            Examples:
                >>> m = Mesh("path/to/my/mesh.gltf")
        )EOS", py::arg("filename"), py::arg("max_physics_triangles")=sl::Mesh::DefaultPhysicsTriangles, py::arg("visual")=true, py::arg("physics")=true)

        .def_static("load_threaded", &Mesh_loadThreaded, R"EOS(
            Load multiple meshes using a thread pool.

            Args:
                filenames (list): List of file names to load
                visual (bool): Should we load visual components?
                physics (bool): Should we load collision meshes?
                max_physics_triangles (int): Maximum number of triangles for
                    collision shape (see :func:`Mesh`).

            Returns:
                list: List of mesh instances
        )EOS", py::arg("filenames"), py::arg("visual")=true, py::arg("physics")=true, py::arg("max_physics_triangles")=sl::Mesh::DefaultPhysicsTriangles)

        .def_property_readonly("bbox", &sl::Mesh::bbox, R"EOS(
            Mesh bounding box.
        )EOS")

        .def("center_bbox", &sl::Mesh::centerBBox, R"EOS(
            Modifies the pretransform such that the bounding box
            (see `bbox`) is centered at the origin.
        )EOS")

        .def("scale_to_bbox_diagonal", &Mesh_scaleToBBoxDiagonal, R"EOS(
            Modifies the pretransform such that the bounding box diagonal
            (see `bbox`) is equal to :attr:`target_diagonal`.

            Args:
                target_diagonal (float): Target diagonal
                mode (str): Scaling mode (default 'exact').
                    If 'order_of_magnitude', the resulting scale factor is the
                    nearest power of 10 that fits. This is useful for detecting
                    the scale of arbitrary mesh files.
        )EOS", py::arg("target_diagonal"), py::arg("mode")="exact")

        .def("update_positions", &Mesh_updatePositions, R"EOS(
            Updates the mesh vertex positions.

            Args:
                vertex_indices (tensor): N (dim=1)
                position_update (tensor): NX3
        )EOS", py::arg("vertex_indices"), py::arg("position_update"))

        .def("update_colors", &Mesh_updateColors, R"EOS(
            Updates the mesh vertex colors.

            Args:
                vertex_indices (tensor): N (dim=1)
                color_update (tensor): NX4
        )EOS", py::arg("vertex_indices"), py::arg("color_update"))

        .def("update_positions_and_colors", &Mesh_updatePositionsAndColors, R"EOS(
            Updates the mesh verticex positions and vertex colors.

            Args:
                vertex_indices (tensor): N (dim=1)
                position_update (tensor): NX3
                color_update (tensor): NX4
        )EOS", py::arg("vertex_indices"), py::arg("position_update"), py::arg("color_update"))

        .def("set_new_positions", &Mesh_setNewPositions, R"EOS(
            Set new vertex positions.

            Args:
                new_positions (tensor): NX3
        )EOS", py::arg("new_positions"))

        .def("set_new_colors", &Mesh_setNewColors, R"EOS(
            Set new vertex colors.

            Args:
                new_colors (tensor): NX4
        )EOS", py::arg("new_colors"))

        .def_property("pretransform", wrapShared(&sl::Mesh::pretransform), wrapShared(&sl::Mesh::setPretransform), R"EOS(
            The current pretransform matrix. Initialized to identity and
            modified by :func:`center_bbox` and :func:`scale_to_bbox_diagonal`.
        )EOS")

        .def_property("class_index",
            &sl::Mesh::classIndex, &sl::Mesh::setClassIndex, R"EOS(
            Class index for training semantic segmentation.
        )EOS")

        .def_property_readonly("points", wrapShared(&sl::Mesh::meshPoints),
        R"EOS(
            The mesh vertices as (Nx3) float tensor.

            WARNING: This can only be used on single-submesh meshes for now.
            On meshes with multiple submeshes this will raise a RuntimeError.
        )EOS")

        .def_property_readonly("normals", wrapShared(&sl::Mesh::meshNormals),
        R"EOS(
            The mesh normals as (Nx3) float tensor.

            WARNING: This can only be used on single-submesh meshes for now.
            On meshes with multiple submeshes this will raise a RuntimeError.
        )EOS")

        .def_property_readonly("faces", wrapShared(&sl::Mesh::meshFaces),
        R"EOS(
            The mesh faces as (N*3) int32 tensor.

            NOTE: Because PyTorch does not support unsigned ints, the indices
            will wrap around at 2**31.

            WARNING: This can only be used on single-submesh meshes for now.
            On meshes with multiple submeshes this will raise a RuntimeError.
        )EOS")

        .def_property_readonly("colors", wrapShared(&sl::Mesh::meshColors),
        R"EOS(
            The mesh normals as (Nx4) float tensor.

            WARNING: This can only be used on single-submesh meshes for now.
            On meshes with multiple submeshes this will raise a RuntimeError.
        )EOS")
    ;

    py::class_<sl::MeshCache>(m, "MeshCache", R"EOS(
            Caches Mesh instances.
        )EOS")

        .def(py::init([](){ return new sl::MeshCache(sl::python::Context::instance()); }))

        .def("add", &sl::MeshCache::add, R"EOS(
            Add a list of meshes to the cache.

            Args:
                meshes (list): list of :class:`Mesh` instances
        )EOS", py::arg("meshes"))
    ;
}

}
}
}
