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
    // Stolen from magnum-bindings until we properly integrate with it
    template<class T> void enumOperators(pybind11::enum_<T>& e) {
        e
            .def("__or__", [](const T& a, const T& b) { return T(typename std::underlying_type<T>::type(a | b)); })
            .def("__and__", [](const T& a, const T& b) { return T(typename std::underlying_type<T>::type(a & b)); })
            .def("__xor__", [](const T& a, const T& b) { return T(typename std::underlying_type<T>::type(a ^ b)); })
            .def("__invert__", [](const T& a) { return T(typename std::underlying_type<T>::type(~a)); })
            .def("__bool__", [](const T& a) { return bool(typename std::underlying_type<T>::type(a)); });
    }

    std::shared_ptr<sl::Mesh> Mesh_factory(
        const py::object& filename,
        bool visual, bool physics, sl::Mesh::Flag flags = {})
    {
        if(!sl::python::Context::instance())
            throw std::logic_error("You need to call init() first!");

        auto mesh = std::make_shared<sl::Mesh>(py::str(filename), sl::python::Context::instance(), flags);
        mesh->load(visual, physics);

        return mesh;
    }

    std::vector<std::shared_ptr<sl::Mesh>> Mesh_loadThreaded(
        const std::vector<py::object>& filenames,
        bool visual, bool physics, const std::vector<sl::Mesh::Flag> flags = {})
    {
        if(!sl::python::Context::instance())
            throw std::logic_error("You need to call init() first!");

        std::vector<std::string> filenameStrings{filenames.size()};
        for(std::size_t i = 0; i < filenames.size(); ++i)
            filenameStrings[i] = py::str(filenames[i]);

        std::vector<sl::Mesh::Flags> meshFlags(flags.size());
        for(std::size_t i = 0; i < flags.size(); ++i)
            meshFlags[i] = flags[i];

        return sl::Mesh::loadThreaded(sl::python::Context::instance(), filenameStrings, visual, physics, meshFlags);
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

    py::list Mesh_physicsMeshData(const std::shared_ptr<sl::Mesh>& mesh)
    {
        auto& meshDataArray = mesh->physicsMeshData();

        py::list ret;

        // The deleter just holds a reference to the mesh until the tensor is deleted.
        struct Deleter
        {
            std::shared_ptr<sl::Mesh> mesh;

            void operator()(void*)
            {}
        };
        Deleter deleter{mesh};

        for(auto& data : meshDataArray)
        {
            if(!data.hasAttribute(Magnum::Trade::MeshAttribute::Position))
                throw std::logic_error{"Physics mesh has no positions"};
            if(data.attributeFormat(Magnum::Trade::MeshAttribute::Position) != Magnum::VertexFormat::Vector3)
                throw std::logic_error{"Physics mesh has wrong position attribute type"};

            if(data.indexType() != Magnum::MeshIndexType::UnsignedInt)
                throw std::logic_error{"Physics mesh has wrong index type"};
            if(data.primitive() != Magnum::MeshPrimitive::Triangles)
                throw std::logic_error{"Physics mesh has wrong primitive type"};

            auto positionData = data.mutableAttribute(Magnum::Trade::MeshAttribute::Position, 0);

            auto positionTensor = torch::from_blob(
                positionData.data(),
                {data.vertexCount(), 3},
                {long(positionData.stride()[0] / sizeof(float)), long(positionData.stride()[1] / sizeof(float))},
                deleter,
                at::kFloat
            );

            auto indexData = data.mutableIndexData();

            auto indexTensor = torch::from_blob(
                indexData.data(),
                {data.indexCount()},
                deleter,
                at::kInt
            ).view({-1, 3});

            py::dict entry;
            entry["positions"] = positionTensor;
            entry["indices"] = indexTensor;
            ret.append(entry);
        }

        return ret;
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
    auto mesh = py::class_<sl::Mesh, std::shared_ptr<sl::Mesh>>(m, "Mesh", R"EOS(
        Represents a mesh shape.

        Represents a loaded mesh file. A mesh can be seen as an object template.
        In order to be rendered, you need to instantiate it (see :ref:`Object`).

        Typical usage
        -------------

        .. code:: python

            import stillleben as sl

            sl.init()

            # Load a mesh
            mesh = sl.Mesh('my_meshfile.gltf')

            # Add an object referencing the mesh to the scene
            obj = sl.Object(mesh)
            scene = sl.Scene((1920, 1080))
            scene.add_object(obj)

        Accessing and manipulating vertex data
        --------------------------------------

        You can use :ref:`points`, :ref:`faces`, :ref:`colors`, :ref:`normals`
        to access common vertex attributes. :ref:`update_positions` and friends
        can be used to write vertex attributes.

        Sub-Meshes
        ----------

        Some mesh file formats (such as .obj) may contain multiple sub-meshes
        with different properties and textures. Stillleben fully supports
        loading and working with such files.
        If you work with the mesh data itself (see :ref:`points`, :ref:`faces`),
        it will feel like working with a big concatenated mesh. This is more
        convenient in most situations and removes a level of indirection
        when performing vertex updates. All meshes live in the same coordinate
        system.
    )EOS");

    py::enum_<sl::Mesh::Flag> flag{mesh, "Flag", "Mesh flags"};
    flag
        .value("NONE", sl::Mesh::Flag{})
        .value("PHYSICS_FORCE_CONVEX_HULL", sl::Mesh::Flag::PhysicsForceConvexHull, R"EOS(
            Always use a convex hull for physics simulation, even if the object is highly concave.
        )EOS")
    ;
    enumOperators(flag);

    mesh
        .def(py::init(&Mesh_factory), R"EOS(
            Constructor

            :param filename: Mesh filename
            :param visual: Should we load visual components?
            :param physics: Should we load collision meshes?
            :param flags: Flags
        )EOS", py::arg("filename"), py::arg("visual")=true, py::arg("physics")=true, py::arg("flags")=sl::Mesh::Flag{})

        .def_static("load_threaded", &Mesh_loadThreaded, R"EOS(
            Load multiple meshes using a thread pool.

            :param filenames: List of file names to load
            :param visual: Should we load visual components?
            :param physics: Should we load collision meshes?
            :param flags: Flags
            :return: List of mesh instances
        )EOS", py::arg("filenames"), py::arg("visual")=true, py::arg("physics")=true, py::arg("flags")=std::vector<sl::Mesh::Flag>{})

        .def_property_readonly("bbox", &sl::Mesh::bbox, R"EOS(
            Mesh bounding box.
        )EOS")

        .def("center_bbox", &sl::Mesh::centerBBox, R"EOS(
            Center mesh.

            Modifies the pretransform such that the bounding box
            (see :ref:`bbox`) is centered at the origin.
        )EOS")

        .def("scale_to_bbox_diagonal", &Mesh_scaleToBBoxDiagonal, R"EOS(
            Rescale mesh.

            :param target_diagonal: Target diagonal
            :param mode: Scaling mode (default 'exact').
                If 'order_of_magnitude', the resulting scale factor is the
                nearest power of 10 that fits. This is useful for detecting
                the scale of arbitrary mesh files.

            Modifies the pretransform such that the bounding box diagonal (see :ref:`bbox`) is equal to `target_diagonal`.
        )EOS", py::arg("target_diagonal"), py::arg("mode")="exact")

        .def("update_positions", &Mesh_updatePositions, R"EOS(
            Updates the mesh vertex positions.

            :param vertex_indices: N (dim=1)
            :param position_update: Nx3
        )EOS", py::arg("vertex_indices"), py::arg("position_update"))

        .def("update_colors", &Mesh_updateColors, R"EOS(
            Updates the mesh vertex colors.

            :param vertex_indices: N (dim=1)
            :param color_update: Nx4
        )EOS", py::arg("vertex_indices"), py::arg("color_update"))

        .def("update_positions_and_colors", &Mesh_updatePositionsAndColors, R"EOS(
            Updates the mesh verticex positions and vertex colors.

            :param vertex_indices: N (dim=1)
            :param position_update: Nx3
            :param color_update: Nx4
        )EOS", py::arg("vertex_indices"), py::arg("position_update"), py::arg("color_update"))

        .def("set_new_positions", &Mesh_setNewPositions, R"EOS(
            Set new vertex positions.

            :param new_positions: Nx3
        )EOS", py::arg("new_positions"))

        .def("set_new_colors", &Mesh_setNewColors, R"EOS(
            Set new vertex colors.

            :param new_colors: Nx4
        )EOS", py::arg("new_colors"))

        .def_property("pretransform", wrapShared(&sl::Mesh::pretransform), wrapShared(&sl::Mesh::setPretransform), R"EOS(
            The current pretransform matrix.

            This is initialized to identity and can be set directly or through
            :ref:`center_bbox` and :ref:`scale_to_bbox_diagonal`.
        )EOS")

        .def_property("class_index",
            &sl::Mesh::classIndex, &sl::Mesh::setClassIndex, R"EOS(
            Class index for training semantic segmentation.

            Zero is usually reserved for the background class.
            Unlike :ref:`Object.instance_index`, this property is not set
            automatically.
        )EOS")

        .def_property_readonly("points", [&](const std::shared_ptr<sl::Mesh>& mesh){
            auto points = mesh->meshPoints();
            return toTorch<std::decay_t<decltype(points)>>::convert(points);
        },
        R"EOS(
            The mesh vertices as (Nx3) float tensor.
        )EOS")

        .def_property_readonly("normals", [&](const std::shared_ptr<sl::Mesh>& mesh){
            auto data = mesh->meshNormals();
            return toTorch<std::decay_t<decltype(data)>>::convert(data);
        },
        R"EOS(
            The mesh normals as (Nx3) float tensor.
        )EOS")

        .def_property_readonly("faces", [&](const std::shared_ptr<sl::Mesh>& mesh){
            auto data = mesh->meshFaces();
            return toTorch<std::decay_t<decltype(data)>>::convert(data);
        },
        R"EOS(
            The mesh faces as (N*3) int32 tensor.

            .. block-info:: Note

                Because PyTorch does not support unsigned ints, the indices
                will wrap around at :math:`2^{31}`.
        )EOS")

        .def_property_readonly("colors", [&](const std::shared_ptr<sl::Mesh>& mesh){
            auto data = mesh->meshColors();
            return toTorch<std::decay_t<decltype(data)>>::convert(data);
        },
        R"EOS(
            The mesh colors as (Nx4) float tensor.

            .. block-warning:: Multiple sub meshes

                This returns a view onto all concatenated sub-meshes.
                Some of them may use vertex coloring, some not. For the former,
                you will see the vertex colors, for the latter just zeros.
        )EOS")

        .def_property_readonly("physics_mesh_data", &Mesh_physicsMeshData,
        R"EOS(
            The convex decomposition used for physics simulation. This is a list
            of dicts, with the keys `positions` and `indices`.
        )EOS")

        .def("dump_physics_meshes", &sl::Mesh::dumpPhysicsMeshes,
        R"EOS(
            Write physics meshes in PLY format into the given directory.
        )EOS")

        .def_property_readonly("filename", &sl::Mesh::filename, "The mesh filename")
    ;

    py::class_<sl::MeshCache>(m, "MeshCache", R"EOS(
        Caches Mesh instances.

        This is mainly useful together with :ref:`Scene.deserialize`, look there
        for more information.
    )EOS")

        .def(py::init([](){ return new sl::MeshCache(sl::python::Context::instance()); }))

        .def("add", &sl::MeshCache::add, R"EOS(
            Add a list of meshes to the cache.

            :param meshes: list of :ref:`Mesh` instances
        )EOS", py::arg("meshes"))
    ;
}

}
}
}
