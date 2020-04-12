// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/object.h>
#include <stillleben/scene.h>
#include <stillleben/render_pass.h>
#include <stillleben/debug.h>
#include <stillleben/image_loader.h>
#include <stillleben/cuda_interop.h>
#include <stillleben/animator.h>
#include <stillleben/light_map.h>
#include <stillleben/mesh_cache.h>
#include <stillleben/contrib/ctpl_stl.h>

#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Containers/ArrayView.h>

#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Magnum.h>

#include <future>
#include <memory>
#include <functional>

#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "py_context.h"
#include "py_magnum.h"

using namespace sl::python;
using namespace sl::python::magnum;

at::Tensor extract(sl::CUDATexture& texture, Magnum::PixelFormat format, int channels, const torch::TensorOptions& opts)
{
#if HAVE_CUDA
    if(sl::python::Context::cudaEnabled())
    {
        auto size = texture.imageSize();
        at::Tensor tensor = torch::empty(
            {size.y(), size.x(), channels},
            opts.device(torch::kCUDA, sl::python::Context::cudaDevice())
        );
        texture.readIntoCUDA(static_cast<uint8_t*>(tensor.data_ptr()));

        return tensor;
    }
    else
#endif
    {
        Magnum::Image2D* img = new Magnum::Image2D{format};

        texture.image(*img);

        at::Tensor tensor = torch::from_blob(img->data(),
            {img->size().y(), img->size().x(), channels},
            [=](void*){ delete img; },
            opts
        );

        return tensor;
    }
}

static at::Tensor readRGBATensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA8Unorm, 4, at::kByte);
}

static at::Tensor readXYZWTensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat);
}

static at::Tensor readCoordTensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).slice(2, 0, 3);
}

static at::Tensor readDepthTensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).select(2, 3);
}

static at::Tensor readShortTensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::R16UI, 1, at::kShort);
}

static at::Tensor readVertexIndicesTensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32UI, 4, at::kInt).slice(2, 0, 3);
}

static at::Tensor readBaryCentricCoeffsTensor(sl::CUDATexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).slice(2, 0, 3);
}

static std::shared_ptr<sl::Mesh> Mesh_factory(
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

static std::vector<std::shared_ptr<sl::Mesh>> Mesh_loadThreaded(
    const std::vector<std::string>& filenames,
    bool visual, bool physics,
    std::size_t maxPhysicsTriangles)
{
    if(!sl::python::Context::instance())
        throw std::logic_error("You need to call init() first!");

    return sl::Mesh::loadThreaded(sl::python::Context::instance(), filenames, visual, physics, maxPhysicsTriangles);
}

static void Mesh_scaleToBBoxDiagonal(const std::shared_ptr<sl::Mesh>& mesh, float diagonal, const std::string& modeStr)
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

static void Mesh_updatePositions(const std::shared_ptr<sl::Mesh>& mesh,
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

static void Mesh_updateColors(const std::shared_ptr<sl::Mesh>& mesh,
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

static void Mesh_updatePositionsAndColors(const std::shared_ptr<sl::Mesh>& mesh,
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

static void Mesh_setNewPositions(const std::shared_ptr<sl::Mesh>& mesh,
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

static void Mesh_setNewColors(const std::shared_ptr<sl::Mesh>& mesh,
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

// Scene
static std::shared_ptr<sl::Scene> Scene_factory(const std::tuple<int, int>& viewportSize)
{
    return std::make_shared<sl::Scene>(sl::python::Context::instance(), sl::ViewportSize{
        std::get<0>(viewportSize),
        std::get<1>(viewportSize)
    });
}

static std::tuple<int, int> Scene_viewport(const std::shared_ptr<sl::Scene>& scene)
{
    auto vp = scene->viewport();
    return std::make_tuple(vp.x(), vp.y());
}

// RenderPass

// Debug

static at::Tensor renderDebugImage(const std::shared_ptr<sl::Scene>& scene)
{
    auto texture = sl::renderDebugImage(*scene);
    Magnum::Image2D* img = new Magnum::Image2D{Magnum::PixelFormat::RGBA8Unorm};

    texture.image(*img);

    at::Tensor tensor = torch::from_blob(img->data(),
        {img->size().y(), img->size().x(), 4},
        [=](void*){ delete img; },
        at::kByte
    );

    return tensor;
}

PYBIND11_MODULE(libstillleben_python, m)
{
    sl::python::Context::init(m);
    sl::python::magnum::init(m);

    m.def("render_debug_image", &renderDebugImage, R"EOS(
        Render a debug image with object coordinate systems
    )EOS");

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

    py::class_<sl::Object, std::shared_ptr<sl::Object>>(m, "Object", R"EOS(
            An instantiated mesh with associated pose and other instance
            properties.
        )EOS")

        .def(py::init([](const std::shared_ptr<sl::Mesh>& mesh, const py::dict& options){
                sl::InstantiationOptions opts;
                for(const auto& entry : options)
                {
                    auto key = entry.first.cast<std::string>();
                    if(key == "color")
                        opts.color = fromTorch<Magnum::Color4>::convert(entry.second.cast<torch::Tensor>());
                    else if(key == "force_color")
                        opts.forceColor = entry.second.cast<bool>();
                    else
                        throw std::invalid_argument("Invalid key in options");
                }

                auto obj = std::make_shared<sl::Object>();
                obj->setMesh(mesh);
                obj->setInstantiationOptions(opts);

                return obj;
            }), R"EOS(
            Constructor

            Args:
                mesh (Mesh): Mesh to instantiate
                options (dict): Dictionary of options. Supported keys:
                    * color (tensor): RGBA color used if no color information is
                      present in the mesh. Defaults to white.
                    * force_color (bool): If true, the color specified in
                      `color` is used even if the mesh is colored.
        )EOS", py::arg("mesh"), py::arg("options")=py::dict())

        .def("pose", wrapShared(&sl::Object::pose), R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.

            Examples:
                >>> obj = Object(Mesh("mesh.gltf"))
                >>> obj.pose()
        )EOS")
        .def("set_pose", wrapShared(&sl::Object::setPose), R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.

            Examples:
                >>> obj = Object(Mesh("mesh.gltf"))
                >>> p = obj.pose()
                >>> p[:3,3] = torch.tensor([0, 1, 0])
        )EOS", py::arg("pose"))

        .def_property("instance_index", &sl::Object::instanceIndex, &sl::Object::setInstanceIndex, R"EOS(
            Instance index for training semantic segmentation. This is
            automatically set by :func:`Scene.addObject` but can also be
            set manually. A manual assignment always takes precedence.
        )EOS")

        .def_property_readonly("mesh", &sl::Object::mesh, R"EOS(
            The associated :class:`Mesh` instance.
        )EOS")

        .def_property("specular_color", wrapShared(&sl::Object::specularColor), wrapShared(&sl::Object::setSpecularColor), R"EOS(
            Specular color for Phong shading. This is multiplied with the
            calculated specular intensity. Set to white for a fully-specular
            object, set to black for a fully diffuse object.
        )EOS")

        .def_property("shininess", &sl::Object::shininess, &sl::Object::setShininess, R"EOS(
            Shininess parameter for Phong shading. Low values result in very
            spread out specular highlights, high values in very small, sharp
            highlights. Default value is 80 (pretty sharp).
        )EOS")

        .def_property("metalness", &sl::Object::metalness, &sl::Object::setMetalness, R"EOS(
            Metalness parameter for PBR shading (0-1).
        )EOS")

        .def_property("roughness", &sl::Object::roughness, &sl::Object::setRoughness, R"EOS(
            Roughness parameter for PBR shading (0-1).
        )EOS")


        .def_property("sticker_rotation", wrapShared(&sl::Object::stickerRotation), wrapShared(&sl::Object::setStickerRotation), R"EOS(
            Sticker rotation quaternion.
        )EOS")

        .def_property("sticker_range", wrapShared(&sl::Object::stickerRange), wrapShared(&sl::Object::setStickerRange), R"EOS(
            Sticker range.
        )EOS")

        .def_property("sticker_texture", &sl::Object::stickerTexture, &sl::Object::setStickerTexture, R"EOS(
            Sticker texture.
        )EOS")
    ;

    py::class_<sl::LightMap, std::shared_ptr<sl::LightMap>>(m, "LightMap", R"EOS(
            An .ibl light map for image-based lighting.
        )EOS")

        .def(py::init(), "Constructor")

        .def(py::init([](const std::string& path){
                return std::make_shared<sl::LightMap>(path, sl::python::Context::instance());
            }),
            R"EOS(
                Constructs and calls load().
            )EOS"
        )

        .def("load", &sl::LightMap::load, R"EOS(
            Opens an .ibl file.

            Args:
                path (str): Path to .ibl file
            Returns:
                bool: True if successful
        )EOS")
    ;

    py::class_<sl::Scene, std::shared_ptr<sl::Scene>>(m, "Scene", R"EOS(
            Represents a scene with multiple objects.
        )EOS")

        .def(py::init(&Scene_factory), R"EOS(
            Constructor

            Args:
                viewport_size (int,int): Size of the rendered image (W,H)
        )EOS", py::arg("viewport_size"))

        .def("camera_pose", wrapShared(&sl::Scene::cameraPose), R"EOS(
            Retrieve current camera pose (see :func:`setCameraPose`).
        )EOS")
        .def("set_camera_pose", wrapShared(&sl::Scene::setCameraPose), R"EOS(
            Set the camera pose within the scene. For most applications, leaving
            this at identity is a good idea - that way your object poses are
            expressed in camera coordinates.

            Args:
                pose (tensor): 4x4 matrix transforming camera coordinates to
                    global coordinates.
        )EOS", py::arg("pose"))
        .def("set_camera_look_at", wrapShared(&sl::Scene::setCameraLookAt), R"EOS(
            Sets the camera pose within the scene using lookAt parameters.

            Args:
                position (tensor): 3D position vector
                look_at (tensor): 3D lookAt vector
                up (tensor): 3D up vector (defaults to Z axis)
        )EOS", py::arg("position"), py::arg("look_at"), py::arg("up")=torch::tensor({0.0, 0.0, 1.0}))

        .def("set_camera_intrinsics", &sl::Scene::setCameraIntrinsics, R"EOS(
            Set the camera intrinsics assuming a pinhole camera with focal
            lengths :math:`f_x`, :math:`f_y`, and projection center :math:`p_x`, :math:`p_y`.

            Note: Magnum may slightly modify the resulting matrix, I have not
            checked the accuracy of this method.

            Args:
                fx (float): :math:`f_x`
                fy (float): :math:`f_y`
                cx (float): :math:`c_x`
                cy (float): :math:`c_y`
        )EOS", py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))

        .def("set_camera_projection", wrapShared(&sl::Scene::setCameraProjection), R"EOS(
            Set the camera intrinsics from a 4x4 matrix.

            Args:
                P (tensor): The matrix.
        )EOS", py::arg("P"))

        .def("projection_matrix", wrapShared(&sl::Scene::projectionMatrix), R"EOS(
                Return the currently used OpenGL projection matrix.
            )EOS")

        .def_property_readonly("viewport", &Scene_viewport, R"EOS(
            The current viewport size (W,H) as set in the constructor.
        )EOS")

        .def_property("background_image",
            &sl::Scene::backgroundImage, &sl::Scene::setBackgroundImage, R"EOS(
            The background image. If None (default), the background color
            (see `background_color`) is used.
        )EOS")

        .def_property("background_color",
            wrapShared(&sl::Scene::backgroundColor), wrapShared(&sl::Scene::setBackgroundColor), R"EOS(
            The background color (RGBA, float, range 0-1). The default is white.
        )EOS")

        .def("min_dist_for_object_diameter", &sl::Scene::minimumDistanceForObjectDiameter, R"EOS(
            Calculates the minimum Z distance from the camera to have an object
            of diameter :attr:`diameter` fully visible in the camera frustrum.

            Args:
                diameter (float): Diameter of the object.
        )EOS", py::arg("diameter"))

        .def("place_object_randomly", wrapShared(&sl::Scene::placeObjectRandomly), R"EOS(
                Generates a random pose for an object of given diameter.

                The pose obeys the following constraints (relative to the camera
                coordinate system):
                * :math:`z` is between `0.8*min_dist_for_object_diameter()` and
                `2.0*min_dist_for_object_diameter()`, and
                * :math:`x` and :math:`y` are choosen such that the object center is
                inside 80% of the camera frustrum in each axis.
            )EOS",
            py::arg("diameter"),
            py::arg("min_size_factor")=sl::pose::DEFAULT_MIN_SIZE_FACTOR
        )

        .def("camera_to_world", wrapShared(&sl::Scene::cameraToWorld), R"EOS(
                Transform a pose from camera coordinates to world coordinates.

                Args:
                    poseInCamera (tensor): 4x4 float pose
            )EOS", py::arg("poseInCamera")
        )

        .def("add_object", &sl::Scene::addObject, R"EOS(
            Adds an object to the scene.

            Args:
                object (Object): Object to be added.
        )EOS", py::arg("object"))

        .def_property_readonly("objects", &sl::Scene::objects, R"EOS(
            Contains all objects added to the scene. See add_object()

            Returns:
                list: List of sl::Object
        )EOS")

        .def("find_noncolliding_pose", [](
            const std::shared_ptr<sl::Scene>& scene,
            const std::shared_ptr<sl::Object>& object,
            const std::string& sampler, int max_iterations,
            py::kwargs kwargs){
                if(sampler == "random")
                {
                    sl::pose::RandomPositionSampler posSampler{
                        scene->projectionMatrix(),
                        object->mesh()->bbox().size().length()
                    };
                    sl::pose::RandomPoseSampler sampler{posSampler};
                    return scene->findNonCollidingPose(*object, sampler, max_iterations);
                }
                else if(sampler == "viewpoint")
                {
                    if(!kwargs.contains("viewpoint"))
                        throw std::invalid_argument{"sampler='viewpoint' needs viewpoint argument"};

                    auto viewPoint = fromTorch<Magnum::Vector3>::convert(
                        kwargs["viewpoint"].cast<at::Tensor>()
                    );

                    sl::pose::RandomPositionSampler posSampler{
                        scene->projectionMatrix(),
                        object->mesh()->bbox().size().length()
                    };
                    sl::pose::ViewPointPoseSampler sampler{posSampler};
                    sampler.setViewPoint(viewPoint);

                    return scene->findNonCollidingPose(*object, sampler, max_iterations);
                }
                else if(sampler == "view_corrected")
                {
                    if(!kwargs.contains("orientation"))
                        throw std::invalid_argument{"sampler='view_corrected' needs orientation argument"};

                    auto orientation = fromTorch<Magnum::Matrix3>::convert(
                        kwargs["orientation"].cast<at::Tensor>()
                    );

                    sl::pose::RandomPositionSampler posSampler{
                        scene->projectionMatrix(),
                        object->mesh()->bbox().size().length()
                    };
                    sl::pose::ViewCorrectedPoseSampler sampler{posSampler, orientation};

                    return scene->findNonCollidingPose(*object, sampler, max_iterations);
                }
                else
                    throw std::invalid_argument{"Unknown sampler"};
            }, R"EOS(
            Finds a non-colliding random pose for an object. The object should
            already have been added using add_object().

            Args:
                object (stillleben.object): The object to place
                sampler (str): "random" for fully random pose, "viewpoint"
                    for a pose that ensures we look from a certain viewpoint
                    onto the object, or "view_corrected" for a perspective-
                    corrected constant orientation.
                max_iterations (int): Maximum number of attempts
                viewpoint (tensor): 3D view point for "viewpoint" sampler
                orientation (tensor): 3x3 orientation matrix for
                    "view_corrected" sampler

            Returns:
                bool: True if a non-colliding pose was found.
        )EOS", py::arg("object"), py::arg("sampler") = "random", py::arg("max_iterations")=10)

        .def("resolve_collisions", &sl::Scene::resolveCollisions, R"EOS(
            Resolve collisions by forward-simulation using the physics engine.
        )EOS")

        .def_property("light_position",
            wrapShared(&sl::Scene::lightPosition),
            wrapShared(&sl::Scene::setLightPosition),
            R"EOS(
                The light position in world coordinates. This is a float tensor
                of size 3.
            )EOS"
        )

        .def_property("ambient_light",
            wrapShared(&sl::Scene::ambientLight),
            wrapShared(&sl::Scene::setAmbientLight),
            R"EOS(
                The color & intensity of the ambient light. This is a float
                tensor of size 3 (RGB, range 0-1). This color is multiplied
                with the object color / texture during rendering.
            )EOS"
        )

        .def("simulate_tabletop_scene", &sl::Scene::simulateTableTopScene, R"EOS(
            Arrange the objects as if they were standing on a supporting surface.
        )EOS", py::arg("vis_cb")=std::function<void()>{})

        .def("choose_random_light_position", &sl::Scene::chooseRandomLightPosition, R"EOS(
            Choose a random light position under the following constraints:

            * The light comes from above (negative Y direction)
            * The light never comes from behind the objects.
        )EOS")


        .def("serialize",
            [](const std::shared_ptr<sl::Scene>& scene){
                std::ostringstream ss;
                Corrade::Utility::Configuration config;

                scene->serialize(config);
                config.save(ss);

                return ss.str();
            }, R"EOS(
                Serialize the scene to a string
            )EOS")

        .def("deserialize",
            [](const std::shared_ptr<sl::Scene>& scene, const std::string& str, sl::MeshCache* cache){
                std::istringstream ss{str};
                Corrade::Utility::Configuration config{ss};

                scene->deserialize(config, cache);
            }, R"EOS(
                Deserialize the scene from a string
            )EOS", py::arg("str"), py::arg("cache")=nullptr)

        .def("load_visual", &sl::Scene::loadVisual, R"EOS(
                Load visual meshes
            )EOS")
        .def("load_physics", &sl::Scene::loadPhysics, R"EOS(
                Load physics meshes
            )EOS")


        .def_property("light_map", &sl::Scene::lightMap, &sl::Scene::setLightMap, R"EOS(
                Light map used for image-based lighting.
            )EOS")

        .def_property("background_plane_pose", wrapShared(&sl::Scene::backgroundPlanePose), wrapShared(&sl::Scene::setBackgroundPlanePose),
            "Pose of the background plane (plane normal is in +Z direction)")
        .def_property("background_plane_size", wrapShared(&sl::Scene::backgroundPlaneSize), wrapShared(&sl::Scene::setBackgroundPlaneSize),
            "Size of the background plane in local X/Y directions")
        .def_property("background_plane_texture", &sl::Scene::backgroundPlaneTexture, &sl::Scene::setBackgroundPlaneTexture,
            "Texture of the background plane")
    ;

    py::class_<sl::RenderPass::Result, ContextSharedPtr<sl::RenderPass::Result>>(m, "RenderPassResult", R"EOS(
            Result of a :class:`RenderPass` run.
        )EOS")

        .def(py::init([](){
                if(!sl::python::Context::instance())
                    throw std::logic_error("Call sl::init() first");

                return std::make_shared<sl::RenderPass::Result>(
                    sl::python::Context::cudaEnabled()
                );
            }), R"EOS(
            Constructor.
         )EOS")

        .def("rgb", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readRGBATensor(result->rgb);
            }, R"EOS(
                Read RGBA tensor.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W x 4) byte tensor with R,G,B,A values.
            )EOS")

        .def("class_index", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readShortTensor(result->classIndex);
            }, R"EOS(
                Read class index map.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. The background index is 0.

                Returns:
                    tensor: (H x W) short tensor with class values.
            )EOS")

        .def("instance_index", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readShortTensor(result->instanceIndex);
            }, R"EOS(
                Read class index map.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. The background index is 0.

                Returns:
                    tensor: (H x W) short tensor with instance values.
            )EOS")

        .def("coordinates", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readCoordTensor(result->objectCoordinates);
            }, R"EOS(
                Read object coordinates map. Each pixel specifies the XYZ
                coordinate of the point in the respective object coordinate
                system.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W x 3) float tensor with coordinates.
            )EOS")

        .def("depth", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readDepthTensor(result->objectCoordinates);
            }, R"EOS(
                Read depth map. Each pixel specifies Z depth in camera frame.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W) float tensor with depth values.
            )EOS")

        .def("coordDepth", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readXYZWTensor(result->objectCoordinates);
            }, R"EOS(
                Read combined coordinate + depth map.

                This is the concatenation of the `coordinates` and `depth`
                fields. Using this avoids a copy.

                Returns:
                    tensor: (H x W x 4) float tensor with coordinate and depth
                        values.
            )EOS")

        .def("normals", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readXYZWTensor(result->normals);
            }, R"EOS(
                Read normal map. Each pixel (XYZW) specifies the normal
                direction in the camera frame (XYZ) and, in the W component,
                the dot product with the camera direction.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W x 4) float tensor with normals.
            )EOS")

        .def("vertex_indices", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readVertexIndicesTensor(result->vertexIndex);
            }, R"EOS(
                Read vertex indices map.
                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W x 3) int tensor with vertex indices.
            )EOS")

        .def("barycentric_coeffs", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readBaryCentricCoeffsTensor(result->barycentricCoeffs);
            }, R"EOS(
                Read barycentric coefficients map.
                    tensor: (H x W x 3) float tensor with barycentric coefficients.
            )EOS")

        .def("cam_coordinates", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readXYZWTensor(result->camCoordinates);
            }, R"EOS(
                Read dense coordinate map. Each pixel contains the coordinates
                of the 3D point in camera space as 4D homogenous coordinates.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W x 4) float tensor (x, y, z, 1)
            )EOS")
    ;

    py::class_<sl::RenderPass, ContextSharedPtr<sl::RenderPass>>(m, "RenderPass", R"EOS(
            Renders a :class:`Scene`.
        )EOS")

        .def(py::init([](const std::string& shading){
                if(shading == "phong")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::Phong, sl::python::Context::cudaEnabled())
                    );
                else if(shading == "flat")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::Flat, sl::python::Context::cudaEnabled())
                    );
                else
                    throw std::invalid_argument("unknown shading type specified");
            }), R"EOS(
            Constructor.

            Args:
                shading (str): Shading type ("phong" or "flat"). Defaults to
                    Phong shading.
         )EOS", py::arg("shading")="phong")

        .def("render",
            [](const ContextSharedPtr<sl::RenderPass>& pass,
                const std::shared_ptr<sl::Scene>& scene,
                std::shared_ptr<sl::RenderPass::Result> result,
                std::shared_ptr<sl::RenderPass::Result> depthBufferResult){
                return ContextSharedPtr<sl::RenderPass::Result>{pass->render(*scene, result, depthBufferResult.get())};
            }, R"EOS(
            Render a scene.

            Args:
                scene (Scene): The scene to render.
                result (RenderPassResult): The caller can pass in a result
                    instance to be filled. If this is None, the internal result
                    instance of the RenderPass object will be used.

                    NOTE: A second render with `result=None` will overwrite
                    the results of the first render.
                depth_peel (RenderPassResult): If you want to retrieve the
                    layer behind the last rendered one, pass in the result of
                    the previous render here (depth peeling).
            Returns:
                RenderPassResult
        )EOS", py::arg("scene"), py::arg("result")=nullptr, py::arg("depth_peel")=nullptr)

        .def_property("ssao_enabled", &sl::RenderPass::ssaoEnabled, &sl::RenderPass::setSSAOEnabled, "SSAO enable")
    ;

    py::class_<sl::Animator>(m, "Animator", R"EOS(
            Generates interpolated object poses.
        )EOS")

        .def(py::init([](const std::vector<at::Tensor>& poses, unsigned int ticks){
            std::vector<Magnum::Matrix4> mPoses;
            for(auto& p : poses)
                mPoses.push_back(fromTorch<Magnum::Matrix4>::convert(p));
            return std::make_unique<sl::Animator>(mPoses, ticks);
        }), "Constructor", py::arg("poses"), py::arg("ticks"))

        .def("__iter__", [](py::object s) { return s; })

        .def("__next__", [](sl::Animator& s){
            if(s.currentTick() >= s.totalTicks())
                throw py::stop_iteration{};

            return toTorch<Magnum::Matrix4>::convert(s());
        })

        .def("__len__", [](sl::Animator& s){ return s.totalTicks(); })
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

    py::class_<sl::ImageLoader>(m, "ImageLoader", R"EOS(
            Multi-threaded image loader.
        )EOS")

        .def(py::init([](const std::string& path){
                return new sl::ImageLoader(path, sl::python::Context::instance());
            }), R"EOS(
            Constructor.

            Args:
                path: Path to the image directory
            )EOS", py::arg("path")
        )

        .def("next", &sl::ImageLoader::nextRectangleTexture, R"EOS(
            Return next image (randomly sampled). This is the same as nextRectangleTexture().
        )EOS")

        .def("next_texture2d", &sl::ImageLoader::nextTexture2D, R"EOS(
            Return next image (randomly sampled) as 2D texture
        )EOS")

        .def("next_rectangle_texture", &sl::ImageLoader::nextRectangleTexture, R"EOS(
            Return next image (randomly sampled) as rectangle texture
        )EOS")
    ;
}
