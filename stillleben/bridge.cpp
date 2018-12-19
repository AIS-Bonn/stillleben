// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/object.h>
#include <stillleben/scene.h>
#include <stillleben/render_pass.h>

#include <Magnum/Image.h>

static std::shared_ptr<sl::Context> g_context;
static bool g_cudaEnabled = false;
static std::string g_installPrefix;

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
static Magnum::Matrix4 torchToMagnum(const at::Tensor& tensor)
{
    auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous().t().contiguous();
    if(cpuTensor.dim() != 2 || cpuTensor.size(0) != 4 || cpuTensor.size(1) != 4)
        throw std::invalid_argument("A pose tensor must be 4x4");

    const float* data = cpuTensor.data<float>();

    Magnum::Matrix4 mat{Magnum::Math::NoInit};

    memcpy(mat.data(), data, 16*sizeof(float));

    return mat;
}

at::Tensor extract(Magnum::GL::RectangleTexture& texture, Magnum::PixelFormat format, int channels, const torch::TensorOptions& opts)
{
#if HAVE_CUDA
    if(g_cudaEnabled)
    {
        sl::CUDAMapper mapper(texture, Magnum::pixelSize(format));

        auto size = texture.imageSize();
        at::Tensor tensor = torch::empty({size.y(), size.x(), channels}, opts);
        mapper.readInto(tensor.data());

        return tensor;
    }
    else
#else
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
#endif
}

static at::Tensor readRGBATensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA8Unorm, 4, at::kByte);
}

static at::Tensor readCoordTensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).slice(2, 0, 3);
}

static at::Tensor readByteTensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::R8UI, 1, at::kByte);
}

static at::Tensor readShortTensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::R16UI, 1, at::kShort);
}


static void init()
{
    g_context = sl::Context::Create(g_installPrefix);
    if(!g_context)
        throw std::runtime_error("Could not create stillleben context");
}

static void initCUDA(unsigned int cudaIndex)
{
    g_context = sl::Context::CreateCUDA(cudaIndex, g_installPrefix);
    if(!g_context)
        throw std::runtime_error("Could not create stillleben context");

    g_cudaEnabled = true;
}

static void setInstallPrefix(const std::string& path)
{
    g_installPrefix = path;
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

// Object
static std::shared_ptr<sl::Object> Object_factory(const std::shared_ptr<sl::Mesh>& mesh)
{
    return sl::Object::instantiate(mesh);
}

static at::Tensor Object_pose(const std::shared_ptr<sl::Object>& object)
{
    return magnumToTorch(object->pose());
}

static void Object_setPose(const std::shared_ptr<sl::Object>& object, at::Tensor& tensor)
{
    object->setPose(torchToMagnum(tensor));
}

// Scene
static std::shared_ptr<sl::Scene> Scene_factory(const std::tuple<int, int>& viewportSize)
{
    return std::make_shared<sl::Scene>(g_context, sl::ViewportSize{
        std::get<0>(viewportSize),
        std::get<1>(viewportSize)
    });
}

static at::Tensor Scene_cameraPose(const std::shared_ptr<sl::Scene>& scene)
{
    return magnumToTorch(scene->cameraPose());
}

static void Scene_setCameraPose(const std::shared_ptr<sl::Scene>& scene, at::Tensor& tensor)
{
    scene->setCameraPose(torchToMagnum(tensor));
}

static std::tuple<int, int> Scene_viewport(const std::shared_ptr<sl::Scene>& scene)
{
    auto vp = scene->viewport();
    return std::make_tuple(vp.x(), vp.y());
}

// RenderPass


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "Init without CUDA support");
    m.def("init_cuda", &initCUDA, R"EOS(
        Init with CUDA support.

        Args:
            device_index (int): Index of CUDA device to use for rendering
    )EOS", py::arg("device_index") = 0);

    m.def("_set_install_prefix", &setInstallPrefix, "set Magnum install prefix");

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

    py::class_<sl::Object, std::shared_ptr<sl::Object>>(m, "Object", R"EOS(
            An instantiated mesh with associated pose and other instance
            properties.
        )EOS")

        .def(py::init(&Object_factory), R"EOS(
            Constructor

            Args:
                mesh (Mesh): Mesh to instantiate
        )EOS", py::arg("mesh"))

        .def("pose", &Object_pose, R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.

            Examples:
                >>> obj = Object(Mesh("mesh.gltf"))
                >>> obj.pose()
        )EOS")
        .def("set_pose", &Object_setPose, R"EOS(
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
    ;

    py::class_<sl::Scene, std::shared_ptr<sl::Scene>>(m, "Scene", R"EOS(
            Represents a scene with multiple objects.
        )EOS")

        .def(py::init(&Scene_factory), R"EOS(
            Constructor

            Args:
                viewport_size (int,int): Size of the rendered image (W,H)
        )EOS", py::arg("viewport_size"))

        .def("camera_pose", &Scene_cameraPose, R"EOS(
            Retrieve current camera pose (see :func:`setCameraPose`).
        )EOS")
        .def("set_camera_pose", &Scene_setCameraPose, R"EOS(
            Set the camera pose within the scene. For most applications, leaving
            this at identity is a good idea - that way your object poses are
            expressed in camera coordinates.

            Args:
                pose (tensor): 4x4 matrix transforming camera coordinates to
                    global coordinates.
        )EOS", py::arg("pose"))

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

        .def_property_readonly("viewport", &Scene_viewport, R"EOS(
            The current viewport size (W,H) as set in the constructor.
        )EOS")

        .def("min_dist_for_object_diameter", &sl::Scene::minimumDistanceForObjectDiameter, R"EOS(
            Calculates the minimum Z distance from the camera to have an object
            of diameter :attr:`diameter` fully visible in the camera frustrum.

            Args:
                diameter (float): Diameter of the object.
        )EOS", py::arg("diameter"))

        .def("add_object", &sl::Scene::addObject, R"EOS(
            Adds an object to the scene.

            Args:
                object (Object): Object to be added.
        )EOS", py::arg("object"))
    ;

    py::class_<sl::RenderPass::Result, std::shared_ptr<sl::RenderPass::Result>>(m, "RenderPassResult", R"EOS(
            Result of a :class:`RenderPass` run.
        )EOS")

        .def("rgb", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readRGBATensor(result->rgb);
            }, R"EOS(
                Read RGBA tensor.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.

                Returns:
                    tensor: (H x W x 4) byte tensor with R,G,B,A values.
            )EOS")

        .def("class_index", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readShortTensor(result->classIndex);
            }, R"EOS(
                Read class index map.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. The background index is 0.

                Returns:
                    tensor: (H x W) short tensor with class values.
            )EOS")

        .def("instance_index", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readShortTensor(result->instanceIndex);
            }, R"EOS(
                Read class index map.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. The background index is 0.

                Returns:
                    tensor: (H x W) short tensor with instance values.
            )EOS")

        .def("valid_mask", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readByteTensor(result->validMask);
            }, R"EOS(
                Read valid mask. If and only if :func:`class_index`,
                :func:`instance_index`, and :func:`coordinates` is valid at this
                particular point, the mask will be non-zero.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. Note: background pixels are
                considered valid, even though they do not have an associated
                coordinate.

                Returns:
                    tensor: (H x W) byte tensor
            )EOS")

        .def("coordinates", [](const std::shared_ptr<sl::RenderPass::Result>& result){
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

    ;

    py::class_<sl::RenderPass>(m, "RenderPass", R"EOS(
            Renders a :class:`Scene`.
        )EOS")

        .def(py::init(), "Constructor")

        .def("render", &sl::RenderPass::render, R"EOS(
            Render a scene.

            Args:
                scene (Scene): The scene to render.

            Returns:
                RenderPassResult
        )EOS", py::arg("scene"))
    ;
}
