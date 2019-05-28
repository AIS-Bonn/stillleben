// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/object.h>
#include <stillleben/scene.h>
#include <stillleben/render_pass.h>
#include <stillleben/debug.h>
#include <stillleben/cuda_interop.h>
#include <stillleben/animator.h>
#include <stillleben/mesh_cache.h>
#include <stillleben/contrib/ctpl_stl.h>

#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/Debug.h>

#include <Magnum/Image.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>

#include <future>
#include <memory>
#include <functional>

#include <pybind11/stl.h>
#include <pybind11/functional.h>

static std::shared_ptr<sl::Context> g_context;
static bool g_cudaEnabled = false;
static unsigned int g_cudaIndex = 0;
static std::string g_installPrefix;

// shared pointer with reference to the context
template<class T>
class ContextSharedPtr : public std::shared_ptr<T>
{
public:
    ContextSharedPtr()
    {
        check();
    }

    explicit ContextSharedPtr(T* ptr)
     : std::shared_ptr<T>(ptr)
    {
        check();
    }

    ContextSharedPtr(const std::shared_ptr<T>& ptr)
     : std::shared_ptr<T>(ptr)
    {
        check();
    }

    ~ContextSharedPtr()
    {
        // Delete our pointee before we release the context.
        this->reset();
    }

    void check()
    {
        if(!g_context)
            throw std::logic_error("Call sl::init() first");

        m_context = g_context;
    }

private:
    std::shared_ptr<sl::Context> m_context;
};

PYBIND11_DECLARE_HOLDER_TYPE(T, ContextSharedPtr<T>);

// Conversion functions
namespace
{
    // Magnum -> Torch
    template<class T>
    struct toTorch
    {
        using Result = T;
        static T convert(const T& t)
        { return t; }
    };

    template<>
    struct toTorch<void>
    {
        using Result = void;
    };

    template<>
    struct toTorch<Magnum::Matrix4>
    {
        using Result = at::Tensor;
        static at::Tensor convert(const Magnum::Matrix4& mat)
        {
            auto tensor = torch::from_blob(
                const_cast<float*>(mat.data()),
                {4,4},
                at::kFloat
            );

            // NOTE: Magnum matrices are column-major
            return tensor.t().clone();
        }
    };

    template<>
    struct toTorch<Magnum::Matrix3>
    {
        using Result = at::Tensor;
        static at::Tensor convert(const Magnum::Matrix3& mat)
        {
            auto tensor = torch::from_blob(
                const_cast<float*>(mat.data()),
                {3,3},
                at::kFloat
            );

            // NOTE: Magnum matrices are column-major
            return tensor.t().clone();
        }
    };

    template<>
    struct toTorch<Magnum::Vector3>
    {
        using Result = at::Tensor;
        static at::Tensor convert(const Magnum::Vector3& vec)
        {
            return torch::from_blob(const_cast<float*>(vec.data()), {3}, at::kFloat).clone();
        }
    };

    template<>
    struct toTorch<Magnum::Vector4>
    {
        using Result = at::Tensor;
        static at::Tensor convert(const Magnum::Vector4& vec)
        {
            return torch::from_blob(const_cast<float*>(vec.data()), {4}, at::kFloat).clone();
        }
    };

    template<>
    struct toTorch<Magnum::Color4>
    {
        using Result = at::Tensor;
        static at::Tensor convert(const Magnum::Color4& vec)
        {
            return toTorch<Magnum::Vector4>::convert(vec);
        }
    };

    // Torch -> Magnum
    template<class T>
    struct fromTorch
    {
        using Type = T;
        static T convert(const T& t)
        { return t; }
    };

    template<>
    struct fromTorch<Magnum::Matrix4>
    {
        using Type = at::Tensor;
        static Magnum::Matrix4 convert(const at::Tensor& tensor)
        {
            auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous().t().contiguous();
            if(cpuTensor.dim() != 2 || cpuTensor.size(0) != 4 || cpuTensor.size(1) != 4)
                throw std::invalid_argument("A pose tensor must be 4x4");

            const float* data = cpuTensor.data<float>();

            Magnum::Matrix4 mat{Magnum::Math::NoInit};

            memcpy(mat.data(), data, 16*sizeof(float));

            return mat;
        }
    };

    template<>
    struct fromTorch<Magnum::Matrix3>
    {
        using Type = at::Tensor;
        static Magnum::Matrix3 convert(const at::Tensor& tensor)
        {
            auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous().t().contiguous();
            if(cpuTensor.dim() != 2 || cpuTensor.size(0) != 3 || cpuTensor.size(1) != 3)
                throw std::invalid_argument("An orientation tensor must be 3x3");

            const float* data = cpuTensor.data<float>();

            Magnum::Matrix3 mat{Magnum::Math::NoInit};

            memcpy(mat.data(), data, 9*sizeof(float));

            return mat;
        }
    };

    template<>
    struct fromTorch<Magnum::Vector3>
    {
        using Type = at::Tensor;
        static Magnum::Vector3 convert(const at::Tensor& tensor)
        {
            auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous();
            if(cpuTensor.dim() != 1 || cpuTensor.size(0) != 3)
                throw std::invalid_argument("A vector tensor must have size 3");

            const float* data = cpuTensor.data<float>();
            Magnum::Vector3 vec{Magnum::Math::NoInit};
            memcpy(vec.data(), data, 3*sizeof(float));

            return vec;
        }
    };

    template<>
    struct fromTorch<Magnum::Vector4>
    {
        using Type = at::Tensor;
        static Magnum::Vector4 convert(const at::Tensor& tensor)
        {
            auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous();
            if(cpuTensor.dim() != 1 || cpuTensor.size(0) != 4)
                throw std::invalid_argument("A vector4 tensor must have size 4");

            const float* data = cpuTensor.data<float>();
            Magnum::Vector4 vec{Magnum::Math::NoInit};
            memcpy(vec.data(), data, 4*sizeof(float));

            return vec;
        }
    };

    template<>
    struct fromTorch<Magnum::Color4>
    {
        using Type = at::Tensor;
        static Magnum::Color4 convert(const at::Tensor& tensor)
        {
            return fromTorch<Magnum::Vector4>::convert(tensor);
        }
    };

    // Automatic wrapping
    template<class T, class R, class ... Args>
    std::function<typename toTorch<std::decay_t<R>>::Result (const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapShared(R (T::*fun)(Args...) const)
    {
        using RConv = toTorch<std::decay_t<R>>;
        return [=](const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
            return RConv::convert((obj.get()->*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
        };
    }
    template<class T, class R, class ... Args>
    std::function<typename toTorch<std::decay_t<R>>::Result (const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapShared(R (T::*fun)(Args...))
    {
        using RConv = toTorch<std::decay_t<R>>;
        return [=](const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
            if constexpr(std::is_same_v<R, void>)
                (obj.get()->*fun)(fromTorch<std::decay_t<Args>>::convert(args)...);
            else
                return RConv::convert((obj.get()->*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
        };
    }

    template<class T, class R, class ... Args>
    std::function<typename toTorch<std::decay_t<R>>::Result (T& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapRef(R (T::*fun)(Args...) const)
    {
        using RConv = toTorch<std::decay_t<R>>;
        return [=](T& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
            return RConv::convert((obj.*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
        };
    }
    template<class T, class R, class ... Args>
    std::function<typename toTorch<std::decay_t<R>>::Result (T& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapRef(R (T::*fun)(Args...))
    {
        using RConv = toTorch<std::decay_t<R>>;
        return [=](T& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
            if constexpr(std::is_same_v<R, void>)
                (obj.*fun)(fromTorch<std::decay_t<Args>>::convert(args)...);
            else
                return RConv::convert((obj.*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
        };
    }
}

at::Tensor extract(Magnum::GL::RectangleTexture& texture, Magnum::PixelFormat format, int channels, const torch::TensorOptions& opts)
{
#if HAVE_CUDA
    if(g_cudaEnabled)
    {
        sl::CUDAMapper mapper(texture, Magnum::pixelSize(format));

        auto size = texture.imageSize();
        at::Tensor tensor = torch::empty(
            {size.y(), size.x(), channels},
            opts.device(torch::kCUDA, g_cudaIndex)
        );
        mapper.readInto(static_cast<uint8_t*>(tensor.data_ptr()));

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

static at::Tensor readRGBATensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA8Unorm, 4, at::kByte);
}

static at::Tensor readXYZWTensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat);
}

static at::Tensor readCoordTensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).slice(2, 0, 3);
}

static at::Tensor readDepthTensor(Magnum::GL::RectangleTexture& texture)
{
    return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).select(2, 3);
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

static void initCUDA(unsigned int cudaIndex, bool useCUDA=true)
{
    g_context = sl::Context::CreateCUDA(cudaIndex, g_installPrefix);
    if(!g_context)
        throw std::runtime_error("Could not create stillleben context");

    g_cudaIndex = cudaIndex;
    g_cudaEnabled = useCUDA;
}

static void setInstallPrefix(const std::string& path)
{
    g_installPrefix = path;
}

static std::shared_ptr<sl::Mesh> Mesh_factory(
    const std::string& filename,
    std::size_t maxPhysicsTriangles)
{
    if(!g_context)
        throw std::logic_error("You need to call init() first!");

    auto mesh = std::make_shared<sl::Mesh>(filename, g_context);
    mesh->load(maxPhysicsTriangles);

    return mesh;
}

static std::vector<std::shared_ptr<sl::Mesh>> Mesh_loadThreaded(
    const std::vector<std::string>& filenames,
    bool visual, bool physics,
    std::size_t maxPhysicsTriangles)
{
    if(!g_context)
        throw std::logic_error("You need to call init() first!");

    return sl::Mesh::loadThreaded(g_context, filenames, visual, physics, maxPhysicsTriangles);
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

// Scene
static std::shared_ptr<sl::Scene> Scene_factory(const std::tuple<int, int>& viewportSize)
{
    return std::make_shared<sl::Scene>(g_context, sl::ViewportSize{
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
    return readRGBATensor(texture);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "Init without CUDA support");
    m.def("init_cuda", &initCUDA, R"EOS(
        Init with CUDA support.

        Args:
            device_index (int): Index of CUDA device to use for rendering
            use_cuda (bool): If false, return results on CPU
    )EOS", py::arg("device_index") = 0, py::arg("use_cuda")=true);

    m.def("_set_install_prefix", &setInstallPrefix, "set Magnum install prefix");

    m.def("render_debug_image", &renderDebugImage, R"EOS(
        Render a debug image with object coordinate systems
    )EOS");

    // Basic geometric types
    py::class_<Magnum::Range3D>(m, "Range3D", R"EOS(
            An axis-aligned 3D range (bounding box).
        )EOS")

        .def_property("min",
            [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.min()); },
            [](Magnum::Range3D& range, at::Tensor min){ range.min() = fromTorch<Magnum::Vector3>::convert(min); }
        )
        .def_property("max",
            [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.max()); },
            [](Magnum::Range3D& range, at::Tensor max){ range.max() = fromTorch<Magnum::Vector3>::convert(max); }
        )
        .def_property_readonly("center",
            [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.center()); }
        )
        .def_property_readonly("size",
            [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.size()); }
        )
        .def_property_readonly("diagonal",
            [](const Magnum::Range3D& range){ return range.size().length(); }
        )

        .def("__repr__", [](const Magnum::Range3D& range){
            using Corrade::Utility::Debug;
            std::ostringstream ss;
            Debug{&ss, Debug::Flag::NoNewlineAtTheEnd}
                << "Range3D(" << range.min() << "," << range.max() << ")";
            return ss.str();
        })
    ;

    // Quaternion <-> Matrix
    m.def("quat_to_matrix",
        [](torch::Tensor& quat){
            if(quat.dim() != 1 || quat.size(0) != 4)
                throw std::invalid_argument{"Quaternion tensor should be one-dimensional tensor of size 4"};

            auto tmp = quat.cpu().contiguous();
            auto quat_view = tmp.accessor<float,1>();
            Magnum::Quaternion magnumQ{{quat_view[0], quat_view[1], quat_view[2]}, quat_view[3]};

            return toTorch<Magnum::Matrix3>::convert(magnumQ.toMatrix());
        },
        R"EOS(
            Convert a quaternion into a 3x3 rotation matrix.

            Args:
                quat (tensor): Should be a size 4 tensor with elements [x y z w]
            Returns:
                tensor: 3x3 rotation matrix
        )EOS");
    m.def("matrix_to_quat",
        [](torch::Tensor& matrix){
            auto magnumMatrix = fromTorch<Magnum::Matrix3>::convert(matrix);

            auto q = Magnum::Quaternion::fromMatrix(magnumMatrix);

            return toTorch<Magnum::Vector4>::convert({q.vector(), q.scalar()});
        },
        R"EOS(
            Convert a 3x3 rotation matrix into a quaternion.

            Args:
                matrix (tensor): 3x3 rotation matrix
            Returns:
                tensor: Quaternion [x y z w]
        )EOS");

    py::class_<Magnum::GL::RectangleTexture, std::shared_ptr<Magnum::GL::RectangleTexture>>(
        m, "Texture", R"EOS(
            An RGBA texture.
        )EOS")

        .def(py::init([](const std::string& path){
            if(!g_context)
                throw std::logic_error("Create a context object before");

            return std::make_shared<Magnum::GL::RectangleTexture>(
                g_context->loadTexture(path)
            );
        }), R"EOS(
            Load the texture from the specified path.
        )EOS", py::arg("path"))

        .def(py::init([](torch::Tensor tensor){
            if(!g_context)
                throw std::logic_error("Create a context object before");

            if(tensor.dim() != 3 || tensor.size(2) != 3 || tensor.scalar_type() != torch::kByte || tensor.device().type() != torch::kCPU)
                throw std::invalid_argument("Input tensor should be a HxWx3 CPU byte tensor");

            tensor = tensor.contiguous();

            Magnum::ImageView2D image{
                Magnum::PixelStorage{}.setAlignment(1),
                Magnum::PixelFormat::RGB8Unorm,
                {static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(0))},
                Corrade::Containers::ArrayView<uint8_t>(tensor.data<uint8_t>(), tensor.numel())
            };

            Magnum::GL::RectangleTexture texture;
            texture.setStorage(Magnum::GL::TextureFormat::RGB8, image.size());
            texture.setSubImage({}, image);

            return texture;
        }), R"EOS(
            Load an RGB texture from the specified HxWx3 CPU byte tensor.
        )EOS", py::arg("tensor"))
    ;

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

            Examples:
                >>> m = Mesh("path/to/my/mesh.gltf")
        )EOS", py::arg("filename"), py::arg("max_physics_triangles")=sl::Mesh::DefaultPhysicsTriangles)

        .def_static("load_threaded", &Mesh_loadThreaded, R"EOS(
            Load multiple meshes using a thread pool.

            Args:
                filenames (list): List of file names to load
                visual (bool): Should we load visual componencts?
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

        .def_property_readonly("pretransform", wrapShared(&sl::Mesh::pretransform), R"EOS(
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
    ;

    py::class_<sl::RenderPass::Result, ContextSharedPtr<sl::RenderPass::Result>>(m, "RenderPassResult", R"EOS(
            Result of a :class:`RenderPass` run.
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

        .def("valid_mask", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
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

    ;

    py::class_<sl::RenderPass, ContextSharedPtr<sl::RenderPass>>(m, "RenderPass", R"EOS(
            Renders a :class:`Scene`.
        )EOS")

        .def(py::init([](const std::string& shading){
                if(shading == "phong")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::Phong)
                    );
                else if(shading == "flat")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::Flat)
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
            [](const ContextSharedPtr<sl::RenderPass>& pass, const std::shared_ptr<sl::Scene>& scene){
                return ContextSharedPtr<sl::RenderPass::Result>{pass->render(*scene)};
            }, R"EOS(
            Render a scene.

            Args:
                scene (Scene): The scene to render.

            Returns:
                RenderPassResult
        )EOS", py::arg("scene"))
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

        .def(py::init([](){ return new sl::MeshCache(g_context); }))
    ;

    // We need to release our context pointer when the python module is
    // unloaded. Otherwise, we release it very late (basically, when the
    // atexit handlers are called. MESA also has atexit handlers, and if they
    // get called before our cleanup code, bad things happen.
    auto cleanup_callback = []() {
        g_context.reset();
    };

    m.add_object("_cleanup", py::capsule(cleanup_callback));
}
