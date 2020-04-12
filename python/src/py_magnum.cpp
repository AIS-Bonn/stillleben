// Basic Magnum types binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_magnum.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>

namespace sl
{
namespace python
{
namespace magnum
{

void init(py::module& m)
{
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

            return toTorch<Magnum::Matrix3>::convert(magnumQ.normalized().toMatrix());
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
            if(!sl::python::Context::instance())
                throw std::logic_error("Create a context object before");

            return std::make_shared<Magnum::GL::RectangleTexture>(
                sl::python::Context::instance()->loadTexture(path)
            );
        }), R"EOS(
            Load the texture from the specified path.
        )EOS", py::arg("path"))

        .def(py::init([](torch::Tensor tensor){
            if(!sl::python::Context::instance())
                throw std::logic_error("Create a context object before");

            if(tensor.dim() != 3 || tensor.size(2) != 3 || tensor.scalar_type() != torch::kByte || tensor.device().type() != torch::kCPU)
                throw std::invalid_argument("Input tensor should be a HxWx3 CPU byte tensor");

            tensor = tensor.contiguous();

            Magnum::ImageView2D image{
                Magnum::PixelStorage{}.setAlignment(1),
                Magnum::PixelFormat::RGB8Unorm,
                {static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(0))},
                Corrade::Containers::ArrayView<uint8_t>(tensor.data_ptr<uint8_t>(), tensor.numel())
            };

            Magnum::GL::RectangleTexture texture;
            texture.setStorage(Magnum::GL::TextureFormat::RGB8, image.size());
            texture.setSubImage({}, image);

            return texture;
        }), R"EOS(
            Load an RGB texture from the specified HxWx3 CPU byte tensor.
        )EOS", py::arg("tensor"))
    ;

    py::class_<Magnum::GL::Texture2D, std::shared_ptr<Magnum::GL::Texture2D>>(
        m, "Texture2D", R"EOS(
            An RGBA texture.
        )EOS")

        .def(py::init([](const std::string& path){
            if(!sl::python::Context::instance())
                throw std::logic_error("Create a context object before");

            return std::make_shared<Magnum::GL::Texture2D>(
                sl::python::Context::instance()->loadTexture2D(path)
            );
        }), R"EOS(
            Load the texture from the specified path.
        )EOS", py::arg("path"))

        .def(py::init([](torch::Tensor tensor){
            if(!sl::python::Context::instance())
                throw std::logic_error("Create a context object before");

            if(tensor.dim() != 3 || tensor.size(2) != 3 || tensor.scalar_type() != torch::kByte || tensor.device().type() != torch::kCPU)
                throw std::invalid_argument("Input tensor should be a HxWx3 CPU byte tensor");

            tensor = tensor.contiguous();

            Magnum::ImageView2D image{
                Magnum::PixelStorage{}.setAlignment(1),
                Magnum::PixelFormat::RGB8Unorm,
                {static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(0))},
                Corrade::Containers::ArrayView<uint8_t>(tensor.data_ptr<uint8_t>(), tensor.numel())
            };

            Magnum::GL::Texture2D texture;
            texture.setStorage(Magnum::Math::log2(image.size().max())+1, Magnum::GL::TextureFormat::RGB8, image.size());
            texture.setSubImage(0, {}, image);
            texture.generateMipmap();

            return texture;
        }), R"EOS(
            Load an RGB texture from the specified HxWx3 CPU byte tensor.
        )EOS", py::arg("tensor"))
    ;
}

}
}
}
