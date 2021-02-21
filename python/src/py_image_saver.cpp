// sl::ImageSaver binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_image_saver.h"
#include "py_context.h"

#include <Corrade/Containers/Pointer.h>

#include <torch/extension.h>

#include <stillleben/image_saver.h>

using namespace Magnum;

namespace sl
{
namespace python
{
namespace ImageSaver
{

class ImageSaverWrapper
{
public:
    ImageSaverWrapper() = default;

    ImageSaverWrapper(const ImageSaverWrapper&) = delete;
    ImageSaverWrapper& operator=(const ImageSaverWrapper&) = delete;

    ImageSaverWrapper& enter()
    {
        m_imageSaver.emplace(sl::python::Context::instance());

        return *this;
    }

    void save(at::Tensor& tensor, const std::string& path)
    {
        if(!m_imageSaver)
            throw std::logic_error{"Call __enter__() first"};

        sl::ImageSaver::Job job;
        job.path = path;
        job.deleter = [tensor](){};

        if(!tensor.device().is_cpu())
            throw std::invalid_argument{"image needs to be on CPU"};

        if(!tensor.is_contiguous())
            throw std::invalid_argument{"image needs to be contiguous"};

        if(tensor.dim() == 3)
        {
            if(tensor.size(2) != 3)
                throw std::invalid_argument{"Color images need to have shape HxWx3"};

            int H = tensor.size(0);
            int W = tensor.size(1);

            if(tensor.scalar_type() == at::kByte)
            {
                job.image = ImageView2D{PixelFormat::RGB8Unorm,
                    {W, H},
                    Containers::arrayCast<char>(Containers::arrayView(tensor.data_ptr<uint8_t>(), tensor.numel()))
                };
            }
            else
                throw std::invalid_argument{"Color images need to have type uint8"};
        }
        else if(tensor.dim() == 2)
        {
            int H = tensor.size(0);
            int W = tensor.size(1);

            if(tensor.scalar_type() == at::kByte)
            {
                job.image = ImageView2D{PixelFormat::R8Unorm,
                    {W, H},
                    Containers::arrayCast<char>(Containers::arrayView(tensor.data_ptr<uint8_t>(), tensor.numel()))
                };
            }
            else if(tensor.scalar_type() == at::kShort)
            {
                job.image = ImageView2D{PixelFormat::R16Unorm,
                    {W, H},
                    Containers::arrayCast<char>(Containers::arrayView(tensor.data_ptr<int16_t>(), tensor.numel()))
                };
            }
            else
                throw std::invalid_argument{"Grayscale images need to be byte or short type"};
        }

        m_imageSaver->save(std::move(job));
    }

    void exit(py::object&, py::object&, py::object&)
    {
        m_imageSaver.reset();
    }

private:
    Containers::Pointer<sl::ImageSaver> m_imageSaver;
};

void init(py::module& m)
{
    py::class_<ImageSaverWrapper>(m, "ImageSaver", R"EOS(
        Multi-threaded image writer.

        Typical usage
        -------------

        .. code:: python

            import stillleben as sl

            sl.init()

            with sl.ImageSaver() as saver:
                saver.save(torch.zeros(640,480,3), '/tmp/test.png')
    )EOS")

        .def(py::init(), R"EOS(
            Constructor.
        )EOS")

        .def("__enter__", &ImageSaverWrapper::enter)
        .def("__exit__", &ImageSaverWrapper::exit)
        .def("save", &ImageSaverWrapper::save, R"EOS(
            Save image.

            Note: This call is asynchronous. You can only assume the image
            has been written once :ref:`__exit__()` has been called.

            :param image: Image tensor. Needs to have uint8 type and shape HxWx3 or HxW (grayscale). Grayscale images can also be uint16.
        )EOS", py::arg("image"), py::arg("path"))
    ;
}

}
}
}
