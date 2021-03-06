// sl::ImageLoader binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_image_loader.h"
#include "py_context.h"

#include <stillleben/image_loader.h>

namespace sl
{
namespace python
{
namespace ImageLoader
{

void init(py::module& m)
{
    py::class_<sl::ImageLoader>(m, "ImageLoader", R"EOS(
            Multi-threaded image loader.

            This class can be used to quickly load images from a specified
            directory - in random order. Note: the directory should only
            contain images.

            It currently supports all image file formats supported by
            Magnum's `AnyImageImporter`_.

            .. _AnyImageImporter: https://doc.magnum.graphics/magnum/classMagnum_1_1Trade_1_1AnyImageImporter.html
        )EOS")

        .def(py::init([](const py::object& path){
                return new sl::ImageLoader(py::str(path), sl::python::Context::instance());
            }), R"EOS(
            Constructor.

            :param path: Path to the image directory
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

}
}
}
