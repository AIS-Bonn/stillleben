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

}
}
}
