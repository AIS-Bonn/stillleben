// sl::LightMap binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_light_map.h"
#include "py_context.h"

#include <stillleben/light_map.h>
#include <stillleben/context.h>

namespace sl
{
namespace python
{
namespace LightMap
{

void init(py::module& m)
{
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

            :param path: Path to .ibl file
            :return: True if successful
        )EOS")
    ;
}

}
}
}
