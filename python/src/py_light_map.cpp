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

            The file is expected to be in `sIBL format`_.
            For a wonderful repository of light maps, see the `sIBL Archive`_.

            The light map can be used during rendering by setting
            :ref:`Scene.light_map`.

            For an example, see :ref:`std:doc:examples/pbr`.

            .. _`sIBL format`: http://www.hdrlabs.com/sibl/index.html
            .. _`sIBL Archive`: http://www.hdrlabs.com/sibl/archive.html
        )EOS")

        .def(py::init(), "Constructor")

        .def(py::init([](const py::object& path){
                return std::make_shared<sl::LightMap>(py::str(path), sl::python::Context::instance());
            }),
            R"EOS(
                Constructs and calls load().
            )EOS",
            py::arg("path")
        )

        .def("load", [](sl::LightMap& map, const py::object& path){
            return map.load(py::str(path), sl::python::Context::instance());
        }, R"EOS(
            Opens an .ibl file.

            :param path: Path to .ibl file
            :return: True if successful
        )EOS", py::arg("path"))
    ;
}

}
}
}
