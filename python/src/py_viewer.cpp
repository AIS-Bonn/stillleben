// sl::Viewer binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_viewer.h"
#include "py_context.h"

#include <stillleben/viewer.h>
#include <stillleben/scene.h>
#include <stillleben/context.h>

namespace sl
{
namespace python
{
namespace Viewer
{

void init(py::module& m)
{
    py::class_<sl::Viewer, std::shared_ptr<sl::Viewer>>(m, "Viewer", R"EOS(
            Interactive scene viewer.
        )EOS")
        .def(py::init([](){
            return std::make_shared<sl::Viewer>(sl::python::Context::instance());
        }), R"EOS(
            Constructor.
        )EOS")
        .def_property("scene", &sl::Viewer::scene, &sl::Viewer::setScene, R"EOS(
            Scene to be shown.
        )EOS")
        .def("run", &sl::Viewer::run)
    ;
}

}
}
}
