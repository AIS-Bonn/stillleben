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

            If you do not have special reason not to do so, use the
            :ref:`stillleben.view` shortcut to launch a viewer.
        )EOS")
        .def(py::init([](){
            return std::make_shared<sl::Viewer>(sl::python::Context::instance());
        }), R"EOS(
            Constructor.
        )EOS")
        .def_property("scene", &sl::Viewer::scene, &sl::Viewer::setScene, R"EOS(
            Scene to be shown.
        )EOS")
        .def("run", &sl::Viewer::run, R"EOS(
            Launch the viewer.

            You should set the :ref:`scene` property first.
            This call is blocking.
        )EOS")
    ;

    m.def("view", [](const std::shared_ptr<sl::Scene>& scene){
        sl::Viewer::view(sl::python::Context::instance(), scene);
    }, R"EOS(
        Shortcut for viewing a scene interactively.

        :param scene: Scene to be shown

        Creates and launches a :ref:`Viewer` for the given scene. This call is
        blocking and returns when the viewer is closed by the user.
    )EOS", py::arg("scene"));
}

}
}
}
