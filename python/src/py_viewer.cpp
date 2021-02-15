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
        .def(py::init<const std::shared_ptr<sl::Scene>&>(), R"EOS(
            Constructor.

            :param scene: Scene to view
        )EOS", py::arg("scene"))
        .def_property_readonly("scene", &sl::Viewer::scene, R"EOS(
            Scene to be shown.
        )EOS")
        .def("run", &sl::Viewer::run, R"EOS(
            Draw until the user closes the window.

            This call is blocking.
        )EOS")

        .def("draw_frame", &sl::Viewer::drawFrame, R"EOS(
            Draw a single frame.

            In most cases, you should call :ref:`run()` instead, which is blocking
            and easier to use.
        )EOS")
    ;

    m.def("view", [](const std::shared_ptr<sl::Scene>& scene){
        sl::Viewer::view(scene);
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
