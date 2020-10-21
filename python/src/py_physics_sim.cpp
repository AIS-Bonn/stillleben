// sl::PhysicsSim binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_physics_sim.h"

#include <stillleben/physics_sim.h>
#include <stillleben/scene.h>

namespace sl
{
namespace python
{
namespace PhysicsSim
{

void init(py::module& m)
{
    py::class_<sl::PhysicsSim>(m, "PhysicsSim", R"EOS(
            Multi-threaded physics simulator.

            This class can be used to simulate multiple scenes in parallel.
        )EOS")

        .def(py::init([](int numThreads = -1){
                return new sl::PhysicsSim(numThreads);
            }), R"EOS(
            Constructor.

            :param num_threads: Number of parallel simulations to run. -1
                indicates that an adaptive number shall be used (currently
                the number of CPU cores divided by 2).
        )EOS", py::arg("num_threads")=-1)

        .def("add_scene", &sl::PhysicsSim::addScene, R"EOS(
            Add a scene to be simulated.

            :param scene: The scene. Take care not to access the scene until
                it has been retrieved using :ref:`retrieve_scene()`.
        )EOS", py::arg("scene"))

        .def("retrieve_scene", &sl::PhysicsSim::retrieveScene, R"EOS(
            Wait until a simulated scene is available and return it.
        )EOS")

        .def_property_readonly("num_threads", &sl::PhysicsSim::numThreads, R"EOS(
            The number of threads used for physics simulation.
        )EOS")
    ;
}

}
}
}

