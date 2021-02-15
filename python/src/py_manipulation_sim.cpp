// Simulate a manipulator interacting with the scene
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_manipulation_sim.h"

#include <stillleben/manipulation_sim.h>
#include <stillleben/scene.h>

#include "py_magnum.h"

namespace sl
{
namespace python
{
namespace ManipulationSim
{

void init(py::module& m)
{
    py::class_<sl::ManipulationSim>(m, "ManipulationSim", R"EOS(
            Simulate a manipulator interacting with the scene.
            
            The manipulator is driven using a spring damping system (basically Cartesian impedance control).
        )EOS")

        .def(py::init([](const std::shared_ptr<sl::Scene>& scene, const std::shared_ptr<sl::Object>& manipulator, const at::Tensor& initialPose){
                return new sl::ManipulationSim(
                    scene, manipulator,
                    magnum::fromTorch<Magnum::Matrix4>::convert(initialPose)
                );
            }), R"EOS(
            Constructor.

            :param scene: The sl.Scene instance to run on
            :param manipulator: The sl.Object to use as manipulator
            :param initial_pose: Initial pose of the manipulator (in world coordinates)
        )EOS", py::arg("scene"), py::arg("manipulator"), py::arg("initial_pose"))
        
        .def("set_spring_parameters", &sl::ManipulationSim::setSpringParameters, R"EOS(
            Set spring parameters for driving the manipulator.
            
            :param stiffness: Spring stiffness in N/m
            :param damping: Spring damping in Ns/m
            :param force_limit: Limit of the applied force in N
        )EOS", py::arg("stiffness"), py::arg("damping"), py::arg("force_limit"))
        
        .def("step", &sl::ManipulationSim::step, R"EOS(
            Simulate a single step.
            
            Take care to take a small enough dt, otherwise the simulation will get unstable.
            
            :param goal_pose: Goal pose to drive the manipulator to
            :param dt: Step length in s
        )EOS", py::arg("goal_pose"), py::arg("dt"))
    ;
}

}
}
}
