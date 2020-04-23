// sl::Animator binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_animator.h"
#include "py_magnum.h"

#include <stillleben/animator.h>

using namespace sl::python::magnum;

namespace sl
{
namespace python
{
namespace Animator
{

void init(py::module& m)
{
    py::class_<sl::Animator>(m, "Animator", R"EOS(
            Generates interpolated object poses.

            Typical usage
            -------------

            .. code:: python

                import stillleben as sl

                # First pose
                p1 = torch.eye(4)

                # Translated second pose
                p2 = p1.clone()
                p2[:3, 3] = torch.tensor([0.0, 1.0, 0.0])

                # Animate for 100 ticks
                anim = sl.Animator([p1, p2], 100)
                for pose in anim:
                    print(pose)
        )EOS")

        .def(py::init([](const std::vector<at::Tensor>& poses, unsigned int ticks){
            std::vector<Magnum::Matrix4> mPoses;
            for(auto& p : poses)
                mPoses.push_back(fromTorch<Magnum::Matrix4>::convert(p));
            return std::make_unique<sl::Animator>(mPoses, ticks);
        }), "Constructor", py::arg("poses"), py::arg("ticks"))

        .def("__iter__", [](py::object s) { return s; })

        .def("__next__", [](sl::Animator& s){
            if(s.currentTick() >= s.totalTicks())
                throw py::stop_iteration{};

            return toTorch<Magnum::Matrix4>::convert(s());
        }, R"EOS(
            Return next pose.
        )EOS")

        .def("__len__", [](sl::Animator& s){ return s.totalTicks(); }, R"EOS(
            Number of ticks.
        )EOS")
    ;
}

}
}
}
