// Simulate a manipulator interacting with the scene
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_MANIPULATION_SIM_H
#define SL_PY_MANIPULATION_SIM_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace ManipulationSim
{

void init(py::module& m);

}
}
}

#endif
