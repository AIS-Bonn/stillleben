// sl::PhysicsSim binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_PHYSICS_SIM_H
#define SL_PY_PHYSICS_SIM_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace PhysicsSim
{

void init(py::module& m);

}
}
}

#endif
