// sl::Mesh binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_MESH_H
#define SL_PY_MESH_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace Mesh
{

void init(py::module& m);

}
}
}

#endif
