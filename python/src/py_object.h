// sl::Object binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_OBJECT_H
#define SL_PY_OBJECT_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace Object
{

void init(py::module& m);

}
}
}

#endif

