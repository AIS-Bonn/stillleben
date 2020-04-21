// sl::Viewer binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_VIEWER_H
#define SL_PY_VIEWER_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace Viewer
{

void init(py::module& m);

}
}
}

#endif

