// sl::RenderPass binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_RENDERPASS_H
#define SL_PY_RENDERPASS_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace RenderPass
{

void init(py::module& m);

}
}
}

#endif

