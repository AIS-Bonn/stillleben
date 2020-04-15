// sl::LightMap binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_LIGHT_MAP_H
#define SL_LIGHT_MAP_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace LightMap
{

void init(py::module& m);

}
}
}

#endif

