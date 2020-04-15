// sl::Animator binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_ANIMATOR_H
#define SL_ANIMATOR_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace Animator
{

void init(py::module& m);

}
}
}

#endif


