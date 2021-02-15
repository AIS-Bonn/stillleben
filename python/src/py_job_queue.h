// sl::JobQueue binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_JOB_QUEUE_H
#define SL_PY_JOB_QUEUE_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace JobQueue
{

void init(py::module& m);

}
}
}

#endif
