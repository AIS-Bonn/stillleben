// sl::ImageSaver binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_IMAGE_SAVER_H
#define SL_PY_IMAGE_SAVER_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace ImageSaver
{

void init(py::module& m);

}
}
}

#endif
