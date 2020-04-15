// sl::ImageLoader binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_IMAGE_LOADER_H
#define SL_IMAGE_LOADER_H

#include <torch/extension.h>

namespace sl
{
namespace python
{
namespace ImageLoader
{

void init(py::module& m);

}
}
}

#endif
