// Base exception class
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/exception.h>

namespace sl
{

Exception::Exception(const char* msg)
 : std::runtime_error(msg)
{}

}
