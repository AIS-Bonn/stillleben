// Operating system support
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_OS_H
#define STILLLEBEN_OS_H

#include <chrono>
#include <string>

namespace sl
{
namespace os
{

using FileTime = std::chrono::time_point<std::chrono::system_clock>;

FileTime modificationTime(const std::string_view& filename);

}
}

#endif
