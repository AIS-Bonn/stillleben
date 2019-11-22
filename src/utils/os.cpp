// Operating system support
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "os.h"

// std::filesystem is not fully supported by gcc on Ubuntu 18.04, so we
// can't use it here...

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <Corrade/Containers/Array.h>
#include <Corrade/Utility/Format.h>

namespace sl
{
namespace os
{

FileTime modificationTime(const std::string_view& filename)
{
    struct stat data;
    if(::stat(filename.data(), &data) != 0)
    {
        throw std::runtime_error{Corrade::Utility::format(
            "Could not get file modification date of '{}': {}",
            filename.data(), strerror(errno)
        )};
    }

    return FileTime{
        std::chrono::seconds{data.st_mtim.tv_sec} + std::chrono::nanoseconds{data.st_mtim.tv_nsec}
    };
}

}
}
