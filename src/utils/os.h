// Operating system support
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_OS_H
#define STILLLEBEN_OS_H

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

namespace sl
{
namespace os
{

using FileTime = std::chrono::time_point<std::chrono::system_clock>;

FileTime modificationTime(const std::string_view& filename);

/**
 * @brief Atomic file replacement
 *
 * This class creates a temporary file that is guaranteed to be unique. On
 * destruction, the file will be moved atomically over the target file.
 **/
class AtomicFileStream : public std::ostream
{
public:
    explicit AtomicFileStream(const std::string& filename);
    ~AtomicFileStream();

private:
    class Private;
    std::unique_ptr<Private> m_d;
};

}
}

#endif
