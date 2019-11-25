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
#include <vector>

#include <Corrade/Containers/Array.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Format.h>
#include <Corrade/Utility/FormatStl.h>

using namespace Corrade::Utility;

namespace sl
{
namespace os
{

FileTime modificationTime(const std::string_view& filename)
{
    struct stat data;
    if(::stat(filename.data(), &data) != 0)
    {
        throw std::runtime_error{format(
            "Could not get file modification date of '{}': {}",
            filename.data(), strerror(errno)
        )};
    }

    return FileTime{
        std::chrono::seconds{data.st_mtim.tv_sec} + std::chrono::nanoseconds{data.st_mtim.tv_nsec}
    };
}


// AtomicFileStream

class AtomicFileStream::Private : public std::streambuf
{
public:
    explicit Private(const std::string& filename)
     : m_filename{filename}
     , m_buf(4096)
    {
        // In order to make this atomic, we create a temporary file, fill
        // it, and move it to the destination.
        std::string TEMPLATE = filename + ".temp-XXXXXX";
        m_temporaryName = {TEMPLATE.begin(), TEMPLATE.end()+1};

        m_fd = mkstemp(m_temporaryName.data());
        if(m_fd < 0)
        {
            throw std::runtime_error(format(
                "Could not create temporary cache file {}: {}",
                m_temporaryName.data(), strerror(errno)
            ));
        }

        // Make it world-readable
        fchmod(m_fd, 0644);

        setp(m_buf.data(), m_buf.data() + m_buf.size());
    }

    ~Private()
    {
        sync();

        close(m_fd);

        // Now move it to the right location (this is atomic)
        if(rename(m_temporaryName.data(), m_filename.c_str()) != 0)
        {
            if(unlink(m_temporaryName.data()) != 0)
            {
                Warning{} << "Could not delete temporary file " << m_temporaryName.data();
            }

            Error{} << formatString(
                "Could not rename file {} -> {}: {}",
                static_cast<const char*>(m_temporaryName.data()),
                m_filename,
                strerror(errno)
            );
            std::terminate();
        }
    }

    int overflow(int c) override
    {
        sync();
        if(c != traits_type::eof())
        {
            *pptr() = c;
            pbump(1);
        }
        return c;
    }

    int sync() override
    {
        if(pptr() > pbase())
        {
            std::size_t count = pptr() - pbase();
            int ret = ::write(m_fd, m_buf.data(), count);
            if(ret < 0 || static_cast<std::size_t>(ret) != count)
            {
                throw std::runtime_error(format(
                    "Could not write to file {}: {}",
                    m_temporaryName.data(), strerror(errno)
                ));
            }

            setp(m_buf.data(), m_buf.data() + m_buf.size());
        }

        return 0;
    }

    std::string m_filename;
    std::vector<char> m_temporaryName;
    int m_fd;
    std::vector<char> m_buf;
};

AtomicFileStream::AtomicFileStream(const std::string& filename)
 // This is ugly as hell, but needed for pimpl.
 : std::ostream{new AtomicFileStream::Private(filename)}
 , m_d{static_cast<AtomicFileStream::Private*>(rdbuf())}
{
}

AtomicFileStream::~AtomicFileStream()
{
}

}
}
