// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_MESH_H
#define STILLLEBEN_MESH_H

#include <memory>

#include <stillleben/exception.h>

namespace sl
{

class Context;

class Mesh
{
public:
    class Private;
    class LoadException : public Exception
    {
        using Exception::Exception;
    };

    Mesh(const std::shared_ptr<Context>& ctx);
    Mesh(const Mesh& other) = delete;
    Mesh(Mesh&& other);
    ~Mesh();

    void load(const std::string& filename);

    const Private& impl() const
    { return *m_d; }
private:
    std::unique_ptr<Private> m_d;
};

}

#endif
