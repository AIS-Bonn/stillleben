// stillleben context
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_CONTEXT_H
#define STILLLEBEN_CONTEXT_H

#include <memory>
#include <string>

namespace Corrade
{
    namespace PluginManager
    {
        template<class> class Manager;
    }
}

namespace Magnum
{
    namespace Trade
    {
        class AbstractImporter;
    }
    namespace GL
    {
        class RectangleTexture;
    }
}

namespace sl
{

class Mesh;

class Context
{
public:
    using Ptr = std::shared_ptr<Context>;

    Context(const Context& other) = delete;
    Context(const Context&& other) = delete;

    static Ptr Create(const std::string& installPrefix = {});
    static Ptr CreateCUDA(unsigned int device = 0, const std::string& installPrefix = {});

    bool makeCurrent();

    std::shared_ptr<Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter>> importerPluginManager();

    Magnum::GL::RectangleTexture loadTexture(const std::string& path);
private:
    class Private;

    Context(const std::string& installPrefix = {});

    std::unique_ptr<Private> m_d;
};

}

#endif
