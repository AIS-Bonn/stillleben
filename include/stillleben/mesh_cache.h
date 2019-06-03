// Loads & manages object meshes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_MESH_CACHE_H
#define STILLLEBEN_MESH_CACHE_H

#include <memory>
#include <map>
#include <vector>

namespace Corrade { namespace Utility { class ConfigurationGroup; } }

namespace sl
{

class Context;
class Mesh;

class MeshCache
{
public:
    explicit MeshCache(const std::shared_ptr<Context>& ctx);
    ~MeshCache();

    std::shared_ptr<Mesh> load(const Corrade::Utility::ConfigurationGroup& group);

    void add(const std::vector<std::shared_ptr<Mesh>>& meshes);
private:
    std::shared_ptr<Context> m_ctx;
    std::map<std::string, std::shared_ptr<Mesh>> m_cache;
};

}

#endif
