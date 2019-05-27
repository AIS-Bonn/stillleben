// Loads & manages object meshes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh_cache.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>

#include <Corrade/Utility/ConfigurationGroup.h>

namespace sl
{

MeshCache::MeshCache(const std::shared_ptr<Context>& ctx)
 : m_ctx{ctx}
{
}

MeshCache::~MeshCache() = default;

std::shared_ptr<Mesh> MeshCache::load(const Corrade::Utility::ConfigurationGroup& group)
{
    std::string filename = group.value("filename");

    auto it = m_cache.find(filename);

    if(it != m_cache.end())
        return it->second;

    auto mesh = std::make_shared<Mesh>(filename, m_ctx);
    mesh->deserialize(group);

    m_cache[filename] = mesh;

    return mesh;
}

}
