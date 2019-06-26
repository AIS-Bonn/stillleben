// IBL light map
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_LIGHT_MAP_H
#define STILLLEBEN_LIGHT_MAP_H

#include <memory>
#include <string>

#include <Magnum/GL/Texture.h>

namespace sl
{

class Context;

class LightMap
{
public:
    LightMap();
    explicit LightMap(const std::string& path, const std::shared_ptr<Context>& ctx);

    bool load(const std::string& path, const std::shared_ptr<Context>& ctx);

    constexpr const std::string& path() const
    { return m_path; }

    inline Magnum::GL::Texture2D& diffuseTexture()
    { return m_diffuseTexture; }

    inline Magnum::GL::Texture2D& specularTexture()
    { return m_specularTexture; }

private:
    std::string m_path;

    Magnum::GL::Texture2D m_diffuseTexture;
    Magnum::GL::Texture2D m_specularTexture;
};

}

#endif
