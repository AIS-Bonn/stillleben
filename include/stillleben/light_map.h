// IBL light map
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_LIGHT_MAP_H
#define STILLLEBEN_LIGHT_MAP_H

#include <memory>
#include <string>

#include <Magnum/GL/CubeMapTexture.h>
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

    inline Magnum::GL::CubeMapTexture& irradianceMap()
    { return m_irradiance; }

    inline Magnum::GL::CubeMapTexture& prefilterMap()
    { return m_prefilter; }

    inline Magnum::GL::Texture2D& brdfLUT()
    { return m_brdfLUT; }

private:
    std::string m_path;

    Magnum::GL::CubeMapTexture m_irradiance;
    Magnum::GL::CubeMapTexture m_prefilter;
    Magnum::GL::Texture2D m_brdfLUT;
};

}

#endif
