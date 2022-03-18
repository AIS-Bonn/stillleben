// IBL light map
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_LIGHT_MAP_H
#define STILLLEBEN_LIGHT_MAP_H

#include <memory>
#include <string>

#include <Corrade/Containers/Array.h>

#include <Magnum/GL/CubeMapTexture.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/Math/Color.h>

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

    inline Magnum::GL::CubeMapTexture& cubeMap()
    { return m_cubeMap; }

    inline Corrade::Containers::ArrayView<const Magnum::Vector3> lightDirections() const
    { return m_lightDirections; }

    inline Corrade::Containers::ArrayView<const Magnum::Color3> lightColors() const
    { return m_lightColors; }

private:
    std::string m_path;

    Magnum::GL::CubeMapTexture m_cubeMap;
    Magnum::GL::CubeMapTexture m_irradiance;
    Magnum::GL::CubeMapTexture m_prefilter;
    Magnum::GL::Texture2D m_brdfLUT;

    Corrade::Containers::Array<Magnum::Vector3> m_lightDirections;
    Corrade::Containers::Array<Magnum::Color3> m_lightColors;
};

}

#endif

