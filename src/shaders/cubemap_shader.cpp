// Convert equirectangular HDR maps to cubemap textures
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "cubemap_shader.h"

#include <Magnum/GL/Version.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>

#include <Magnum/Math/Matrix4.h>

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>

using namespace Magnum;

namespace sl
{

CubeMapShader::CubeMapShader()
{
    Utility::Resource rs("stillleben-data");

    const auto version = GL::Version::GL450;

    GL::Shader vert{version, GL::Shader::Type::Vertex},
        frag{version, GL::Shader::Type::Fragment};

    vert.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("cubemap_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("cubemap_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

sl::CubeMapShader& CubeMapShader::bindInputTexture(Magnum::GL::Texture2D& texture)
{
    texture.bind(0);
    return *this;
}

void CubeMapShader::setProjection(const Magnum::Matrix4& projection)
{
    setUniform(m_projectionUniform, projection);
}

void CubeMapShader::setView(const Magnum::Matrix4& view)
{
    setUniform(m_viewUniform, view);
}

}
