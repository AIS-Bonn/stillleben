// Render cubemap as background
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "background_cube_shader.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/CubeMapTexture.h>
#include <Magnum/GL/Extensions.h>

using namespace Magnum;

namespace sl
{

enum: Int
{
    RGBLayer = 0
};

BackgroundCubeShader::BackgroundCubeShader()
{
    Utility::Resource rs("stillleben-data");

    const auto version = GL::Version::GL450;

    GL::Shader vert{version, GL::Shader::Type::Vertex},
        frag{version, GL::Shader::Type::Fragment};

    if(GL::Context::current().isExtensionDisabled<GL::Extensions::ARB::explicit_attrib_location>(version))
    {
        vert.addSource("#define DISABLE_GL_ARB_explicit_attrib_location\n");
        frag.addSource("#define DISABLE_GL_ARB_explicit_attrib_location\n");
    }
    if(GL::Context::current().isExtensionDisabled<GL::Extensions::ARB::shading_language_420pack>(version))
    {
        vert.addSource("#define DISABLE_GL_ARB_shading_language_420pack\n");
        frag.addSource("#define DISABLE_GL_ARB_shading_language_420pack\n");
    }
    if(GL::Context::current().isExtensionDisabled<GL::Extensions::ARB::explicit_uniform_location>(version))
    {
        vert.addSource("#define DISABLE_GL_ARB_explicit_uniform_location\n");
        frag.addSource("#define DISABLE_GL_ARB_explicit_uniform_location\n");
    }

    vert.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("common.glsl"))
        .addSource(rs.get("background_cube_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("background_cube_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
    #else
    if(!GL::Context::current().isVersionSupported(GL::Version::GLES300))
    #endif
    {
        bindAttributeLocation(Position::Location, "position");
    }

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
    #endif
    {
        m_uniform_view = uniformLocation("view");
        m_uniform_proj = uniformLocation("projection");
    }

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::shading_language_420pack>(version))
    #endif
    {
        setUniform(uniformLocation("cubeMap"), RGBLayer);
    }
}

BackgroundCubeShader& BackgroundCubeShader::bindRGB(GL::CubeMapTexture& texture)
{
    texture.bind(RGBLayer);
    return *this;
}

void BackgroundCubeShader::setViewMatrix(const Magnum::Matrix4& view)
{
    setUniform(m_uniform_view, view);
}

void BackgroundCubeShader::setProjectionMatrix(const Magnum::Matrix4& projection)
{
    setUniform(m_uniform_proj, projection);
}

}
