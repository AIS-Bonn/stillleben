// Screen-space Ambient Occlusion shader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "ssao_apply_shader.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>

#include <Magnum/ImageView.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/PixelFormat.h>

#include <random>

namespace sl
{

enum: Int
{
    NormalLayer = 0,
    AOLayer = 1,
    CoordinateLayer = 2
};

SSAOApplyShader::SSAOApplyShader()
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
        .addSource(rs.get("ssao_apply_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("ssao_apply_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
        bindAttributeLocation(Position::Location, "position");

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

SSAOApplyShader& SSAOApplyShader::bindColor(GL::Texture2D& texture)
{
    texture.bind(NormalLayer);
    return *this;
}

SSAOApplyShader& SSAOApplyShader::bindAO(GL::Texture2D& texture)
{
    texture.bind(AOLayer);
    return *this;
}

SSAOApplyShader& SSAOApplyShader::bindCoordinates(GL::RectangleTexture& texture)
{
    texture.bind(CoordinateLayer);
    return *this;
}

}
