// Resolve the multisampled RBOs to their aggregated values
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "resolve_shader.h"

#include <Corrade/Utility/Resource.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/MultisampleTexture.h>
#include <Magnum/GL/Extensions.h>

namespace sl
{

enum: Int
{
    RGBLayer = 0,
    ObjectCoordinateLayer = 1,
    ClassIndexLayer = 2,
    InstanceIndexLayer = 3,
};

ResolveShader::ResolveShader(unsigned int msaa_factor)
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

    char msaa_str[100];
    snprintf(msaa_str, sizeof(msaa_str), "#define MSAA_SAMPLES %d\n", msaa_factor);

    vert.addSource(msaa_str)
        .addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("resolve_shader.vert"));
    frag.addSource(msaa_str)
        .addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("resolve_shader.frag"));

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
}

sl::ResolveShader& ResolveShader::bindRGB(GL::MultisampleTexture2D& texture)
{
    texture.bind(RGBLayer);
    return *this;
}

sl::ResolveShader& ResolveShader::bindCoordinates(GL::MultisampleTexture2D& texture)
{
    texture.bind(ObjectCoordinateLayer);
    return *this;
}

sl::ResolveShader& ResolveShader::bindClassIndex(GL::MultisampleTexture2D& texture)
{
    texture.bind(ClassIndexLayer);
    return *this;
}

sl::ResolveShader& ResolveShader::bindInstanceIndex(GL::MultisampleTexture2D& texture)
{
    Debug{} << "Binding instance index texture to" << InstanceIndexLayer;
    texture.bind(InstanceIndexLayer);
    return *this;
}

}
