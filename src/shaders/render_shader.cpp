// Shader which outputs all needed information
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>
// with parts taken from the Magnum engine

#include "render_shader.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/Extensions.h>

using namespace Magnum;

namespace sl
{

namespace
{
    enum: Int
    {
        AmbientTextureLayer = 0,
        DiffuseTextureLayer = 1,
        SpecularTextureLayer = 2
    };
}

RenderShader::RenderShader(const Flags flags)
 : _flags(flags)
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

    vert.addSource(flags ? "#define TEXTURED\n" : "")
        .addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("render_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(flags & Flag::AmbientTexture ? "#define AMBIENT_TEXTURE\n" : "")
        .addSource(flags & Flag::DiffuseTexture ? "#define DIFFUSE_TEXTURE\n" : "")
        .addSource(flags & Flag::SpecularTexture ? "#define SPECULAR_TEXTURE\n" : "")
        .addSource(flags & Flag::AlphaMask ? "#define ALPHA_MASK\n" : "")
        .addSource(rs.get("render_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
    #else
    if(!GL::Context::current().isVersionSupported(GL::Version::GLES300))
    #endif
    {
        bindAttributeLocation(Position::Location, "position");
        bindAttributeLocation(Normal::Location, "normal");
        if(flags & (Flag::AmbientTexture|Flag::DiffuseTexture|Flag::SpecularTexture))
            bindAttributeLocation(TextureCoordinates::Location, "textureCoordinates");
    }

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
    #endif
    {
        _meshToObjectMatrixUniform = uniformLocation("meshToObject");
        _objectToCamMatrixUniform = uniformLocation("objectToCam");
        _projectionMatrixUniform = uniformLocation("projectionMatrix");
        _normalMatrixUniform = uniformLocation("normalMatrix");
        _lightPositionUniform = uniformLocation("lightPosition");
        _ambientColorUniform = uniformLocation("ambientColor");
        _diffuseColorUniform = uniformLocation("diffuseColor");
        _specularColorUniform = uniformLocation("specularColor");
        _lightColorUniform = uniformLocation("lightColor");
        _shininessUniform = uniformLocation("shininess");
        _classIndexUniform = uniformLocation("classIndex");
        _instanceIndexUniform = uniformLocation("instanceIndex");
        if(flags & Flag::AlphaMask) _alphaMaskUniform = uniformLocation("alphaMask");
    }

    #ifndef MAGNUM_TARGET_GLES
    if(flags && !GL::Context::current().isExtensionSupported<GL::Extensions::ARB::shading_language_420pack>(version))
    #endif
    {
        if(flags & Flag::AmbientTexture) setUniform(uniformLocation("ambientTexture"), AmbientTextureLayer);
        if(flags & Flag::DiffuseTexture) setUniform(uniformLocation("diffuseTexture"), DiffuseTextureLayer);
        if(flags & Flag::SpecularTexture) setUniform(uniformLocation("specularTexture"), SpecularTextureLayer);
    }
}

RenderShader& RenderShader::bindAmbientTexture(GL::Texture2D& texture)
{
    CORRADE_ASSERT(_flags & Flag::AmbientTexture,
        "Shaders::RenderShader::bindAmbientTexture(): the shader was not created with ambient texture enabled", *this);
    texture.bind(AmbientTextureLayer);
    return *this;
}

RenderShader& RenderShader::bindDiffuseTexture(GL::Texture2D& texture)
{
    CORRADE_ASSERT(_flags & Flag::DiffuseTexture,
        "Shaders::RenderShader::bindDiffuseTexture(): the shader was not created with diffuse texture enabled", *this);
    texture.bind(DiffuseTextureLayer);
    return *this;
}

RenderShader& RenderShader::bindSpecularTexture(GL::Texture2D& texture)
{
    CORRADE_ASSERT(_flags & Flag::SpecularTexture,
        "Shaders::RenderShader::bindSpecularTexture(): the shader was not created with specular texture enabled", *this);
    texture.bind(SpecularTextureLayer);
    return *this;
}

RenderShader& RenderShader::bindTextures(GL::Texture2D* ambient, GL::Texture2D* diffuse, GL::Texture2D* specular)
{
    CORRADE_ASSERT(_flags & (Flag::AmbientTexture|Flag::DiffuseTexture|Flag::SpecularTexture),
        "Shaders::RenderShader::bindTextures(): the shader was not created with any textures enabled", *this);
    GL::AbstractTexture::bind(AmbientTextureLayer, {ambient, diffuse, specular});
    return *this;
}

RenderShader& RenderShader::setAlphaMask(Float mask)
{
    CORRADE_ASSERT(_flags & Flag::AlphaMask,
        "Shaders::RenderShader::setAlphaMask(): the shader was not created with alpha mask enabled", *this);
    setUniform(_alphaMaskUniform, mask);
    return *this;
}

}
