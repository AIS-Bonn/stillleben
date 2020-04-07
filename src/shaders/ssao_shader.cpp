// Screen-space Ambient Occlusion shader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "ssao_shader.h"

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
    CoordinateLayer = 0,
    NormalLayer = 1,
    NoiseLayer = 2
};

SSAOShader::SSAOShader()
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
        .addSource(rs.get("ssao_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("ssao_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
        bindAttributeLocation(Position::Location, "position");

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    // Create noise texture
    Corrade::Containers::Array<Magnum::Vector3> noiseData{16};

    // We need some deterministic random numbers.
    std::mt19937 random{0xdeadbeef};
    std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);

    for(int i = 0; i < 16; ++i)
    {
        // around Z axis in tangent space
        noiseData[i] = Magnum::Vector3{
            2.0f*randomFloat(random) - 1.0f,
            2.0f*randomFloat(random) - 1.0f,
            0.0f
        };
    }

    m_noiseTexture = GL::Texture2D{};
    m_noiseTexture.setStorage(1, GL::TextureFormat::RGB32F, {4,4});
    m_noiseTexture.setSubImage(0, {}, ImageView2D{PixelFormat::RGB32F, {4,4}, noiseData});
    m_noiseTexture.setWrapping({SamplerWrapping::Repeat, SamplerWrapping::Repeat});
    m_noiseTexture.setMinificationFilter(SamplerFilter::Nearest, SamplerMipmap::Nearest);
    m_noiseTexture.setMagnificationFilter(SamplerFilter::Nearest);

    for(int i = 0; i < 64; ++i)
    {
        Vector3 sample{
            2.0f * randomFloat(random) - 1.0f,
            2.0f * randomFloat(random) - 1.0f,
            randomFloat(random)
        };

        sample = randomFloat(random) * sample.normalized();

        float scale = static_cast<float>(i) / 64.0f;

        // scale samples s.t. they're more aligned to center of kernel
        sample *= Math::lerp(0.1f, 1.0f, scale*scale);

        m_ssaoKernel[i] = sample;
    }
}

SSAOShader& SSAOShader::bindCoordinates(GL::RectangleTexture& texture)
{
    texture.bind(CoordinateLayer);
    return *this;
}

SSAOShader& SSAOShader::bindNormals(GL::RectangleTexture& texture)
{
    texture.bind(NormalLayer);
    return *this;
}

SSAOShader& SSAOShader::bindNoise()
{
    m_noiseTexture.bind(NoiseLayer);
    setUniform(m_samplesUniform, m_ssaoKernel);
    return *this;
}

SSAOShader& SSAOShader::setProjection(const Magnum::Matrix4& projection)
{
    setUniform(m_projectionUniform, projection);
    return *this;
}

}
