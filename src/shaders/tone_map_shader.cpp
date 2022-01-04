// HDR tone mapping & gamma correction
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "tone_map_shader.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <Corrade/Utility/FormatStl.h>

#include <Magnum/ImageView.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/PixelFormat.h>

#include <random>

namespace sl
{

namespace
{
    enum class TextureInput : Int
    {
        Color,
        ObjectLuminance
    };

    template<class T>
    constexpr Int eVal(T val)
    { return static_cast<Int>(val); }
}

ToneMapShader::ToneMapShader()
{
    Utility::Resource rs("stillleben-data");

    const auto version = GL::Version::GL450;

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
        throw std::runtime_error{"The shading system needs the EXPLICIT_ATTRIB_LOCATION GL extension"};

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
        throw std::runtime_error{"The shading system needs the EXPLICIT_UNIFORM_LOCATION GL extension"};

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::shading_language_420pack>(version))
        throw std::runtime_error{"Need SHADING_LANGUAGE_420PACK"};

    GL::Shader vert{version, GL::Shader::Type::Vertex},
        frag{version, GL::Shader::Type::Fragment};

    std::string header = Corrade::Utility::formatString(R"EOS(
// Mesh attributes
#define POSITION_ATTRIBUTE_LOCATION {}
)EOS",
        Shaders::GenericGL3D::Position::Location
    );

    header += Corrade::Utility::formatString(R"EOS(
// Texture samplers
#define COLOR_TEXTURE {}
#define OBJECT_LUMINANCE_TEXTURE {}
)EOS",
        eVal(TextureInput::Color),
        eVal(TextureInput::ObjectLuminance)
    );

    vert.addSource(header)
        .addSource(rs.get("tone_map_shader.vert"));
    frag.addSource(header)
        .addSource(rs.get("tone_map_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

ToneMapShader& ToneMapShader::bindColor(GL::Texture2D& texture)
{
    texture.bind(eVal(TextureInput::Color));
    return *this;
}

ToneMapShader& ToneMapShader::bindObjectLuminance(GL::Texture2D& texture)
{
    texture.bind(eVal(TextureInput::ObjectLuminance));
    return *this;
}

}
