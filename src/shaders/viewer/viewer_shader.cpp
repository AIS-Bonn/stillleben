// Visualize normals as RGB
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "viewer_shader.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/Resource.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/Math/Color.h>

using namespace Magnum;

namespace sl
{

enum: Int
{
    RGBLayer = 0,
    ObjectCoordinateLayer = 1,
    NormalLayer = 2,
    InstanceIndexLayer = 3,
    ClassIndexLayer = 4
};

ViewerShader::ViewerShader(Magnum::UnsignedInt maxClass, Magnum::UnsignedInt maxInstance)
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

    frag.addSource(Corrade::Utility::formatString(
        "#define MAX_CLASS {}\n"
        "#define MAX_INSTANCE {}\n",
        maxClass, maxInstance
    ));

    vert.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("common.glsl"))
        .addSource(rs.get("viewer/viewer_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("viewer/viewer_shader.frag"));

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

    m_uniform_bbox = 0;
    m_uniform_instanceColors = maxClass+1;

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

sl::ViewerShader& ViewerShader::bindRGB(GL::RectangleTexture& texture)
{
    texture.bind(RGBLayer);
    return *this;
}

sl::ViewerShader& ViewerShader::bindObjectCoordinates(GL::RectangleTexture& texture)
{
    texture.bind(ObjectCoordinateLayer);
    return *this;
}

sl::ViewerShader& ViewerShader::bindInstanceIndex(GL::RectangleTexture& texture)
{
    texture.bind(InstanceIndexLayer);
    return *this;
}

sl::ViewerShader& ViewerShader::bindClassIndex(GL::RectangleTexture& texture)
{
    texture.bind(ClassIndexLayer);
    return *this;
}

sl::ViewerShader& ViewerShader::bindNormals(GL::RectangleTexture& texture)
{
    texture.bind(NormalLayer);
    return *this;
}

sl::ViewerShader& ViewerShader::setObjectBBoxes(const Corrade::Containers::Array<Magnum::Vector3>& bboxes)
{
    setUniform(m_uniform_bbox, bboxes);
    return *this;
}

sl::ViewerShader& ViewerShader::setInstanceColors(const Corrade::Containers::ArrayView<Magnum::Color4>& colors)
{
    setUniform(m_uniform_instanceColors, colors);
    return *this;
}

}
