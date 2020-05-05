// Visualize normals as RGB
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "viewer_shader.h"

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/mesh.h>

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
using namespace Magnum::Math::Literals;

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

ViewerShader::ViewerShader(const std::shared_ptr<sl::Scene>& scene)
 : m_scene{scene}
{
    Magnum::UnsignedInt maxClass = 0;
    Magnum::UnsignedInt maxInstance = 0;
    std::vector<sl::Mesh*> meshes;
    const Magnum::Color4 BG_COLOR{0x2f363fff_rgbaf};

    for(const auto& obj : scene->objects())
    {
        maxClass = std::max(maxClass, obj->mesh()->classIndex());
        maxInstance = std::max(maxInstance, obj->instanceIndex());
        meshes.resize(maxClass+1);
        meshes[obj->mesh()->classIndex()] = obj->mesh().get();
    }

    Corrade::Containers::Array<Magnum::Color4> instanceColors(maxInstance+1);
    instanceColors[0] = BG_COLOR;

    Corrade::Containers::Array<Magnum::Vector3> bboxes(maxInstance+1);
    bboxes[0] = {scene->backgroundPlaneSize(), 1.0f};

    for(const auto& obj : scene->objects())
    {
        instanceColors[obj->instanceIndex()] =
            Magnum::Color4::fromHsv(Magnum::ColorHsv{
                Magnum::Deg(360.0) / (maxInstance+1) * obj->instanceIndex(),
                1.0,
                1.0
            });
        bboxes[obj->instanceIndex()] = obj->mesh()->bbox().size();
    }

    Corrade::Containers::Array<Magnum::Color4> classColors(meshes.size());
    classColors[0] = BG_COLOR;

    for(Magnum::UnsignedInt i = 0; i < meshes.size(); ++i)
    {
        if(!meshes[i])
            continue;

        classColors[i] =
            Magnum::Color4::fromHsv(Magnum::ColorHsv{
                Magnum::Deg(360.0) / (maxClass+1) * i,
                1.0,
                1.0
            });
    }

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
    m_uniform_instanceColors = maxInstance+1;
    m_uniform_classColors = m_uniform_instanceColors + maxInstance+1;

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    setUniform(m_uniform_bbox, bboxes);
    setUniform(m_uniform_instanceColors, instanceColors);
    setUniform(m_uniform_classColors, classColors);
}

ViewerShader::~ViewerShader() = default;

void ViewerShader::setData(sl::RenderPass::Result& result)
{
    GL::RectangleTexture::bind(0, {
        &result.rgb,
        &result.objectCoordinates,
        &result.normals,
        &result.instanceIndex,
        &result.classIndex
    });
}

}
