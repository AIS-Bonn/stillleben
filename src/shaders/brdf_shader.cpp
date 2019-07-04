// Pre-compute BRDF LUT
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "brdf_shader.h"

#include <Magnum/GL/Version.h>
#include <Magnum/GL/Shader.h>

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>

using namespace Magnum;

namespace sl
{

BRDFShader::BRDFShader()
{
    Utility::Resource rs("stillleben-data");

    const auto version = GL::Version::GL450;

    GL::Shader vert{version, GL::Shader::Type::Vertex},
        frag{version, GL::Shader::Type::Fragment};

    vert.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("brdf_shader.vert"));
    frag.addSource(rs.get("compatibility.glsl"))
        .addSource(rs.get("brdf_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

}
