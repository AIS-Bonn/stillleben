// Shader for creating shadow maps
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_SHADERS_SHADOW_SHADER_H
#define SL_SHADERS_SHADOW_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class ShadowShader : public GL::AbstractShaderProgram
{
public:
    // Input attributes
    typedef Shaders::GenericGL2D::Position Position;

    ShadowShader();

    explicit ShadowShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    ShadowShader(const ShadowShader&) = delete;
    ShadowShader(ShadowShader&&) noexcept = default;

    ShadowShader& operator=(const ShadowShader&) = delete;
    ShadowShader& operator=(ShadowShader&&) noexcept = default;

    ShadowShader& setTransformation(const Matrix4& transformation);
};

}

#endif
