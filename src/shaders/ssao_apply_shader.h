// Screen-space Ambient Occlusion shader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SSAO_APPLY_SHADER_H
#define STILLLEBEN_SSAO_APPLY_SHADER_H

#include <Corrade/Containers/Array.h>

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class SSAOApplyShader : public GL::AbstractShaderProgram
{
public:
    // Input attributes
    typedef Shaders::Generic2D::Position Position;

    // Outputs
    enum: UnsignedInt {
        ColorOutput = 0
    };

    /**
     * @brief Constructor
     */
    SSAOApplyShader();

    /**
     * @brief Construct without creating the underlying OpenGL object
     *
     * The constructed instance is equivalent to moved-from state. Useful
     * in cases where you will overwrite the instance later anyway. Move
     * another object over it to make it useful.
     *
     * This function can be safely used for constructing (and later
     * destructing) objects even without any OpenGL context being active.
     */
    explicit SSAOApplyShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    /** @brief Copying is not allowed */
    SSAOApplyShader(const SSAOApplyShader&) = delete;

    /** @brief Move constructor */
    SSAOApplyShader(SSAOApplyShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    SSAOApplyShader& operator=(const SSAOApplyShader&) = delete;

    /** @brief Move assignment */
    SSAOApplyShader& operator=(SSAOApplyShader&&) noexcept = default;

    SSAOApplyShader& bindAO(GL::RectangleTexture& texture);
    SSAOApplyShader& bindColor(GL::RectangleTexture& texture);
    SSAOApplyShader& bindCoordinates(GL::RectangleTexture& texture);

private:
};

}

#endif


