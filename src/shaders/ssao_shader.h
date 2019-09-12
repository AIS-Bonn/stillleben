// Screen-space Ambient Occlusion shader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SSAO_SHADER_H
#define STILLLEBEN_SSAO_SHADER_H

#include <Corrade/Containers/Array.h>

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class SSAOShader : public GL::AbstractShaderProgram
{
public:
    // Input attributes
    typedef Shaders::Generic2D::Position Position;

    // Outputs
    enum: UnsignedInt {
        AOOutput = 0
    };

    /**
     * @brief Constructor
     */
    SSAOShader();

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
    explicit SSAOShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    /** @brief Copying is not allowed */
    SSAOShader(const SSAOShader&) = delete;

    /** @brief Move constructor */
    SSAOShader(SSAOShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    SSAOShader& operator=(const SSAOShader&) = delete;

    /** @brief Move assignment */
    SSAOShader& operator=(SSAOShader&&) noexcept = default;

    SSAOShader& bindCoordinates(GL::RectangleTexture& texture);
    SSAOShader& bindNormals(GL::RectangleTexture& texture);
    SSAOShader& bindNoise();

    SSAOShader& setProjection(const Magnum::Matrix4& projection);

private:
    GL::Texture2D m_noiseTexture{NoCreate};
    UnsignedInt m_samplesUniform{0};
    UnsignedInt m_projectionUniform{65};

    Corrade::Containers::Array<Magnum::Vector3> m_ssaoKernel{64};
};

}

#endif

