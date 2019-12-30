// Render cubemap as background
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_BACKGROUND_CUBEMAP_SHADER_H
#define STILLLEBEN_BACKGROUND_CUBEMAP_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

class BackgroundCubeShader : public Magnum::GL::AbstractShaderProgram
{
public:
    // Input attributes
    using Position = Magnum::Shaders::Generic2D::Position;

    // Outputs
    enum: Magnum::UnsignedInt {
        ColorOutput = 0
    };

    /**
     * @brief Constructor
     */
    BackgroundCubeShader();

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
    explicit BackgroundCubeShader(Magnum::NoCreateT) noexcept
     : Magnum::GL::AbstractShaderProgram{Magnum::NoCreate}
    {}

    /** @brief Copying is not allowed */
    BackgroundCubeShader(const BackgroundCubeShader&) = delete;

    /** @brief Move constructor */
    BackgroundCubeShader(BackgroundCubeShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    BackgroundCubeShader& operator=(const BackgroundCubeShader&) = delete;

    /** @brief Move assignment */
    BackgroundCubeShader& operator=(BackgroundCubeShader&&) noexcept = default;

    BackgroundCubeShader& bindRGB(Magnum::GL::CubeMapTexture& texture);

    void setViewMatrix(const Magnum::Matrix4& view);

    void setProjectionMatrix(const Magnum::Matrix4& projection);

private:
    Magnum::UnsignedInt m_uniform_view{0};
    Magnum::UnsignedInt m_uniform_proj{1};
};

}

#endif
