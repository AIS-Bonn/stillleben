// Blit a background texture onto the framebuffer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_BACKGROUND_SHADER_H
#define STILLLEBEN_BACKGROUND_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class BackgroundShader : public GL::AbstractShaderProgram
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
    BackgroundShader();

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
    explicit BackgroundShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    /** @brief Copying is not allowed */
    BackgroundShader(const BackgroundShader&) = delete;

    /** @brief Move constructor */
    BackgroundShader(BackgroundShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    BackgroundShader& operator=(const BackgroundShader&) = delete;

    /** @brief Move assignment */
    BackgroundShader& operator=(BackgroundShader&&) noexcept = default;

    BackgroundShader& bindRGB(GL::RectangleTexture& texture);

private:
};

}

#endif
