// HDR tone mapping & gamma correction
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_SHADERS_TONE_MAPPING_H
#define SL_SHADERS_TONE_MAPPING_H

#include <Corrade/Containers/Array.h>

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class ToneMapShader : public GL::AbstractShaderProgram
{
public:
    // Input attributes
    typedef Shaders::GenericGL2D::Position Position;

    // Outputs
    enum: UnsignedInt {
        ColorOutput = 0
    };

    /**
     * @brief Constructor
     */
    ToneMapShader();

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
    explicit ToneMapShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    /** @brief Copying is not allowed */
    ToneMapShader(const ToneMapShader&) = delete;

    /** @brief Move constructor */
    ToneMapShader(ToneMapShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    ToneMapShader& operator=(const ToneMapShader&) = delete;

    /** @brief Move assignment */
    ToneMapShader& operator=(ToneMapShader&&) noexcept = default;

    ToneMapShader& bindColor(GL::Texture2D& texture);
    ToneMapShader& bindObjectLuminance(GL::Texture2D& texture);

private:
};

}

#endif
