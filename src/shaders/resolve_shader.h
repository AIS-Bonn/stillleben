// Resolve the multisampled RBOs to their aggregated values
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_RESOLVE_SHADER_H
#define STILLLEBEN_RESOLVE_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class ResolveShader : public GL::AbstractShaderProgram
{
public:
    // Input attributes
    typedef Shaders::Generic2D::Position Position;

    // Outputs
    enum: UnsignedInt {
        ColorOutput = 0,
        ObjectCoordinatesOutput = 1,
        ClassIndexOutput = 2,
        InstanceIndexOutput = 3,
        ValidMaskOutput = 4
    };

    /**
     * @brief Constructor
     */
    explicit ResolveShader(unsigned int msaa_factor);

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
    explicit ResolveShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    /** @brief Copying is not allowed */
    ResolveShader(const ResolveShader&) = delete;

    /** @brief Move constructor */
    ResolveShader(ResolveShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    ResolveShader& operator=(const ResolveShader&) = delete;

    /** @brief Move assignment */
    ResolveShader& operator=(ResolveShader&&) noexcept = default;

    ResolveShader& bindRGB(GL::MultisampleTexture2D& texture);
    ResolveShader& bindCoordinates(GL::MultisampleTexture2D& texture);
    ResolveShader& bindClassIndex(GL::MultisampleTexture2D& texture);
    ResolveShader& bindInstanceIndex(GL::MultisampleTexture2D& texture);

private:
};

}

#endif
