// Pre-compute BRDF LUT
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SHADERS_BRDF_SHADER_H
#define STILLLEBEN_SHADERS_BRDF_SHADER_H

// Based on Joey de Vries' PBR shaders from https://learnopengl.com/
// which are licensed under CC BY-NC 4.0
// https://twitter.com/JoeyDeVriez

#include <Magnum/GL/AbstractShaderProgram.h>

namespace sl
{

class BRDFShader : public Magnum::GL::AbstractShaderProgram
{
public:
    BRDFShader();

    explicit BRDFShader(Magnum::NoCreateT) noexcept
     : Magnum::GL::AbstractShaderProgram{Magnum::NoCreate}
    {}

    /** @brief Copying is not allowed */
    BRDFShader(const BRDFShader&) = delete;

    /** @brief Move constructor */
    BRDFShader(BRDFShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    BRDFShader& operator=(const BRDFShader&) = delete;

    /** @brief Move assignment */
    BRDFShader& operator=(BRDFShader&&) noexcept = default;
};

}

#endif

