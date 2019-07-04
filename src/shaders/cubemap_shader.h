// Convert equirectangular HDR maps to cubemap textures
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SHADERS_CUBEMAP_SHADER_H
#define STILLLEBEN_SHADERS_CUBEMAP_SHADER_H

// Based on Joey de Vries' PBR shaders from https://learnopengl.com/
// which are licensed under CC BY-NC 4.0
// https://twitter.com/JoeyDeVriez

#include <Magnum/GL/AbstractShaderProgram.h>

namespace sl
{

class CubeMapShader : public Magnum::GL::AbstractShaderProgram
{
public:
    enum class Phase
    {
        EquirectangularConversion,
        IrradianceConvolution,
        Prefilter
    };

    explicit CubeMapShader(Phase phase);

    explicit CubeMapShader(Magnum::NoCreateT) noexcept
     : Magnum::GL::AbstractShaderProgram{Magnum::NoCreate}
    {}

    /** @brief Copying is not allowed */
    CubeMapShader(const CubeMapShader&) = delete;

    /** @brief Move constructor */
    CubeMapShader(CubeMapShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    CubeMapShader& operator=(const CubeMapShader&) = delete;

    /** @brief Move assignment */
    CubeMapShader& operator=(CubeMapShader&&) noexcept = default;

    /** @brief Bind input equirectangular texture */
    CubeMapShader& bindInputTexture(Magnum::GL::Texture2D& texture);

    /** @brief Bind input cubemap texture */
    CubeMapShader& bindInputTexture(Magnum::GL::CubeMapTexture& texture);

    void setProjection(const Magnum::Matrix4& projection);
    void setView(const Magnum::Matrix4& view);
    void setRoughness(const Magnum::Float roughness);
private:
    Phase m_phase;

    Magnum::Int m_projectionUniform{0};
    Magnum::Int m_viewUniform{1};
    Magnum::Int m_roughnessUniform{2};
};

}

#endif
