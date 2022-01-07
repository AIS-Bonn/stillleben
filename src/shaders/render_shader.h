// Shader which outputs all needed information
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef RENDER_SHADER_H
#define RENDER_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/GL.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Range.h>

#include <Magnum/Trade/Trade.h>

namespace sl
{

class LightMap;

class MaterialOverride
{
public:
    MaterialOverride& metallic(Magnum::Float m)
    { _metallic = m; return *this; }

    MaterialOverride& roughness(Magnum::Float r)
    { _roughness = r; return *this; }

    constexpr Magnum::Float metallic() const
    {
        return _metallic;
    }
    constexpr Magnum::Float roughness() const
    {
        return _roughness;
    }
private:
    Magnum::Float _metallic = -1.0f;
    Magnum::Float _roughness = -1.0f;
};

class RenderShader : public Magnum::GL::AbstractShaderProgram
{
public:
    /**
     * @brief Vertex position
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector3 "Vector3".
     */
    typedef Magnum::Shaders::GenericGL3D::Position Position;

    /**
     * @brief Normal direction
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector3 "Vector3".
     */
    typedef Magnum::Shaders::GenericGL3D::Normal Normal;
    typedef Magnum::Shaders::GenericGL3D::Tangent Tangent;

    /**
     * @brief 2D texture coordinates
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector2 "Vector2", used only if at least one of
     * @ref Flag::AmbientTexture, @ref Flag::DiffuseTexture and
     * @ref Flag::SpecularTexture is set.
     */
    typedef Magnum::Shaders::GenericGL3D::TextureCoordinates TextureCoordinates;


    /**
     * @brief Vertex colors
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector4 "Vector4"
     */
    typedef Magnum::Shaders::GenericGL3D::Color4 VertexColors;

    /**
     * @brief Vertex index
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::UnsignedInt
     */
    typedef Magnum::GL::Attribute<4, Magnum::UnsignedInt> VertexIndex;

    enum: Magnum::UnsignedInt {
        ColorOutput = 0,
        ObjectCoordinatesOutput = 1,
        ClassIndexOutput = 2,
        InstanceIndexOutput = 3,
        NormalOutput = 4,
        VertexIndexOutput = 5,
        BarycentricCoeffsOutput = 6,
        CamCoordinatesOutput = 7
    };

    /**
     * @brief Flag
     *
     * @see @ref Flags, @ref flags()
     */
    enum class Flag: Magnum::UnsignedByte {
        Flat = 1 << 0
    };

    /**
     * @brief Flags
     *
     * @see @ref flags()
     */
    typedef Corrade::Containers::EnumSet<Flag> Flags;

    /**
     * @brief Constructor
     * @param flags     Flags
     */
    explicit RenderShader();

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
    explicit RenderShader(Magnum::NoCreateT) noexcept: Magnum::GL::AbstractShaderProgram{Magnum::NoCreate} {}

    /** @brief Copying is not allowed */
    RenderShader(const RenderShader&) = delete;

    /** @brief Move constructor */
    RenderShader(RenderShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    RenderShader& operator=(const RenderShader&) = delete;

    /** @brief Move assignment */
    RenderShader& operator=(RenderShader&&) noexcept = default;

    /** @brief Flags */
    void setFlags(Flags flags);
    Flags flags() const { return _flags; }

    RenderShader& bindDepthTexture(Magnum::GL::RectangleTexture& texture);

    //! @name Transformations
    //@{
    RenderShader& setTransformations(const Magnum::Matrix4& meshToObject, const Magnum::Matrix4& objectToWorld, const Magnum::Matrix4& worldToCam);

    RenderShader& setProjectionMatrix(const Magnum::Matrix4& projection);
    //@}


    //! @name Semantic information
    //@{
    RenderShader& setClassIndex(unsigned int classIndex);
    RenderShader& setInstanceIndex(unsigned int instanceIndex);
    //@}

    //! @name Material & lighting
    //@{
    RenderShader& setLightMap(LightMap& lightMap);
    RenderShader& setManualLighting(
        const Corrade::Containers::ArrayView<Magnum::Vector3>& directions,
        const Corrade::Containers::ArrayView<Magnum::Color3>& colors,
        const Magnum::Color3& ambientLight
    );

    RenderShader& setMaterial(
        const Magnum::Trade::MaterialData& material,
        const Corrade::Containers::ArrayView<Corrade::Containers::Optional<Magnum::GL::Texture2D>>& textures,
        const MaterialOverride& materialOverride = {}
    );

    RenderShader& setMaterial(
        const Magnum::Trade::MaterialData& material,
        const Corrade::Containers::ArrayView<Magnum::GL::Texture2D*>& textures,
        const MaterialOverride& materialOverride = {}
    );

    RenderShader& setShadowMap(
        Magnum::GL::Texture2DArray& shadowMaps,
        const Corrade::Containers::ArrayView<Magnum::Matrix4>& shadowMatrices
    );
    //@}

    //@{
    RenderShader& setStickerProjection(const Magnum::Matrix4 proj);
    RenderShader& setStickerRange(const Magnum::Range2D& range);

    RenderShader& bindStickerTexture(Magnum::GL::RectangleTexture& texture);
    //@}

private:
    Flags _flags;
};

}

CORRADE_ENUMSET_OPERATORS(sl::RenderShader::Flags)

#endif

