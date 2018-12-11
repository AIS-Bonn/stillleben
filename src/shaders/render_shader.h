// Shader which outputs all needed information
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef RENDER_SHADER_H
#define RENDER_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

namespace sl
{

using namespace Magnum;

class RenderShader : public GL::AbstractShaderProgram
{
public:
    /**
     * @brief Vertex position
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector3 "Vector3".
     */
    typedef Shaders::Generic3D::Position Position;

    /**
     * @brief Normal direction
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector3 "Vector3".
     */
    typedef Shaders::Generic3D::Normal Normal;

    /**
     * @brief 2D texture coordinates
     *
     * @ref shaders-generic "Generic attribute",
     * @ref Magnum::Vector2 "Vector2", used only if at least one of
     * @ref Flag::AmbientTexture, @ref Flag::DiffuseTexture and
     * @ref Flag::SpecularTexture is set.
     */
    typedef Shaders::Generic3D::TextureCoordinates TextureCoordinates;

    enum: UnsignedInt {
        ColorOutput = 0,
        ObjectCoordinatesOutput = 1,
        ClassIndexOutput = 2,
        InstanceIndexOutput = 3
    };

    /**
     * @brief Flag
     *
     * @see @ref Flags, @ref flags()
     */
    enum class Flag: UnsignedByte {
        /**
         * Multiply ambient color with a texture.
         * @see @ref setAmbientColor(), @ref setAmbientTexture()
         */
        AmbientTexture = 1 << 0,

        /**
         * Multiply diffuse color with a texture.
         * @see @ref setDiffuseColor(), @ref setDiffuseTexture()
         */
        DiffuseTexture = 1 << 1,

        /**
         * Multiply specular color with a texture.
         * @see @ref setSpecularColor(), @ref setSpecularTexture()
         */
        SpecularTexture = 1 << 2,

        /**
         * Enable alpha masking. If the combined fragment color has an
         * alpha less than the value specified with @ref setAlphaMask(),
         * given fragment is discarded.
         *
         * This uses the @glsl discard @ce operation which is known to have
         * considerable performance impact on some platforms. While useful
         * for cheap alpha masking that doesn't require depth sorting,
         * with proper depth sorting and blending you'll usually get much
         * better performance and output quality.
         */
        AlphaMask = 1 << 3
    };

    /**
     * @brief Flags
     *
     * @see @ref flags()
     */
    typedef Containers::EnumSet<Flag> Flags;

    /**
     * @brief Constructor
     * @param flags     Flags
     */
    explicit RenderShader(Flags flags = {});

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
    explicit RenderShader(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

    /** @brief Copying is not allowed */
    RenderShader(const RenderShader&) = delete;

    /** @brief Move constructor */
    RenderShader(RenderShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    RenderShader& operator=(const RenderShader&) = delete;

    /** @brief Move assignment */
    RenderShader& operator=(RenderShader&&) noexcept = default;

    /** @brief Flags */
    Flags flags() const { return _flags; }

    /**
     * @brief Set ambient color
     * @return Reference to self (for method chaining)
     *
     * If @ref Flag::AmbientTexture is set, default value is
     * @cpp 0xffffffff_rgbaf @ce and the color will be multiplied with
     * ambient texture, otherwise default value is @cpp 0x00000000_rgbaf @ce.
     * @see @ref bindAmbientTexture()
     */
    RenderShader& setAmbientColor(const Color4& color) {
        setUniform(_ambientColorUniform, color);
        return *this;
    }

    /**
     * @brief Bind an ambient texture
     * @return Reference to self (for method chaining)
     *
     * Expects that the shader was created with @ref Flag::AmbientTexture
     * enabled.
     * @see @ref bindTextures(), @ref setAmbientColor()
     */
    RenderShader& bindAmbientTexture(GL::Texture2D& texture);

    #ifdef MAGNUM_BUILD_DEPRECATED
    /** @brief @copybrief bindAmbientTexture()
     * @deprecated Use @ref bindAmbientTexture() instead.
     */
    CORRADE_DEPRECATED("use bindAmbientTexture() instead") RenderShader& setAmbientTexture(GL::Texture2D& texture) {
        return bindAmbientTexture(texture);
    }
    #endif

    /**
     * @brief Set diffuse color
     * @return Reference to self (for method chaining)
     *
     * Initial value is @cpp 0xffffffff_rgbaf @ce.
     * @see @ref bindDiffuseTexture()
     */
    RenderShader& setDiffuseColor(const Color4& color) {
        setUniform(_diffuseColorUniform, color);
        return *this;
    }

    /**
     * @brief Bind a diffuse texture
     * @return Reference to self (for method chaining)
     *
     * Expects that the shader was created with @ref Flag::DiffuseTexture
     * enabled.
     * @see @ref bindTextures(), @ref setDiffuseColor()
     */
    RenderShader& bindDiffuseTexture(GL::Texture2D& texture);

    #ifdef MAGNUM_BUILD_DEPRECATED
    /** @brief @copybrief bindDiffuseTexture()
     * @deprecated Use @ref bindDiffuseTexture() instead.
     */
    CORRADE_DEPRECATED("use bindDiffuseTexture() instead") RenderShader& setDiffuseTexture(GL::Texture2D& texture) {
        return bindDiffuseTexture(texture);
    }
    #endif

    /**
     * @brief Set specular color
     * @return Reference to self (for method chaining)
     *
     * Initial value is @cpp 0xffffffff_rgbaf @ce. Color will be multiplied
     * with specular texture if @ref Flag::SpecularTexture is set. If you
     * want to have a fully diffuse material, set specular color to
     * @cpp 0x000000ff_rgbaf @ce.
     * @see @ref bindSpecularTexture()
     */
    RenderShader& setSpecularColor(const Color4& color) {
        setUniform(_specularColorUniform, color);
        return *this;
    }

    /**
     * @brief Bind a specular texture
     * @return Reference to self (for method chaining)
     *
     * Expects that the shader was created with @ref Flag::SpecularTexture
     * enabled.
     * @see @ref bindTextures(), @ref setSpecularColor()
     */
    RenderShader& bindSpecularTexture(GL::Texture2D& texture);

    #ifdef MAGNUM_BUILD_DEPRECATED
    /** @brief @copybrief bindSpecularTexture()
     * @deprecated Use @ref bindSpecularTexture() instead.
     */
    CORRADE_DEPRECATED("use bindSpecularTexture() instead") RenderShader& setSpecularTexture(GL::Texture2D& texture) {
        return bindSpecularTexture(texture);
    }
    #endif

    /**
     * @brief Bind textures
     * @return Reference to self (for method chaining)
     *
     * A particular texture has effect only if particular texture flag from
     * @ref RenderShader::Flag "Flag" is set, you can use @cpp nullptr @ce for the
     * rest. Expects that the shader was created with at least one of
     * @ref Flag::AmbientTexture, @ref Flag::DiffuseTexture or
     * @ref Flag::SpecularTexture enabled. More efficient than setting each
     * texture separately.
     * @see @ref bindAmbientTexture(), @ref bindDiffuseTexture(),
     *      @ref bindSpecularTexture()
     */
    RenderShader& bindTextures(GL::Texture2D* ambient, GL::Texture2D* diffuse, GL::Texture2D* specular);

    /**
     * @brief Set shininess
     * @return Reference to self (for method chaining)
     *
     * The larger value, the harder surface (smaller specular highlight).
     * Initial value is @cpp 80.0f @ce.
     */
    RenderShader& setShininess(Float shininess) {
        setUniform(_shininessUniform, shininess);
        return *this;
    }

    /**
     * @brief Set alpha mask value
     * @return Reference to self (for method chaining)
     *
     * Expects that the shader was created with @ref Flag::AlphaMask
     * enabled. Fragments with alpha values smaller than the mask value
     * will be discarded. Initial value is @cpp 0.5f @ce. See the flag
     * documentation for further information.
     */
    RenderShader& setAlphaMask(Float mask);

    /**
     * @brief Set transformation matrix
     * @return Reference to self (for method chaining)
     *
     * You need to set also @ref setNormalMatrix() with a corresponding
     * value. Initial value is an identity matrix.
     */
    RenderShader& setMeshToObjectMatrix(const Matrix4& meshToObject) {
        setUniform(_meshToObjectMatrixUniform, meshToObject);
        return *this;
    }

    /**
     * @brief Set world transformation matrix
     * @return Reference to self (for method chaining)
     *
     * Initial value is an identity matrix.
     */
    RenderShader& setObjectToCamMatrix(const Matrix4& matrix) {
        setUniform(_objectToCamMatrixUniform, matrix);
        return *this;
    }

    /**
     * @brief Set normal matrix
     * @return Reference to self (for method chaining)
     *
     * The matrix doesn't need to be normalized, as the renormalization
     * must be done in the shader anyway. You need to set also
     * @ref setTransformationMatrix() with a corresponding value. Initial
     * value is an identity matrix.
     */
    RenderShader& setNormalMatrix(const Matrix3x3& matrix) {
        setUniform(_normalMatrixUniform, matrix);
        return *this;
    }

    /**
     * @brief Set projection matrix
     * @return Reference to self (for method chaining)
     *
     * Initial value is an identity matrix (i.e., an orthographic
     * projection of the default @f$ [ -\boldsymbol{1} ; \boldsymbol{1} ] @f$
     * cube).
     */
    RenderShader& setProjectionMatrix(const Matrix4& matrix) {
        setUniform(_projectionMatrixUniform, matrix);
        return *this;
    }

    /**
     * @brief Set light position
     * @return Reference to self (for method chaining)
     *
     * Initial value is a zero vector --- that will in most cases cause the
     * object to be rendered black (or in the ambient color), as the light
     * is inside of it.
     */
    RenderShader& setLightPosition(const Vector3& light) {
        setUniform(_lightPositionUniform, light);
        return *this;
    }

    /**
     * @brief Set light color
     * @return Reference to self (for method chaining)
     *
     * Initial value is @cpp 0xffffffff_rgbaf @ce.
     */
    RenderShader& setLightColor(const Color4& color) {
        setUniform(_lightColorUniform, color);
        return *this;
    }

private:
    Flags _flags;
    Int _meshToObjectMatrixUniform{0},
        _objectToCamMatrixUniform{1},
        _projectionMatrixUniform{2},
        _normalMatrixUniform{3},
        _lightPositionUniform{4},
        _ambientColorUniform{5},
        _diffuseColorUniform{6},
        _specularColorUniform{7},
        _lightColorUniform{8},
        _shininessUniform{9},
        _alphaMaskUniform{10},
        _classIndexUniform{11},
        _instanceIndexUniform{12};
};

}

CORRADE_ENUMSET_OPERATORS(sl::RenderShader::Flags)

#endif

