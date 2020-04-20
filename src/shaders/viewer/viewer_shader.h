// Visualize normals as RGB
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_SHADERS_NORMAL_SHADER_H
#define SL_SHADERS_NORMAL_SHADER_H

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/Shaders/Generic.h>

#include <stillleben/render_pass.h>

namespace sl
{

class ViewerShader : public Magnum::GL::AbstractShaderProgram
{
public:
    // Input attributes
    typedef Magnum::Shaders::Generic2D::Position Position;

    // Outputs
    enum: Magnum::UnsignedInt {
        ColorOutput = 0,
        NormalOutput = 1,
        SegmentationOutput = 2,
        CoordinateOutput = 3
    };

    /**
     * @brief Constructor
     */
    explicit ViewerShader(const std::shared_ptr<sl::Scene>& scene);

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
    explicit ViewerShader(Magnum::NoCreateT) noexcept
     : Magnum::GL::AbstractShaderProgram{Magnum::NoCreate} {}

    ~ViewerShader();

    /** @brief Copying is not allowed */
    ViewerShader(const ViewerShader&) = delete;

    /** @brief Move constructor */
    ViewerShader(ViewerShader&&) noexcept = default;

    /** @brief Copying is not allowed */
    ViewerShader& operator=(const ViewerShader&) = delete;

    /** @brief Move assignment */
    ViewerShader& operator=(ViewerShader&&) noexcept = default;

    void setData(RenderPass::Result& result);

private:
    std::shared_ptr<sl::Scene> m_scene;
    Magnum::Int m_uniform_bbox{0};
    Magnum::Int m_uniform_instanceColors{0};
};

}

#endif
