// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_RENDER_PASS_H
#define STILLLEBEN_RENDER_PASS_H

#include <Magnum/GL/MultisampleTexture.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/Shaders/MeshVisualizer.h>

#include <stillleben/cuda_interop.h>

#include <functional>
#include <memory>

namespace sl
{

class BackgroundShader;
class BackgroundCubeShader;
class RenderShader;
class Scene;
class SSAOShader;
class SSAOApplyShader;
class Object;

class RenderPass
{
public:
    enum class Type
    {
        PBR,
        Phong,
        Flat
    };

    explicit RenderPass(Type type = Type::PBR, bool cuda = false);
    ~RenderPass();

    struct Result
    {
        explicit Result(bool cuda);
        ~Result();

    private:
        CUDAMapper m_mapper;

    public:
        CUDATexture rgb;
        CUDATexture objectCoordinates;
        CUDATexture classIndex;
        CUDATexture instanceIndex;
        CUDATexture normals;

        CUDATexture vertexIndex;
        CUDATexture barycentricCoeffs;

        CUDATexture camCoordinates;

        constexpr bool isMapped() const
        { return m_mapped; }

    private:
        friend class RenderPass;

        void mapCUDA();
        void unmapCUDA();

        bool m_mapped = false;
    };

    void setSSAOEnabled(bool enabled);
    constexpr bool ssaoEnabled() const
    { return m_ssaoEnabled; }

    void setDrawPhysicsEnabled(bool enabled);
    constexpr bool drawPhysicsEnabled() const
    { return m_drawPhysics; }

    enum class DrawBounding
    {
        Disabled,
        Boxes,
        Spheres
    };

    void setDrawBounding(DrawBounding drawBounding);
    constexpr DrawBounding drawBounding() const
    { return m_drawBounding; }

    using DrawPredicate = std::function<bool(const std::shared_ptr<sl::Object>)>;

    std::shared_ptr<Result> render(Scene& scene, const std::shared_ptr<Result>& result = {}, RenderPass::Result* depthBufferResult = nullptr, const DrawPredicate& drawPredicate = {});

    Type type() const
    { return m_type; }
private:
    bool m_initialized = false;
    bool m_cuda;

    Type m_type;

    Magnum::GL::Framebuffer m_framebuffer;
    Magnum::GL::Renderbuffer m_depthbuffer;

    Magnum::GL::Framebuffer m_ssaoFramebuffer;
    Magnum::GL::RectangleTexture m_ssaoTexture;

    Magnum::GL::Framebuffer m_ssaoApplyFramebuffer;
    Magnum::GL::RectangleTexture m_ssaoRGBInputTexture;

    std::unique_ptr<RenderShader> m_shaderTextured;
    std::unique_ptr<RenderShader> m_shaderVertexColors;
    std::unique_ptr<RenderShader> m_shaderUniform;

    std::unique_ptr<BackgroundShader> m_backgroundShader;
    std::unique_ptr<BackgroundCubeShader> m_backgroundCubeShader;

    std::unique_ptr<SSAOShader> m_ssaoShader;
    std::unique_ptr<SSAOApplyShader> m_ssaoApplyShader;

    std::unique_ptr<Magnum::Shaders::MeshVisualizer3D> m_meshShader;

    Magnum::GL::Mesh m_quadMesh;
    Magnum::GL::Mesh m_cubeMesh;
    Magnum::GL::Mesh m_backgroundPlaneMesh;
    Magnum::GL::Mesh m_sphereMesh;

    std::shared_ptr<Result> m_result;

    Magnum::GL::RectangleTexture m_zeroMinDepth;

    bool m_ssaoEnabled = true;
    bool m_drawPhysics = false;
    DrawBounding m_drawBounding = DrawBounding::Disabled;
};

}

#endif
