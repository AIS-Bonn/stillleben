// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_RENDER_PASS_H
#define STILLLEBEN_RENDER_PASS_H

#include <Magnum/GL/MultisampleTexture.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Framebuffer.h>

#include <stillleben/cuda_interop.h>

#include <memory>

namespace sl
{

class BackgroundShader;
class RenderShader;
class Scene;
class SSAOShader;
class SSAOApplyShader;

class RenderPass
{
public:
    enum class Type
    {
        Phong,
        Flat
    };

    explicit RenderPass(Type type = Type::Phong, bool cuda = false);
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

        [[deprecated("valid mask is not used anymore")]]
        CUDATexture validMask;

        CUDATexture camCoordinates;

    private:
        friend class RenderPass;

        void mapCUDA();
        void unmapCUDA();
    };

    void setSSAOEnabled(bool enabled);
    constexpr bool ssaoEnabled() const
    { return m_ssaoEnabled; }

    std::shared_ptr<Result> render(Scene& scene);
private:
    bool m_initialized = false;
    bool m_cuda;

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

    std::unique_ptr<SSAOShader> m_ssaoShader;
    std::unique_ptr<SSAOApplyShader> m_ssaoApplyShader;

    Magnum::GL::Mesh m_quadMesh;
    Magnum::GL::Mesh m_backgroundPlaneMesh;

    std::shared_ptr<Result> m_result;

    bool m_ssaoEnabled = true;
};

}

#endif
