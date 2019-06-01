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
class ResolveShader;
class Scene;

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
        CUDATexture validMask;

    private:
        friend class RenderPass;

        void mapCUDA();
        void unmapCUDA();
    };

    std::shared_ptr<Result> render(Scene& scene);
private:
    bool m_initialized = false;
    bool m_cuda;

    Magnum::GL::Framebuffer m_framebuffer;
    Magnum::GL::Framebuffer m_resolvedBuffer;

    unsigned int m_msaa_factor = 4;
    Magnum::GL::MultisampleTexture2D m_msaa_rgb;
    Magnum::GL::MultisampleTexture2D m_msaa_depth;
    Magnum::GL::MultisampleTexture2D m_msaa_objectCoordinates;
    Magnum::GL::MultisampleTexture2D m_msaa_classIndex;
    Magnum::GL::MultisampleTexture2D m_msaa_instanceIndex;
    Magnum::GL::MultisampleTexture2D m_msaa_normal;

    std::unique_ptr<RenderShader> m_shaderTextured;
    std::unique_ptr<RenderShader> m_shaderVertexColors;
    std::unique_ptr<RenderShader> m_shaderUniform;

    std::unique_ptr<ResolveShader> m_resolveShader;

    std::unique_ptr<BackgroundShader> m_backgroundShader;

    Magnum::GL::Mesh m_quadMesh;

    std::shared_ptr<Result> m_result;
};

}

#endif
