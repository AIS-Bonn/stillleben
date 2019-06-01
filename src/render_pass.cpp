// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/mesh.h>
#include <stillleben/cuda_interop.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/DebugOutput.h>
#include <Magnum/Image.h>

#include <Magnum/MeshTools/Compile.h>

#include <Magnum/Primitives/Square.h>

#include <Magnum/PixelFormat.h>

#include <Magnum/Trade/MeshData2D.h>

#include "shaders/render_shader.h"
#include "shaders/resolve_shader.h"
#include "shaders/background_shader.h"

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

namespace
{
constexpr RenderShader::Flags flagsForType(RenderPass::Type type)
{
    switch(type)
    {
        case RenderPass::Type::Flat:
            return RenderShader::Flag::Flat;
        case RenderPass::Type::Phong:
            return {};
        default:
            return {};
    }
}

}

RenderPass::Result::Result()
 : rgb{m_mapper}
 , objectCoordinates{m_mapper}
 , classIndex{m_mapper}
 , instanceIndex{m_mapper}
 , normals{m_mapper}
 , validMask{m_mapper}
{
}

RenderPass::Result::~Result() = default;


void RenderPass::Result::mapCUDA()
{
#if HAVE_CUDA
    m_mapper.mapAll();
#endif
}

void RenderPass::Result::unmapCUDA()
{
#if HAVE_CUDA
    m_mapper.unmapAll();
#endif
}


RenderPass::RenderPass(Type type, bool cuda)
 : m_cuda{cuda}
 , m_framebuffer{Magnum::NoCreate}
 , m_resolvedBuffer{Magnum::NoCreate}
 , m_shaderTextured{std::make_unique<RenderShader>(flagsForType(type) | RenderShader::Flag::DiffuseTexture)}
 , m_shaderVertexColors{std::make_unique<RenderShader>(flagsForType(type) | RenderShader::Flag::VertexColors)}
 , m_shaderUniform{std::make_unique<RenderShader>(flagsForType(type))}
 , m_resolveShader{std::make_unique<ResolveShader>(m_msaa_factor)}
 , m_backgroundShader{std::make_unique<BackgroundShader>()}
{
    m_quadMesh = MeshTools::compile(Primitives::squareSolid(Primitives::SquareTextureCoords::DontGenerate));
    m_result = std::make_shared<Result>();
}

RenderPass::~RenderPass()
{
}

std::shared_ptr<RenderPass::Result> RenderPass::render(Scene& scene)
{
    scene.loadVisual();

    constexpr Color4 invalid{3000.0, 3000.0, 3000.0, 3000.0};

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);

    // Setup the framebuffer
    auto viewport = scene.viewport();

    if(!m_initialized || m_framebuffer.viewport().size() != scene.viewport())
    {
        m_result.reset();

        m_framebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};
        m_resolvedBuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

        m_msaa_rgb.setStorage(m_msaa_factor, GL::TextureFormat::RGBA8, viewport);
        m_msaa_depth.setStorage(m_msaa_factor, GL::TextureFormat::DepthComponent24, viewport);
        m_msaa_objectCoordinates.setStorage(m_msaa_factor, GL::TextureFormat::RGBA32F, viewport);

        // Note: We use float format here because of a bug in Mesa
        // https://bugs.freedesktop.org/show_bug.cgi?id=109057
        m_msaa_classIndex.setStorage(m_msaa_factor, GL::TextureFormat::R32F, viewport);
        m_msaa_instanceIndex.setStorage(m_msaa_factor, GL::TextureFormat::R32F, viewport);

        m_msaa_normal.setStorage(m_msaa_factor, GL::TextureFormat::RGBA32F, viewport);

        m_result = std::make_shared<Result>();

        m_result->rgb.setStorage(GL::TextureFormat::RGBA8, 4, viewport);
        m_result->objectCoordinates.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
        m_result->classIndex.setStorage(GL::TextureFormat::R16UI, 2, viewport);
        m_result->instanceIndex.setStorage(GL::TextureFormat::R16UI, 2, viewport);
        m_result->normals.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
        m_result->validMask.setStorage(GL::TextureFormat::R8UI, 1, viewport);

        m_initialized = true;
    }
    else
    {
        // Unmap from CUDA so that we can write into it
        m_result->unmapCUDA();
    }

    m_framebuffer
        .attachTexture(
            GL::Framebuffer::ColorAttachment{0},
            m_msaa_rgb
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{1},
            m_msaa_objectCoordinates
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{2},
            m_msaa_classIndex
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{3},
            m_msaa_instanceIndex
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{4},
            m_msaa_normal
        )
        .attachTexture(GL::Framebuffer::BufferAttachment::Depth, m_msaa_depth)
        .mapForDraw({
            {RenderShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}},
            {RenderShader::ObjectCoordinatesOutput, GL::Framebuffer::ColorAttachment{1}},
            {RenderShader::ClassIndexOutput, GL::Framebuffer::ColorAttachment{2}},
            {RenderShader::InstanceIndexOutput, GL::Framebuffer::ColorAttachment{3}},
            {RenderShader::NormalOutput, GL::Framebuffer::ColorAttachment{4}}
        })
    ;

    if(m_framebuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid MSAA framebuffer status:" << m_framebuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    m_framebuffer.bind();

    m_framebuffer.clear(GL::FramebufferClear::Depth);

    // Do we have a background texture?
    if(scene.backgroundImage())
    {
        m_backgroundShader->bindRGB(*scene.backgroundImage());
        m_quadMesh.draw(*m_backgroundShader);

        // Draw on top
        m_framebuffer.clear(GL::FramebufferClear::Depth);
    }
    else
    {
        m_framebuffer.clearColor(0, scene.backgroundColor());
    }

    m_framebuffer.clearColor(1, invalid);
    m_framebuffer.clearColor(2, Vector4ui{0});
    m_framebuffer.clearColor(3, Vector4ui{0});
    m_framebuffer.clearColor(4, 0x00000000_rgbaf);

    // Let the fun begin!
    for(auto& object : scene.objects())
    {
        Matrix4 objectToCam = scene.camera().object().absoluteTransformationMatrix().inverted() * object->pose();
        Matrix4 objectToCamInv = objectToCam.inverted();

        for(auto& shader : {std::ref(m_shaderTextured), std::ref(m_shaderUniform), std::ref(m_shaderVertexColors)})
        {
            (*shader.get())
                .setObjectToCamMatrix(objectToCam)
                .setClassIndex(object->mesh()->classIndex())
                .setInstanceIndex(object->instanceIndex())
            ;
        }

        object->draw(scene.camera(), [&](const Matrix4& meshToCam, SceneGraph::Camera3D& cam, Drawable* drawable) {
            Matrix4 meshToObject = objectToCamInv * meshToCam;

            if(drawable->texture())
            {
                (*m_shaderTextured)
                    .setLightPosition(scene.lightPosition())
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .bindDiffuseTexture(*drawable->texture())
                ;

                drawable->mesh().draw(*m_shaderTextured);
            }
            else if(drawable->hasVertexColors())
            {
                (*m_shaderVertexColors)
                    .setLightPosition(scene.lightPosition())
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .setDiffuseColor(drawable->color())
                ;

                drawable->mesh().draw(*m_shaderVertexColors);
            }
            else
            {
                (*m_shaderUniform)
                    .setLightPosition(scene.lightPosition())
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .setDiffuseColor(drawable->color())
                ;

                drawable->mesh().draw(*m_shaderUniform);
            }
        });
    }

    // Resolve the MSAA render buffers
    // For this purpose, we render a quad with a custom shader.

    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    m_resolvedBuffer
        .attachTexture(GL::Framebuffer::ColorAttachment{0}, m_result->rgb)
        .attachTexture(GL::Framebuffer::ColorAttachment{1}, m_result->objectCoordinates)
        .attachTexture(GL::Framebuffer::ColorAttachment{2}, m_result->classIndex)
        .attachTexture(GL::Framebuffer::ColorAttachment{3}, m_result->instanceIndex)
        .attachTexture(GL::Framebuffer::ColorAttachment{4}, m_result->normals)
        .attachTexture(GL::Framebuffer::ColorAttachment{5}, m_result->validMask)
        .mapForDraw({
            {ResolveShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}},
            {ResolveShader::ObjectCoordinatesOutput, GL::Framebuffer::ColorAttachment{1}},
            {ResolveShader::ClassIndexOutput, GL::Framebuffer::ColorAttachment{2}},
            {ResolveShader::InstanceIndexOutput, GL::Framebuffer::ColorAttachment{3}},
            {ResolveShader::NormalOutput, GL::Framebuffer::ColorAttachment{4}},
            {ResolveShader::ValidMaskOutput, GL::Framebuffer::ColorAttachment{5}}
        })
        .bind()
    ;

    m_resolvedBuffer.clearColor(0, 0x00000000_rgbaf);
    m_resolvedBuffer.clearColor(3, Vector4ui(1));
    m_resolvedBuffer.clearColor(4, 0x00000000_rgbaf);
    m_resolvedBuffer.clearColor(5, Vector4ui(6));

    if(m_resolvedBuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid output framebuffer status:" << m_resolvedBuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    (*m_resolveShader)
        .bindRGB(m_msaa_rgb)
        .bindCoordinates(m_msaa_objectCoordinates)
        .bindClassIndex(m_msaa_classIndex)
        .bindInstanceIndex(m_msaa_instanceIndex)
        .bindNormals(m_msaa_normal)
    ;

    m_quadMesh.draw(*m_resolveShader);

    m_result->unmapCUDA();

    return m_result;
}

}
