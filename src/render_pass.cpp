// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>

#include <Magnum/MeshTools/Compile.h>

#include <Magnum/Primitives/Square.h>

#include <Magnum/Trade/MeshData2D.h>

#include "shaders/render_shader.h"
#include "shaders/resolve_shader.h"

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

RenderPass::RenderPass()
 : m_shaderTextured{std::make_unique<RenderShader>(RenderShader::Flag::DiffuseTexture)}
 , m_shaderUniform{std::make_unique<RenderShader>()}
 , m_resolveShader{std::make_unique<ResolveShader>(m_msaa_factor)}
{
    m_quadMesh = MeshTools::compile(Primitives::squareSolid(Primitives::SquareTextureCoords::DontGenerate));
}

RenderPass::~RenderPass()
{
}

std::shared_ptr<RenderPass::Result> RenderPass::render(Scene& scene)
{
    constexpr Color4 invalid{-3000.0, -3000.0, -3000.0, -3000.0};

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);

    // Setup the framebuffer
    // TODO: Is it worth it to recycle the framebuffer?
    auto viewport = scene.viewport();
    GL::Framebuffer framebuffer{Range2Di::fromSize({}, viewport)};

    m_msaa_rgb.setStorage(m_msaa_factor, GL::TextureFormat::RGBA8, viewport);
    m_msaa_depth.setStorage(m_msaa_factor, GL::TextureFormat::DepthComponent24, viewport);
    m_msaa_objectCoordinates.setStorage(m_msaa_factor, GL::TextureFormat::RGBA32F, viewport);
    m_msaa_classIndex.setStorage(m_msaa_factor, GL::TextureFormat::R16UI, viewport);
    m_msaa_instanceIndex.setStorage(m_msaa_factor, GL::TextureFormat::R16UI, viewport);

    framebuffer
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
        .attachTexture(GL::Framebuffer::BufferAttachment::Depth, m_msaa_depth)
        .mapForDraw({
            {RenderShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}},
            {RenderShader::ObjectCoordinatesOutput, GL::Framebuffer::ColorAttachment{1}},
            {RenderShader::ClassIndexOutput, GL::Framebuffer::ColorAttachment{2}},
            {RenderShader::InstanceIndexOutput, GL::Framebuffer::ColorAttachment{3}},
        })
    ;

    if(framebuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid MSAA framebuffer status:" << framebuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    framebuffer.bind();

    framebuffer.clearColor(0, 0x00000000_rgbaf);
    framebuffer.clearColor(1, invalid);
    framebuffer.clearColor(2, Vector4ui(-1, -1, -1, -1));
    framebuffer.clearColor(3, Vector4ui(-1, -1, -1, -1));
    framebuffer.clear(GL::FramebufferClear::Depth);

    // Let the fun begin!
    for(auto& object : scene.objects())
    {
        Matrix4 objectToCam = scene.camera().object().absoluteTransformationMatrix().inverted() * object->pose();
        Matrix4 objectToCamInv = objectToCam.inverted();

        m_shaderTextured->setObjectToCamMatrix(objectToCam);
        m_shaderUniform->setObjectToCamMatrix(objectToCam);

        object->draw(scene.camera(), [&](const Matrix4& meshToCam, SceneGraph::Camera3D& cam, Drawable* drawable) {
            Matrix4 meshToObject = objectToCamInv * meshToCam;

            if(drawable->texture())
            {
                (*m_shaderTextured)
                    .setLightPosition(cam.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}))
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .bindDiffuseTexture(*drawable->texture())
                ;

                drawable->mesh().draw(*m_shaderTextured);
            }
            else
            {
                (*m_shaderUniform)
                    .setLightPosition(cam.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}))
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

    auto ret = std::make_shared<Result>();

    GL::Framebuffer resolvedBuffer{Range2Di::fromSize({}, viewport)};

    GL::Renderbuffer depthBuffer;
    depthBuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, viewport);

    ret->rgb.setStorage(GL::TextureFormat::RGBA8, viewport);
    ret->objectCoordinates.setStorage(GL::TextureFormat::RGBA32F, viewport);
    ret->classIndex.setStorage(GL::TextureFormat::R16UI, viewport);
    ret->instanceIndex.setStorage(GL::TextureFormat::R16UI, viewport);
    ret->validMask.setStorage(GL::TextureFormat::R8UI, viewport);

    resolvedBuffer
        .attachTexture(GL::Framebuffer::ColorAttachment{0}, ret->rgb)
        .attachTexture(GL::Framebuffer::ColorAttachment{1}, ret->objectCoordinates)
        .attachTexture(GL::Framebuffer::ColorAttachment{2}, ret->classIndex)
        .attachTexture(GL::Framebuffer::ColorAttachment{3}, ret->instanceIndex)
        .attachTexture(GL::Framebuffer::ColorAttachment{4}, ret->validMask)
        .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depthBuffer)
        .mapForDraw({
            {ResolveShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}},
            {ResolveShader::ObjectCoordinatesOutput, GL::Framebuffer::ColorAttachment{1}},
            {ResolveShader::ClassIndexOutput, GL::Framebuffer::ColorAttachment{2}},
            {ResolveShader::InstanceIndexOutput, GL::Framebuffer::ColorAttachment{3}},
            {ResolveShader::ValidMaskOutput, GL::Framebuffer::ColorAttachment{4}}
        })
        .bind()
    ;

    resolvedBuffer.clearColor(0, 0x00000000_rgbaf);
    resolvedBuffer.clearColor(4, Vector4ui(0));

    if(resolvedBuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid output framebuffer status:" << resolvedBuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    (*m_resolveShader)
        .bindRGB(m_msaa_rgb)
        .bindCoordinates(m_msaa_objectCoordinates)
        .bindClassIndex(m_msaa_classIndex)
        .bindInstanceIndex(m_msaa_instanceIndex)
    ;

    m_quadMesh.draw(*m_resolveShader);

    return ret;
}

}
