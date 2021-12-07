// Simple Phong-based render pass for color image
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/phong_pass.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

std::shared_ptr<Magnum::GL::RectangleTexture> PhongPass::render(Scene& scene)
{
    scene.loadVisual();

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);

    // Setup the framebuffer
    // TODO: Is it worth it to recycle the framebuffer?
    auto viewport = scene.viewport();
    GL::Framebuffer framebuffer{Range2Di::fromSize({}, viewport)};

    m_msaa_rgb.setStorageMultisample(m_msaa_factor, GL::RenderbufferFormat::RGBA8, viewport);
    m_msaa_depth.setStorageMultisample(m_msaa_factor, GL::RenderbufferFormat::DepthComponent24, viewport);

    framebuffer
        .attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, m_msaa_rgb)
        .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, m_msaa_depth)
        .mapForDraw({
            {0, GL::Framebuffer::ColorAttachment{0}},
        })
    ;

    if(framebuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid MSAA framebuffer status:" << framebuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    framebuffer.bind();

    framebuffer.clearColor(0, 0x00000000_rgbaf);
    framebuffer.clear(GL::FramebufferClear::Depth);

    // Let the fun begin!
    for(auto& object : scene.objects())
    {
        object->draw(scene.camera(), [&](const Matrix4& transformationMatrix, SceneGraph::Camera3D& cam, Drawable* drawable){
            if(drawable->texture())
            {
                m_shaderTextured
                    .setLightPositions({Vector4{cam.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}), 1.0f}})
                    .setTransformationMatrix(transformationMatrix)
                    .setNormalMatrix(transformationMatrix.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .bindDiffuseTexture(*drawable->texture())
                ;

                m_shaderTextured.draw(drawable->mesh());
            }
            else
            {
                m_shaderUniform
                    .setLightPositions({Vector4{cam.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}), 1.0f}})
                    .setTransformationMatrix(transformationMatrix)
                    .setNormalMatrix(transformationMatrix.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .setDiffuseColor(drawable->color())
                ;

                m_shaderUniform.draw(drawable->mesh());
            }
        });
    }

    // Resolve the MSAA render buffers
    auto ret = std::make_shared<Magnum::GL::RectangleTexture>();

    GL::Framebuffer resolvedBuffer{Range2Di::fromSize({}, viewport)};

    ret->setStorage(GL::TextureFormat::RGBA8, viewport);

    resolvedBuffer.attachTexture(GL::Framebuffer::ColorAttachment{0}, *ret);

    framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{0});
    resolvedBuffer.mapForDraw(GL::Framebuffer::ColorAttachment{0});
    GL::AbstractFramebuffer::blit(framebuffer, resolvedBuffer,
        {{}, resolvedBuffer.viewport().size()}, GL::FramebufferBlit::Color);

    return ret;
}

}
