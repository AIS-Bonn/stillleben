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

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

std::shared_ptr<Magnum::GL::RectangleTexture> PhongPass::render(Scene& scene)
{
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

    framebuffer.clearColor(0, 0x00000000_rgbaf);
    framebuffer.clear(GL::FramebufferClear::Depth);

    // Let the fun begin!
    for(auto& object : scene.objects())
    {
        object->draw(scene.camera(), [&](const Matrix4& transformationMatrix, SceneGraph::Camera3D& cam, Drawable* drawable){
            if(drawable->texture())
            {
                Debug{} << "Render with texture";
                Debug{} << transformationMatrix;
                Debug{} << cam.projectionMatrix();
                m_shaderTextured
                    .setLightPosition(cam.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}))
                    .setTransformationMatrix(transformationMatrix)
                    .setNormalMatrix(transformationMatrix.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .bindDiffuseTexture(*drawable->texture())
                ;

                drawable->mesh().draw(m_shaderTextured);
            }
            else
            {
                Debug{} << "Render uniform";
                m_shaderUniform
                    .setLightPosition(cam.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}))
                    .setTransformationMatrix(transformationMatrix)
                    .setNormalMatrix(transformationMatrix.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .setDiffuseColor(drawable->color())
                ;

                drawable->mesh().draw(m_shaderUniform);
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
