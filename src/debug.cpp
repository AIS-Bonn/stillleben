// Debug visualizations
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/debug.h>
#include <stillleben/object.h>
#include <stillleben/scene.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

std::shared_ptr<GL::RectangleTexture> renderDebugImage(Scene& scene)
{
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);

    auto viewport = scene.viewport();
    GL::Framebuffer framebuffer{Range2Di::fromSize({}, viewport)};

    auto texture = std::make_shared<GL::RectangleTexture>();
    texture->setStorage(GL::TextureFormat::RGBA8, viewport);

    GL::Renderbuffer depthBuffer;
    depthBuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, viewport);

    framebuffer
        .attachTexture(GL::Framebuffer::ColorAttachment{0}, *texture)
        .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depthBuffer)
        .mapForDraw({
            {0, GL::Framebuffer::ColorAttachment{0}}
        })
    ;

    if(framebuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid framebuffer status:" << framebuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    framebuffer.bind();

    framebuffer.clearColor(0, 0x00000000_rgbaf);
    framebuffer.clear(GL::FramebufferClear::Depth);

    for(auto& object : scene.objects())
    {
        scene.camera().draw(object->debugDrawables());
    }

    return texture;
}

}
