// Simple Phong-based render pass for color image
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_PHONG_PASS_H
#define STILLLEBEN_PHONG_PASS_H

#include <stillleben/render_pass.h>

#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/Shaders/Phong.h>

namespace sl
{

class PhongPass : public RenderPass
{
public:
    std::shared_ptr<RenderBuffer> render(Scene& scene);
private:
    unsigned int m_msaa_factor = 8;
    Magnum::GL::Renderbuffer m_msaa_rgb;
    Magnum::GL::Renderbuffer m_msaa_depth;

    Magnum::Shaders::Phong m_shaderTextured{Magnum::Shaders::Phong::Flag::DiffuseTexture};
    Magnum::Shaders::Phong m_shaderUniform;
};

}

#endif
