// Simple Phong-based render pass for color image
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_PHONG_PASS_H
#define STILLLEBEN_PHONG_PASS_H

#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Shaders/PhongGL.h>

#include <memory>

namespace sl
{

class Scene;

class PhongPass
{
public:
    std::shared_ptr<Magnum::GL::RectangleTexture> render(Scene& scene);
private:
    unsigned int m_msaa_factor = 8;
    Magnum::GL::Renderbuffer m_msaa_rgb;
    Magnum::GL::Renderbuffer m_msaa_depth;

    Magnum::Shaders::PhongGL m_shaderTextured{Magnum::Shaders::PhongGL::Flag::DiffuseTexture};
    Magnum::Shaders::PhongGL m_shaderUniform;
};

}

#endif
