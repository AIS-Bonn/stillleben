// Abstract base class for a render pass: Defines how the scene is rendered
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_RENDER_PASS_H
#define STILLLEBEN_RENDER_PASS_H

#include <stillleben/render_buffer.h>
#include <stillleben/scene.h>

namespace sl
{

class RenderPass
{
public:
    virtual std::shared_ptr<RenderBuffer> render(Scene& scene) = 0;
};

}

#endif
