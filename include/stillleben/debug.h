// Debug visualizations
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_DEBUG_H
#define STILLLEBEN_DEBUG_H

#include <Magnum/GL/RectangleTexture.h>

#include <memory>

namespace sl
{

class Scene;

Magnum::GL::RectangleTexture renderDebugImage(Scene& scene);
// Magnum::GL::RectangleTexture renderPhysicsDebugImage(Scene& scene);

}

#endif
