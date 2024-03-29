// Interactive scene viewer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_VIEWER_H
#define STILLLEBEN_VIEWER_H

#include <stillleben/render_pass.h>

#include <memory>

namespace sl
{

class Scene;

class Viewer
{
public:
    explicit Viewer(const std::shared_ptr<Scene>& scene);
    ~Viewer();

    // no funny stuff
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    std::shared_ptr<Scene> scene() const;

    void run();

    void drawFrame();

    static void view(const std::shared_ptr<Scene>& scene);

private:
    class Private;

    void setup();
    void draw();

    bool mainLoopIteration();

    std::unique_ptr<Private> m_d;
};

}

#endif
