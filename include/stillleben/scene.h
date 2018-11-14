// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SCENE_H
#define STILLLEBEN_SCENE_H

#include <stillleben/math.h>

#include <memory>
#include <vector>

namespace sl
{

class Context;
class Object;


class Scene
{
public:
    Scene(const std::shared_ptr<Context>& ctx);
    Scene(const Scene& other) = delete;
    Scene(Scene&& other);
    ~Scene();

    void setCameraPose(const PoseMatrix& pose);
    PoseMatrix cameraPose() const;

    void setCameraIntrinsics(float fx, float fy, float cx, float cy);

    void addObject(const std::shared_ptr<Object>& object);
    const std::vector<std::shared_ptr<Object>>& objects() const;

private:
    class Private;
    std::unique_ptr<Private> m_d;
};

}

#endif
