// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SCENE_H
#define STILLLEBEN_SCENE_H

#include <stillleben/math.h>
#include <stillleben/common.h>
#include <stillleben/object.h>
#include <stillleben/pose.h>

#include <Corrade/Containers/Optional.h>
#include <Magnum/SceneGraph/Camera.h>

#include <memory>
#include <vector>
#include <random>

class btDiscreteDynamicsWorld;
class btDynamicsWorld;

namespace Magnum
{
namespace BulletIntegration { class DebugDraw; }
}

namespace sl
{

class Context;

class Scene
{
public:
    Scene(const std::shared_ptr<Context>& ctx, const ViewportSize& size);
    Scene(const Scene& other) = delete;
    Scene(Scene&& other) = delete;
    ~Scene();

    //! @name Camera and viewport settings
    //@{
    void setCameraPose(const Magnum::Matrix4& pose);
    Magnum::Matrix4 cameraPose() const;

    void setCameraIntrinsics(float fx, float fy, float cx, float cy);
    void setCameraProjection(const Magnum::Matrix4& projection);
    void setCameraFromFOV(Magnum::Rad fov);

    Magnum::Matrix4 projectionMatrix() const;

    ViewportSize viewport() const
    { return m_camera->viewport(); }

    Magnum::SceneGraph::Camera3D& camera()
    { return *m_camera; }
    //@}

    void setBackgroundImage(std::shared_ptr<Magnum::GL::RectangleTexture>& texture);
    const std::shared_ptr<Magnum::GL::RectangleTexture>& backgroundImage() const
    { return m_backgroundImage; }

    //! @name Object placement
    //@{
    Magnum::Matrix4 cameraToWorld(const Magnum::Matrix4& poseInCamera) const;

    Magnum::Matrix4 placeObjectRandomly(
        float diameter, float minSizeFactor = pose::DEFAULT_MIN_SIZE_FACTOR)
    {
        pose::RandomPositionSampler posSampler{projectionMatrix(), diameter};
        posSampler.setMinSizeFactor(minSizeFactor);

        pose::RandomPoseSampler sampler{posSampler};
        return sampler(m_randomGenerator);
    }

    float minimumDistanceForObjectDiameter(float diameter) const
    { return pose::minimumDistanceForObjectDiameter(diameter, projectionMatrix()); }
    //@}

    void addObject(const std::shared_ptr<Object>& object);
    const std::vector<std::shared_ptr<Object>>& objects() const
    { return m_objects; }

    //! @name Physics
    //@{
    void drawPhysicsDebug();

    bool performCollisionCheck() const;

    using OrientationHint = Corrade::Containers::Optional<Magnum::Matrix3>;

    template<class Sampler>
    bool findNonCollidingPose(Object& object, Sampler& poseSampler, int maxIterations = 10);

    bool resolveCollisions();
    //@}

    //! @name Lighting
    //@{
    void setLightPosition(const Magnum::Vector3& lightPosition);
    Magnum::Vector3 lightPosition() const
    { return m_lightPosition; }

    void chooseRandomLightPosition();
    //@}

private:
    struct BulletStuff;

    static void constrainingTickCallback(btDynamicsWorld* world, float timeStep);
    bool isObjectColliding(Object& object);

    std::shared_ptr<Context> m_ctx;

    Scene3D m_scene;
    Object3D m_cameraObject;
    Magnum::SceneGraph::Camera3D* m_camera = nullptr;

    std::vector<std::shared_ptr<Object>> m_objects;

    std::mt19937 m_randomGenerator;

    std::shared_ptr<Magnum::GL::RectangleTexture> m_backgroundImage;

    std::unique_ptr<BulletStuff> m_bulletStuff;
    std::unique_ptr<btDiscreteDynamicsWorld> m_physicsWorld;

    // Keep Bullet dependency isolated -> unique_ptr here
    std::unique_ptr<Magnum::BulletIntegration::DebugDraw> m_physicsDebugDraw;

    Magnum::Vector3 m_lightPosition;
};

// IMPLEMENTATION

template<class Sampler>
bool Scene::findNonCollidingPose(Object& object, Sampler& poseSampler, int maxIterations)
{
    for(int i = 0; i < maxIterations; ++i)
    {
        // Sample new pose
        object.setPose(poseSampler(m_randomGenerator));

        // Check if collides with other objects
        if(!isObjectColliding(object))
            return true; // success!
    }

    return false;
}

}

#endif
