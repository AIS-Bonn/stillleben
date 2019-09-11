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
#include <functional>
#include <optional>

namespace physx
{
    class PxDefaultCpuDispatcher;
    class PxScene;
}

namespace sl
{

class Context;
class MeshCache;
class LightMap;

/**
 * @brief Scene
 *
 * This class represents an arrangement of objects. Scenes can be simulated
 * (yielding a different arrangement) or rendered (creating an image).
 *
 * Scenes can also be serialized and deserialized to/from an internal data
 * format. This functionality uses the @ref Corrade::Utility::Configuration
 * framework. The resulting format looks like this:
 *
 * @code{.ini}
 * # The OpenGL projection matrix
 * projection=<4x4 projection matrix>
 *
 * # The camera pose
 * cameraPose=<4x4 transformation matrix>
 *
 * # The light position
 * lightPosition=<3D vector>
 *
 * # The number of objects
 * numObjects=N
 *
 * # Description of the first object
 * [object]
 * mesh="/path/to/my/mesh.gltf"
 * pose=<4x4 transformation matrix>
 * instanceIndex=N
 *
 * # Description of the second object
 * [object]
 * ...
 * @endcode
 **/
class Scene
{
public:
    Scene(const std::shared_ptr<Context>& ctx, const ViewportSize& size);
    Scene(const Scene& other) = delete;
    Scene(Scene&& other) = delete;
    ~Scene();

    //! @name Serialization & Deserialization
    //@{

    void serialize(Corrade::Utility::ConfigurationGroup& group) const;
    void deserialize(const Corrade::Utility::ConfigurationGroup& group, MeshCache* cache = nullptr);

    void loadVisual();
    void loadPhysics();

    //@}

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

    //! @name Background scene
    //@{
    void setBackgroundImage(std::shared_ptr<Magnum::GL::RectangleTexture>& texture);
    const std::shared_ptr<Magnum::GL::RectangleTexture>& backgroundImage() const
    { return m_backgroundImage; }

    void setBackgroundColor(const Magnum::Color4& color);
    Magnum::Color4 backgroundColor() const
    { return m_backgroundColor; }
    //@}

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

    void clearObjects();

    //! @name Physics
    //@{
    void drawPhysicsDebug();

    using OrientationHint = Corrade::Containers::Optional<Magnum::Matrix3>;

    template<class Sampler>
    bool findNonCollidingPose(Object& object, Sampler& poseSampler, int maxIterations = 10);

    bool resolveCollisions();

    void simulateTableTopScene(const std::function<void(int)>& visCallback = {});
    //@}

    //! @name Lighting
    //@{
    void setLightPosition(const Magnum::Vector3& lightPosition);
    Magnum::Vector3 lightPosition() const
    { return m_lightPosition; }

    void chooseRandomLightPosition();

    void setAmbientLight(const Magnum::Color3& ambientLight);
    Magnum::Color3 ambientLight() const
    { return m_ambientLight; }

    void setLightMap(const std::shared_ptr<LightMap>& lightMap);
    std::shared_ptr<LightMap> lightMap() const
    { return m_lightMap; }
    //@}

private:
    bool isObjectColliding(Object& object);

    std::shared_ptr<Context> m_ctx;

    Scene3D m_scene;
    Object3D m_cameraObject;
    Magnum::SceneGraph::Camera3D* m_camera = nullptr;

    std::vector<std::shared_ptr<Object>> m_objects;

    std::mt19937 m_randomGenerator;

    std::shared_ptr<Magnum::GL::RectangleTexture> m_backgroundImage;
    Magnum::Color4 m_backgroundColor{1.0f, 1.0f, 1.0f, 1.0f};

    PhysXUnique<physx::PxDefaultCpuDispatcher> m_physicsDispatcher;
    PhysXUnique<physx::PxScene> m_physicsScene;

    Magnum::Vector3 m_lightPosition;
    Magnum::Color3 m_ambientLight;

    std::shared_ptr<LightMap> m_lightMap;
};

// IMPLEMENTATION

template<class Sampler>
bool Scene::findNonCollidingPose(Object& object, Sampler& poseSampler, int maxIterations)
{
    loadPhysics();

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
