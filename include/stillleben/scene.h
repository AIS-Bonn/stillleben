// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SCENE_H
#define STILLLEBEN_SCENE_H

#include <stillleben/math.h>
#include <stillleben/common.h>
#include <stillleben/object.h>
#include <stillleben/pose.h>

#include <Corrade/Containers/Optional.h>
#include <Corrade/Containers/StaticArray.h>
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

    std::shared_ptr<Context> context() const
    { return m_ctx; }

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
    void setCameraLookAt(const Magnum::Vector3& position, const Magnum::Vector3& lookAt, const Magnum::Vector3& up = Magnum::Vector3::zAxis());
    Magnum::Matrix4 cameraPose() const;
    void chooseRandomCameraPose();

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

    void setBackgroundPlanePose(const Magnum::Matrix4& pose);
    Magnum::Matrix4 backgroundPlanePose() const
    { return m_backgroundPlanePose; }

    void setBackgroundPlaneSize(const Magnum::Vector2& size);
    Magnum::Vector2 backgroundPlaneSize() const
    { return m_backgroundPlaneSize; }

    void setBackgroundPlaneTexture(const std::shared_ptr<Magnum::GL::Texture2D>& texture);
    std::shared_ptr<Magnum::GL::Texture2D> backgroundPlaneTexture() const
    { return m_backgroundPlaneTexture; }
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

    void removeObject(const std::shared_ptr<Object>& object);

    //! @name Physics
    //@{
    template<class Sampler>
    bool findNonCollidingPose(Object& object, Sampler& poseSampler, int maxIterations = 10);

    void simulateTableTopScene(const std::function<void(int)>& visCallback = {});

    void simulate(float dt);

    void checkCollisions();
    //@}

    //! @name Lighting
    //@{
    void setLightMap(const std::shared_ptr<LightMap>& lightMap);
    std::shared_ptr<LightMap> lightMap() const
    { return m_lightMap; }

    void setLightDirections(const Corrade::Containers::ArrayView<const Magnum::Vector3>& directions);
    void setLightDirections(const std::initializer_list<Magnum::Vector3>& directions)
    { setLightDirections(Corrade::Containers::arrayView(directions)); }
    Corrade::Containers::StaticArrayView<NumLights, Magnum::Vector3> lightDirections();

    void setLightColors(const Corrade::Containers::ArrayView<const Magnum::Color3>& colors);
    void setLightColors(const std::initializer_list<Magnum::Color3>& colors)
    { setLightColors(Corrade::Containers::arrayView(colors)); }
    Corrade::Containers::StaticArrayView<NumLights, Magnum::Color3> lightColors();

    [[deprecated("use chooseRandomLightDirection")]]
    void chooseRandomLightPosition()
    { chooseRandomLightDirection(); }

    void chooseRandomLightDirection();

    void setAmbientLight(const Magnum::Color3& ambientLight);
    Magnum::Color3 ambientLight() const
    { return m_ambientLight; }

    void setManualExposure(Magnum::Float exposure);
    Magnum::Float manualExposure() const
    { return m_manualExposure; }
    //@}

private:
    class SimulationCallback;

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
    std::unique_ptr<SimulationCallback> m_simCallback;

    Corrade::Containers::StaticArray<NumLights, Magnum::Vector3> m_lightDirections;
    Corrade::Containers::StaticArray<NumLights, Magnum::Color3> m_lightColors{
        Magnum::Color3{300.0f},
        Magnum::Color3{},
        Magnum::Color3{}
    };

    Magnum::Color3 m_ambientLight;

    std::shared_ptr<LightMap> m_lightMap;

    Magnum::Matrix4 m_backgroundPlanePose{Magnum::Math::IdentityInit};
    Magnum::Vector2 m_backgroundPlaneSize{};
    std::shared_ptr<Magnum::GL::Texture2D> m_backgroundPlaneTexture;

    Magnum::Float m_manualExposure = -1.0f;
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
