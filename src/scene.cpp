// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/scene.h>

#include <stillleben/context.h>
#include <stillleben/object.h>
#include <stillleben/light_map.h>
#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>

#include <Corrade/Utility/ConfigurationGroup.h>
#include <Corrade/Utility/Format.h>

#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/ConfigurationValue.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

#include <sstream>

#include "physx_impl.h"

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

Scene::Scene(const std::shared_ptr<Context>& ctx, const ViewportSize& viewportSize)
 : m_ctx{ctx}
{
    // Every scene needs a camera
    const Rad FOV_X = Deg(58.0);

    m_cameraObject.setParent(&m_scene);
    (*(m_camera = new SceneGraph::Camera3D{m_cameraObject}))
        .setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::NotPreserved)
        .setViewport(viewportSize);

    setCameraFromFOV(FOV_X);

    std::random_device dev;
    m_randomGenerator.seed(dev());

    auto& physics = ctx->physxPhysics();

    m_physicsDispatcher.reset(physx::PxDefaultCpuDispatcherCreate(2));

    physx::PxSceneDesc desc(physics.getTolerancesScale());
    desc.gravity = physx::PxVec3(0.0f, 9.81f, 0.0f);
    desc.cpuDispatcher = m_physicsDispatcher.get();
    desc.filterShader = physx::PxDefaultSimulationFilterShader;

    m_physicsScene.reset(physics.createScene(desc));
}

Scene::~Scene()
{
    clearObjects();
}

void Scene::clearObjects()
{
    // First unset back-references to us, then release the shared_ptr
    for(auto& obj : m_objects)
    {
        obj->setParentSceneObject(nullptr);
        obj->setPhysicsScene(nullptr);
    }

    m_objects.clear();
}

void Scene::setCameraPose(const Magnum::Matrix4& pose)
{
    if(!pose.isRigidTransformation())
    {
        std::stringstream ss;
        ss << "Camera pose is not rigid:\n";
        Debug{&ss} << pose;
        throw std::invalid_argument{ss.str()};
    }

    m_cameraObject.setTransformation(pose);
}

void Scene::setCameraLookAt(const Magnum::Vector3& position, const Magnum::Vector3& lookAt, const Magnum::Vector3& up)
{
    // Magnum::Matrix4::lookAt assumes a -Z camera, so do things ourselves...
    const auto zAxis = (lookAt - position).normalized();
    const auto xAxis = Magnum::Math::cross(zAxis, up).normalized();
    const auto yAxis = Magnum::Math::cross(zAxis, xAxis).normalized();

    setCameraPose(
        Magnum::Matrix4::from(Magnum::Matrix3{xAxis, yAxis, zAxis}, position)
    );
}

Magnum::Matrix4 Scene::cameraPose() const
{
    return m_cameraObject.absoluteTransformationMatrix();
}

void Scene::setCameraIntrinsics(float fx, float fy, float cx, float cy)
{
    // Source: https://blog.noctua-software.com/opencv-opengl-projection-matrix.html

    // far and near
    constexpr float f = 10.0f;
    constexpr float n = 0.1;

    const float H = m_camera->viewport().y();
    const float W = m_camera->viewport().x();

    const float L = -cx * n / fx;
    const float R = (W-cx) * n / fx;
    const float T = -cy * n / fy;
    const float B = (H-cy) * n / fy;

    // Caution, this is column-major
    // We perform an ugly hack here: We keep X and Y directions, but flip Z
    // with respect to the usual OpenGL conventions (in line with usual
    // computer vision practice). While we in fact keep a right-handed
    // coordinate system all the way, OpenGL expects a left-handed NDC
    // coordinate system. That affects triangle winding order
    // (see render_pass.cpp)
    Matrix4 P{
        {2.0f*n/(R-L),         0.0f,                   0.0f, 0.0f},
        {        0.0f, 2.0f*n/(B-T),                   0.0f, 0.0f},
        { (R+L)/(L-R),  (T+B)/(T-B),            (f+n)/(f-n), 1.0f},
        {        0.0f,         0.0f, (2.0f * f * n) / (n-f), 0.0f}
    };

    m_camera->setProjectionMatrix(P);
}

void Scene::setCameraProjection(const Magnum::Matrix4& P)
{
    m_camera->setProjectionMatrix(P);
}

void Scene::setCameraFromFOV(Magnum::Rad fov)
{
    const float H = m_camera->viewport().y();
    const float W = m_camera->viewport().x();

    const float cx = W/2;
    const float cy = H/2;
    const float fx = W / (2.0 * Magnum::Math::tan(fov/2.0));
    const float fy = H / (2.0 * Magnum::Math::tan(fov/2.0));

    setCameraIntrinsics(fx, fy, cx, cy);
}

Magnum::Matrix4 Scene::projectionMatrix() const
{
    return m_camera->projectionMatrix();
}

void Scene::addObject(const std::shared_ptr<Object>& obj)
{
    m_objects.push_back(obj);

    obj->setParentSceneObject(&m_scene);
    obj->setPhysicsScene(m_physicsScene.get());

    // Automatically set the instance index if not set by the user already
    if(obj->instanceIndex() == 0)
        obj->setInstanceIndex(m_objects.size());
}

template<class Generator>
Magnum::Quaternion randomQuaternion(Generator& g)
{
    std::normal_distribution<float> normalDist;
    Quaternion q{
        Vector3{normalDist(g), normalDist(g), normalDist(g)},
        normalDist(g)
    };

    return q.normalized();
}

Magnum::Matrix4 Scene::cameraToWorld(const Magnum::Matrix4& poseInCamera) const
{
    return m_camera->cameraMatrix().invertedOrthogonal() * poseInCamera;
}

void Scene::setBackgroundImage(std::shared_ptr<Magnum::GL::RectangleTexture>& texture)
{
    m_backgroundImage = texture;
}

void Scene::setBackgroundColor(const Magnum::Color4& color)
{
    m_backgroundColor = color;
}

void Scene::drawPhysicsDebug()
{
    throw std::logic_error("Not implemented");
}

namespace
{
    class AutoCollisionFilter : public physx::PxQueryFilterCallback
    {
    public:
        explicit AutoCollisionFilter(const physx::PxActor* actor)
         : m_actor{actor}
        {}

        physx::PxQueryHitType::Enum postFilter(const physx::PxFilterData& filterData, const physx::PxQueryHit& hit) override
        {
            return physx::PxQueryHitType::eTOUCH;
        }

        physx::PxQueryHitType::Enum preFilter(const physx::PxFilterData& filterData, const physx::PxShape* shape, const physx::PxRigidActor* actor, physx::PxHitFlags& queryFlags) override
        {
            if(actor == m_actor)
                return physx::PxQueryHitType::eNONE;

            return physx::PxQueryHitType::eTOUCH;
        }
    private:
        const physx::PxActor* m_actor;
    };
}

bool Scene::isObjectColliding(Object& object)
{
    auto& body = object.rigidBody();

    std::vector<physx::PxShape*> shapes(body.getNbShapes());
    body.getShapes(shapes.data(), shapes.size());

    AutoCollisionFilter filter(&body);

    for(auto shape : shapes)
    {
        // The PhysX API is crappy here: be careful not to remove the copy
        // in the first line, and, conversely, don't copy in the second line.
        auto geometryHolder = shape->getGeometry();
        const auto& geometry = geometryHolder.any();

        auto pose = body.getGlobalPose() * shape->getLocalPose();

        physx::PxOverlapBuffer cb(nullptr, 0);

        bool contacts = m_physicsScene->overlap(geometry, pose, cb,
            physx::PxQueryFilterData(physx::PxQueryFlag::eDYNAMIC | physx::PxQueryFlag::eSTATIC | physx::PxQueryFlag::eANY_HIT | physx::PxQueryFlag::ePREFILTER),
            &filter
        );

        if(contacts)
            return true;
    }

    return false;
}

namespace
{
    template<class F>
    struct Caller
    {
        Caller(F&& f)
         : m_cb(f)
        {
        }

        Caller(Caller<F>&& other)
         : m_cb(std::move(other.m_cb))
        {}

        ~Caller()
        {
            m_cb();
        }
    private:
        F m_cb;
    };

    template<class F>
    [[nodiscard]] Caller<F> finally(F&& f)
    {
        return {std::move(f)};
    }
}

bool Scene::resolveCollisions()
{
//     constexpr int maxIterations = 40;
//
//     // Remove gravity
//     auto grav = m_physicsWorld->getGravity();
//     m_physicsWorld->setGravity(btVector3(0, 0, 0));
//
//     // Set it back when we exit
//     auto _ = finally([&](){m_physicsWorld->setGravity(grav);});
//
//     for(int i = 0; i < maxIterations; ++i)
//     {
//         m_physicsWorld->performDiscreteCollisionDetection();
//
//         int numContacts = 0;
//         {
//             auto* dispatcher = m_physicsWorld->getDispatcher();
//
//             int numManifolds = dispatcher->getNumManifolds();
//             for(int i = 0; i < numManifolds; ++i)
//             {
//                 auto* manifold = dispatcher->getManifoldByIndexInternal(i);
//
//                 numContacts += manifold->getNumContacts();
//             }
//         }
//
//         if(numContacts == 0)
//             return true;
//
//         m_physicsWorld->setInternalTickCallback(&Scene::constrainingTickCallback, this);
//         m_physicsWorld->stepSimulation(0.1, 1, 0.1);
//     }

    return false;
}

void Scene::setLightPosition(const Magnum::Vector3& position)
{
    m_lightPosition = position;
}

void Scene::setAmbientLight(const Magnum::Color3& color)
{
    m_ambientLight = color;
}

void Scene::chooseRandomLightPosition()
{
    // We want to have the light coming from above, but not from behind the
    // objects. We first determine the light position relative to the camera.

    Magnum::Vector3 meanPosition;
    for(auto& obj : m_objects)
    {
        meanPosition += (m_camera->cameraMatrix() * obj->pose()).translation();
    }
    if(!m_objects.empty())
        meanPosition /= m_objects.size();

    std::normal_distribution<float> normalDist;
    Magnum::Vector3 randomDirection = Magnum::Vector3{
        normalDist(m_randomGenerator),
        -std::abs(normalDist(m_randomGenerator)), // always from above
        -std::abs(normalDist(m_randomGenerator)) // always on camera side
    }.normalized();

    Magnum::Vector3 lightPositionInCam = meanPosition + 1000.0f * randomDirection;

    setLightPosition(m_camera->cameraMatrix().invertedRigid().transformPoint(lightPositionInCam));
}

void Scene::simulateTableTopScene(const std::function<void(int)>& visCallback)
{
    loadPhysics();

    // What kind of objects do we have in the scene?
    float maxDiameter = 0.0f;
    float minDistVis = 0.05f;
    for(auto& obj : m_objects)
    {
        float diameter = obj->mesh()->bbox().size().length();
        maxDiameter = std::max(maxDiameter, diameter);
        minDistVis = std::max(minDistVis, minimumDistanceForObjectDiameter(diameter));
    }

    // Choose a nice camera pose
    {
        // We want to look at the origin with a distance d.
        std::uniform_real_distribution<float> dDist{1.5f*minDistVis, 3.0f*minDistVis};

        float d = dDist(m_randomGenerator);

        // Columns!
        Magnum::Matrix3 camRot{
            -Magnum::Vector3::yAxis(),
            -Magnum::Vector3::zAxis(),
            Magnum::Vector3::xAxis()
        };

        Magnum::Matrix4 basePose = Magnum::Matrix4::from(camRot, Magnum::Vector3(-d, 0.0f, 0.0f));

        std::uniform_real_distribution<float> azimuthDist{-M_PI, M_PI};
        constexpr Magnum::Rad ELEVATION_LIMIT{Magnum::Deg{30.0f}};
        std::uniform_real_distribution<float> elevationDist{
            static_cast<float>(ELEVATION_LIMIT), M_PI/2.0f - static_cast<float>(ELEVATION_LIMIT)
        };

        Magnum::Matrix4 pose =
            Magnum::Matrix4::rotationZ(Magnum::Rad{azimuthDist(m_randomGenerator)}) *
            Magnum::Matrix4::rotationY(Magnum::Rad{elevationDist(m_randomGenerator)}) *
            basePose;

        setCameraPose(pose);
    }

    // Define the plane. It always goes through the origin.
    constexpr Magnum::Vector3 normal{0.0f, 0.0f, 1.0f};

    // Add it to the physics scene
    Vector3 xAxis = Vector3::xAxis();
    Vector3 zAxis = normal;
    Vector3 yAxis = Math::cross(zAxis, xAxis);

    Matrix3 rot{xAxis, yAxis, zAxis};

    Matrix4 T = Matrix4::from(rot, Vector3{});

    auto& physics = m_ctx->physxPhysics();

    constexpr Magnum::Vector3 BOX_SIZE{30.0, 30.0, 0.04};

    physx::PxBoxGeometry boxGeom{BOX_SIZE.x(), BOX_SIZE.y(), BOX_SIZE.z()};
    PhysXHolder<physx::PxMaterial> material{physics.createMaterial(0.5f, 0.5f, 0.0f)};
    PhysXHolder<physx::PxShape> shape{physics.createShape(boxGeom, *material, true)};
    PhysXHolder<physx::PxRigidStatic> actor{physics.createRigidStatic(physx::PxTransform{T})};
    actor->attachShape(*shape);

    // Call setBackgroundPlanePose() with the correct pose so that the visual
    // matches up with the physics simulation
    {
        std::uniform_real_distribution<float> planeRotDist(-M_PI, M_PI);
        Magnum::Rad planeRot{planeRotDist(m_randomGenerator)};

        Matrix4 topSidePlanePose = T * Matrix4::rotationZ(planeRot) * Matrix4::translation({0.0, 0.0, BOX_SIZE.z()/2});
        setBackgroundPlanePose(topSidePlanePose);
    }

    m_physicsScene->addActor(*actor);
    auto remover = finally([&](){ m_physicsScene->removeActor(*actor); });

    m_physicsScene->setGravity(physx::PxVec3{-9.81f * normal});

    // Arrange the objects randomly above the plane
    std::uniform_real_distribution<float> posDist{-2.0f*maxDiameter, 2.0f*maxDiameter};

    float z = 0.0f;
    for(auto& obj : m_objects)
    {
        z += maxDiameter;
        Magnum::Vector3 pos = z*normal /*+ Magnum::Vector3{
            posDist(m_randomGenerator), posDist(m_randomGenerator), posDist(m_randomGenerator)
        }*/;
        Magnum::Quaternion q = randomQuaternion(m_randomGenerator);

        Magnum::Matrix4 pose = Magnum::Matrix4::from(q.toMatrix(), pos);
        obj->setPose(pose);

        obj->rigidBody().setRigidBodyFlag(physx::PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
    }

    // We simulate a strong fake gravity towards the center
    const Vector3 gravityCenter = 0.1*normal;

    constexpr unsigned int FPS = 25;
    constexpr float TIME_PER_FRAME = 1.0f / FPS;
    constexpr unsigned int SUBSTEPS = 4;

    const int maxIterations = 100;
    for(int i = 0; i < maxIterations; ++i)
    {
        if(visCallback)
            visCallback(i);

        for(auto& obj : m_objects)
        {
            Magnum::Vector3 diff = gravityCenter - obj->pose().translation();

            Magnum::Vector3 dir = diff.normalized();

            float magnitude = 5.0f;
            if(diff.dot() < 0.05f*0.05f)
                magnitude = 0.5f * diff.length() / 0.05f;

            obj->rigidBody().addForce(physx::PxVec3{magnitude * dir});
        }

        for(unsigned int i = 0; i < SUBSTEPS; ++i)
        {
            m_physicsScene->simulate(TIME_PER_FRAME / SUBSTEPS);
            m_physicsScene->fetchResults(true);
        }

        for(auto& obj : m_objects)
        {
            obj->updateFromPhysics();
        }
    }

    for(auto& obj : m_objects)
        obj->rigidBody().wakeUp();

    // Simulate further with only gravity
    for(int i = 0; i < maxIterations; ++i)
    {
        if(visCallback)
            visCallback(maxIterations + i);

        for(auto& obj : m_objects)
        {
            Magnum::Vector3 dir = (gravityCenter - obj->pose().translation()).normalized();
            obj->rigidBody().clearForce();
            obj->rigidBody().addForce(physx::PxVec3{0.5f * dir});
        }

        for(unsigned int i = 0; i < SUBSTEPS; ++i)
        {
            m_physicsScene->simulate(TIME_PER_FRAME / SUBSTEPS);
            m_physicsScene->fetchResults(true);
        }

        for(auto& obj : m_objects)
        {
            obj->updateFromPhysics();
        }
    }
}

void Scene::serialize(Corrade::Utility::ConfigurationGroup& group) const
{
    group.setValue("projection", m_camera->projectionMatrix());

    auto cameraPose = m_cameraObject.transformationMatrix();
    group.setValue("cameraPosition", cameraPose.translation());
    group.setValue("cameraRotation", Quaternion::fromMatrix(cameraPose.rotationScaling()));

    group.setValue("lightPosition", m_lightPosition);
    group.setValue("ambientLight", m_ambientLight);
    group.setValue("numObjects", m_objects.size());

    for(std::size_t i = 0; i < m_objects.size(); ++i)
    {
        const auto& obj = m_objects[i];

        auto objGroup = group.addGroup("object");
        obj->serialize(*objGroup);
    }

    if(m_lightMap)
    {
        group.setValue("lightMap", m_lightMap->path());
    }

    group.setValue("backgroundPlanePose", m_backgroundPlanePose);
    group.setValue("backgroundPlaneSize", m_backgroundPlaneSize);
}

void Scene::deserialize(const Corrade::Utility::ConfigurationGroup& group, MeshCache* cache)
{
    if(group.hasValue("projection"))
        m_camera->setProjectionMatrix(group.value<Magnum::Matrix4>("projection"));

    if(group.hasValue("cameraPosition") && group.hasValue("cameraRotation"))
    {
        m_cameraObject.setTransformation(Matrix4::from(
            group.value<Magnum::Quaternion>("cameraRotation").toMatrix(),
            group.value<Magnum::Vector3>("cameraPosition")
        ));
    }

    if(group.hasValue("lightPosition"))
        m_lightPosition = group.value<Magnum::Vector3>("lightPosition");

    if(group.hasValue("ambientLight"))
        m_ambientLight = group.value<Magnum::Color3>("ambientLight");

    if(group.hasValue("lightMap"))
        m_lightMap = std::make_shared<LightMap>(group.value("lightMap"), m_ctx);

    if(group.hasValue("backgroundPlanePose"))
        m_backgroundPlanePose = group.value<Magnum::Matrix4>("backgroundPlanePose");
    if(group.hasValue("backgroundPlaneSize"))
        m_backgroundPlaneSize = group.value<Magnum::Vector2>("backgroundPlaneSize");

    std::unique_ptr<MeshCache> localCache;
    if(!cache)
    {
        localCache = std::make_unique<MeshCache>(m_ctx);
        cache = localCache.get();
    }

    clearObjects();

    auto objectGroups = group.groups("object");
    for(const auto& objectGroup : group.groups("object"))
    {
        auto obj = std::make_shared<Object>();
        obj->deserialize(*objectGroup, *cache);

        addObject(obj);
    }
}

void Scene::loadVisual()
{
    for(auto& obj : m_objects)
        obj->loadVisual();
}

void Scene::loadPhysics()
{
    for(auto& obj : m_objects)
        obj->loadPhysics();
}

void Scene::setLightMap(const std::shared_ptr<LightMap>& lightMap)
{
    m_lightMap = lightMap;
}

void Scene::setBackgroundPlanePose(const Matrix4& pose)
{
    m_backgroundPlanePose = pose;
}

void Scene::setBackgroundPlaneSize(const Vector2& size)
{
    m_backgroundPlaneSize = size;
}

void Scene::setBackgroundPlaneTexture(const std::shared_ptr<Magnum::GL::Texture2D>& texture)
{
    m_backgroundPlaneTexture = texture;
}

}
