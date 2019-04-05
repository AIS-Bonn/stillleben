// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/scene.h>

#include <stillleben/object.h>
#include <stillleben/mesh.h>

#include <btBulletDynamicsCommon.h>

#include <Magnum/BulletIntegration/DebugDraw.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

struct Scene::BulletStuff
{
    std::unique_ptr<btDbvtBroadphase> broadphase;
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfig;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver;
};

Scene::Scene(const std::shared_ptr<Context>& ctx, const ViewportSize& viewportSize)
 : m_ctx{ctx}
 , m_bulletStuff{std::make_unique<BulletStuff>()}
 , m_physicsDebugDraw{std::make_unique<BulletIntegration::DebugDraw>()}
{
    // Every scene needs a camera
    const Rad FOV_X = Deg(58.0);

    m_cameraObject.setParent(&m_scene);
    (*(m_camera = new SceneGraph::Camera3D{m_cameraObject}))
        .setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setViewport(viewportSize);

    setCameraFromFOV(FOV_X);

    std::random_device dev;
    m_randomGenerator.seed(dev());

    m_bulletStuff->broadphase = std::make_unique<btDbvtBroadphase>();
    m_bulletStuff->collisionConfig = std::make_unique<btDefaultCollisionConfiguration>();
    m_bulletStuff->dispatcher = std::make_unique<btCollisionDispatcher>(m_bulletStuff->collisionConfig.get());
    m_bulletStuff->solver = std::make_unique<btSequentialImpulseConstraintSolver>();
    m_physicsWorld = std::make_unique<btDiscreteDynamicsWorld>(
        m_bulletStuff->dispatcher.get(),
        m_bulletStuff->broadphase.get(),
        m_bulletStuff->solver.get(),
        m_bulletStuff->collisionConfig.get()
    );

    m_physicsWorld->setGravity({0.0, 0.0, -10.0f});

    m_physicsDebugDraw->setMode(
        BulletIntegration::DebugDraw::Mode::DrawWireframe
        | BulletIntegration::DebugDraw::Mode::DrawFrames
    );
    m_physicsWorld->setDebugDrawer(m_physicsDebugDraw.get());
}

Scene::~Scene()
{
    // The SceneObject destructor of this instance will delete child objects,
    // but they are reference counted using shared_ptr => first release them
    for(auto& obj : m_objects)
    {
        obj->setParentSceneObject(nullptr);
        obj->setPhysicsWorld(nullptr);
    }
}

void Scene::setCameraPose(const Magnum::Matrix4& pose)
{
    m_cameraObject.setTransformation(pose);
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
    constexpr float n = 0.01;

    const float H = m_camera->viewport().y();
    const float W = m_camera->viewport().x();

    const float L = -cx * n / fx;
    const float R = (W-cx) * n / fx;
    const float T = -cy * n / fy;
    const float B = (H-cy) * n / fy;

    // Caution, this is column-major
    Matrix4 P{
        {2.0f*n/(R-L), 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f*n/(B-T), 0.0f, 0.0f},
        {(R+L)/(L-R), (T+B)/(T-B), (f+n)/(f-n), 1.0f},
        {0.0f, 0.0f, (2.0f * f * n) / (n-f), 0.0f}
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
    obj->setPhysicsWorld(m_physicsWorld.get());

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

void Scene::drawPhysicsDebug()
{
    m_physicsDebugDraw->setTransformationProjectionMatrix(
        m_camera->projectionMatrix() * m_camera->cameraMatrix()
    );
    m_physicsWorld->debugDrawWorld();
}

bool Scene::performCollisionCheck() const
{
    m_physicsWorld->performDiscreteCollisionDetection();

    auto* dispatcher = m_physicsWorld->getDispatcher();

    int numManifolds = dispatcher->getNumManifolds();
    int numContacts = 0;
    for(int i = 0; i < numManifolds; ++i)
    {
        auto* manifold = dispatcher->getManifoldByIndexInternal(i);

        numContacts += manifold->getNumContacts();
    }

    Debug{} << "performCollisionCheck: found" << numContacts << "in" << numManifolds << "manifolds.";

    return numContacts != 0;
}

// Ancient C++
namespace
{
    struct CollisionCallback : public btCollisionWorld::ContactResultCallback
    {
        btScalar addSingleResult(btManifoldPoint &, const btCollisionObjectWrapper *, int, int, const btCollisionObjectWrapper *, int, int) override
        {
            m_numContacts++;
            return 0; // apparently unused (*argh*)
        }

        inline int numContacts() const
        { return m_numContacts; }
    private:
        int m_numContacts = 0;
    };
}

bool Scene::isObjectColliding(Object& object)
{
    // Check if collides with other objects
    CollisionCallback counter;
    m_physicsWorld->contactTest(&object.rigidBody(), counter);

    return counter.numContacts() != 0;
}

constexpr float ANGULAR_VELOCITY_LIMIT = 20.0 / 180.0 * M_PI;
constexpr float LINEAR_VELOCITY_LIMIT = 0.1;

void Scene::constrainingTickCallback(btDynamicsWorld* world, float timeStep)
{
    Scene* self = reinterpret_cast<Scene*>(world->getWorldUserInfo());

    for(auto& obj : self->m_objects)
    {
        auto& rigidBody = obj->rigidBody();
        rigidBody.setDamping(0.5, 0.5);

//	Debug{} << "Object" << (i++);
        btVector3 angularVelocity = rigidBody.getAngularVelocity();
//	Debug{} << "angular:" << angularVelocity.getX() << angularVelocity.getY() << angularVelocity.getZ();

        float angVelNorm = angularVelocity.norm();
        if(angVelNorm > ANGULAR_VELOCITY_LIMIT)
            angularVelocity = angularVelocity / angVelNorm * ANGULAR_VELOCITY_LIMIT;

        rigidBody.setAngularVelocity(angularVelocity);

        btVector3 linearVelocity = rigidBody.getLinearVelocity();
//	Debug{} << "linear:" << linearVelocity.getX() << linearVelocity.getY() << linearVelocity.getZ();

        float linVelNorm = linearVelocity.norm();
        if(linVelNorm > LINEAR_VELOCITY_LIMIT)
            linearVelocity = linearVelocity / linVelNorm * LINEAR_VELOCITY_LIMIT;

        rigidBody.setLinearVelocity(linearVelocity);
    }
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
    constexpr int maxIterations = 40;

    // Remove gravity
    auto grav = m_physicsWorld->getGravity();
    m_physicsWorld->setGravity(btVector3(0, 0, 0));

    // Set it back when we exit
    auto _ = finally([&](){m_physicsWorld->setGravity(grav);});

    for(int i = 0; i < maxIterations; ++i)
    {
        m_physicsWorld->performDiscreteCollisionDetection();

        int numContacts = 0;
        {
            auto* dispatcher = m_physicsWorld->getDispatcher();

            int numManifolds = dispatcher->getNumManifolds();
            for(int i = 0; i < numManifolds; ++i)
            {
                auto* manifold = dispatcher->getManifoldByIndexInternal(i);

                numContacts += manifold->getNumContacts();
            }
        }

        if(numContacts == 0)
            return true;

        m_physicsWorld->setInternalTickCallback(&Scene::constrainingTickCallback, this);
        m_physicsWorld->stepSimulation(0.1, 1, 0.1);
    }

    return false;
}

void Scene::setLightPosition(const Magnum::Vector3& position)
{
    m_lightPosition = position;
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

    setLightPosition(m_camera->cameraMatrix().invertedOrthogonal().transformPoint(lightPositionInCam));
}

void Scene::simulateTableTopScene(const std::function<void(int)>& visCallback)
{
    // Choose a plane normal. We want it to lie between [0 -1 0] and [0 0 -1].
    std::uniform_real_distribution<float> angleDist(30.0*M_PI/180.0, M_PI/2.0 - 30.0*M_PI/180.0);
    Magnum::Rad angle{angleDist(m_randomGenerator)};

    Magnum::Vector3 normal{0.0f, -Math::sin(angle), -Math::cos(angle)};

    // The plane always goes through a point [0 0 d].
    float minDistVis = 0.1f;
    float maxDiameter = 0.0f;
    for(auto& obj : m_objects)
    {
        float diameter = obj->mesh()->bbox().size().length();
        maxDiameter = std::max(maxDiameter, diameter);
        minDistVis = std::max(minDistVis, minimumDistanceForObjectDiameter(diameter));
    }
    std::uniform_real_distribution<float> dDist{0.8f*minDistVis, 4.0f*minDistVis};

    Magnum::Vector3 p{0.0f, 0.0f, dDist(m_randomGenerator)};

    float planeConstant = Magnum::Math::dot(normal, p);

    Debug{} << "Plane has normal" << normal << "and constant" << planeConstant;
    Debug{} << "Focal point is" << p;

    // Add it to the physics scene
    Vector3 xAxis = Vector3::xAxis();
    Vector3 zAxis = normal;
    Vector3 yAxis = Math::cross(zAxis, xAxis);

    Matrix3 rot{xAxis, yAxis, zAxis};

    Matrix4 T = Matrix4::from(rot, p);

    btBoxShape boxShape(btVector3{2.0, 2.0, 0.05});
    btDefaultMotionState boxState(btTransform{T});
    btRigidBody::btRigidBodyConstructionInfo info(
        0.0, &boxState, &boxShape
    );
    btRigidBody boxBody{info};

//     btStaticPlaneShape planeShape{btVector3{normal}, planeConstant};
//     btDefaultMotionState planeState;
//     btRigidBody planeBody{0.0, &planeState, &planeShape};
//     planeBody.setFriction(0.0);

    m_physicsWorld->addRigidBody(&boxBody);
    auto remover = finally([&]{ m_physicsWorld->removeRigidBody(&boxBody); });

    // Switch on gravity
    m_physicsWorld->setGravity(btVector3{-9.81f * normal});

    // Arrange the objects randomly above the plane
    std::uniform_real_distribution<float> posDist{-2.0f*maxDiameter, 2.0f*maxDiameter};
    Debug{} << "Initial object poses:";
    float z = 0.0f;
    for(auto& obj : m_objects)
    {
        z += maxDiameter;
        Magnum::Vector3 pos = p + z*normal /*+ Magnum::Vector3{
            posDist(m_randomGenerator), posDist(m_randomGenerator), posDist(m_randomGenerator)
        }*/;
        Magnum::Quaternion q = randomQuaternion(m_randomGenerator);

        Magnum::Matrix4 pose = Magnum::Matrix4::from(q.toMatrix(), pos);
        obj->setPose(pose);

        Debug{} << pose;
    }

    // We simulate a strong fake gravity towards p
    const Vector3 gravityCenter = p + 0.1*normal;

    const int maxIterations = 40;
    for(int i = 0; i < maxIterations; ++i)
    {
        if(visCallback)
            visCallback(i);

        Debug{} << "Iteration";
        for(auto& obj : m_objects)
        {
            Magnum::Vector3 dir = (gravityCenter - obj->pose().translation()).normalized();
            obj->rigidBody().applyCentralForce(btVector3{5.0f * dir});
            Debug{} << "Obj pos:" << obj->pose().translation();
//             Debug{} << "plane pos:" << Vector3{planeBody.getWorldTransform().getOrigin()};
        }

//         m_physicsWorld->setInternalTickCallback(&Scene::constrainingTickCallback, this);
        m_physicsWorld->stepSimulation(0.02, 10);
    }
}

}
