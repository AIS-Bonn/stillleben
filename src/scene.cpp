// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/scene.h>

#include <stillleben/context.h>
#include <stillleben/object.h>
#include <stillleben/light_map.h>
#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>

#include <Corrade/Containers/StaticArray.h>
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
    const float fy = fx;

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

void Scene::chooseRandomCameraPose()
{
    // Basic idea:
    //   1) Pick a random direction from which to look at the scene
    //   2) Find a camera position that keeps all objects inside the frustum

    // 1) is very simple.
    Matrix4 cameraRot;
    {
        std::uniform_real_distribution<float> azimuthDist{-M_PI, M_PI};
        constexpr Magnum::Rad ELEVATION_LIMIT{Magnum::Deg{30.0f}};
        std::uniform_real_distribution<float> elevationDist{
            static_cast<float>(ELEVATION_LIMIT), M_PI/2.0f - static_cast<float>(ELEVATION_LIMIT)
        };

        // Rotation into image coordinate system
        Magnum::Matrix3 camRot{
            -Magnum::Vector3::yAxis(),
            -Magnum::Vector3::zAxis(),
            Magnum::Vector3::xAxis()
        };

        cameraRot =
            Matrix4::rotationZ(Magnum::Rad{azimuthDist(m_randomGenerator)}) *
            Matrix4::rotationY(Magnum::Rad{elevationDist(m_randomGenerator)}) *
            Matrix4::from(camRot, {});
    }

    // If there are no objects, there is nothing to do.
    if(m_objects.empty())
    {
        Warning{} << "You called Scene::chooseRandomCameraPose() without objects";
        setCameraPose(Matrix4::translation({0.0, 0.0, -1.0f}) * cameraRot);
        return;
    }

    // 2) is quite complex. Approach:
    //   a) Determine the normals of the frustum planes (normals facing inward).
    //   b) Place each frustum plane so that it touches each object diameter sphere exactly.
    //      Record the maximum (i.e. the plane placement furthest in normal direction).
    //   c) The left/right and top/bottom frustom planes will intersect in a line, respectively.
    //   d) Choose the "backmost" line and project the other line on it. The resulting intersection
    //      point is the optimal candidate for the camere position.

    // To simplify everything, we are working in a rotated coordinate system aligned
    // with the camera coordinate system.
    Matrix4 toWorkSystem = cameraRot.invertedRigid();

    Containers::Array<Vector3> objectPoints(m_objects.size() * 8);
    for(std::size_t i = 0; i < m_objects.size(); ++i)
    {
        Range3D bbox = m_objects[i]->mesh()->bbox();
        Matrix4 trans = toWorkSystem * m_objects[i]->pose();

        objectPoints[i*8 + 0] = trans.transformPoint(bbox.backBottomLeft());
        objectPoints[i*8 + 1] = trans.transformPoint(bbox.backBottomRight());
        objectPoints[i*8 + 2] = trans.transformPoint(bbox.backTopLeft());
        objectPoints[i*8 + 3] = trans.transformPoint(bbox.backTopRight());
        objectPoints[i*8 + 4] = trans.transformPoint(bbox.frontBottomLeft());
        objectPoints[i*8 + 5] = trans.transformPoint(bbox.frontBottomRight());
        objectPoints[i*8 + 6] = trans.transformPoint(bbox.frontTopLeft());
        objectPoints[i*8 + 7] = trans.transformPoint(bbox.frontTopRight());
    }

    // a) Frustum planes (left, right, top, bottom)
    Containers::StaticArray<4, Vector4> frustum;
    {
        const auto proj = projectionMatrix();
        frustum[0] = proj.row(3) + proj.row(0); // left
        frustum[1] = proj.row(3) - proj.row(0); // right
        frustum[2] = proj.row(3) + proj.row(1); // top
        frustum[3] = proj.row(3) - proj.row(1); // bottom

        // normalize
        for(auto& f : frustum)
            f = f / f.xyz().length();
    }

    // b) Find maximum for each plane
    {
        for(auto& plane : frustum)
        {
            Vector3 normal = plane.xyz();
            float min_lambda = std::numeric_limits<float>::max();

            for(const auto& p : objectPoints)
                min_lambda = std::min(min_lambda, Math::dot(normal, p));

            plane.w() = -min_lambda;
        }
    }

    // c) Left/right and Top/bottom intersection
    float lr_x = 0.0f;
    float lr_z = 0.0f;
    {
        // Simply solve in 2D as we know that the frustum is aligned
        if(std::abs(frustum[0].y()) > 1e-3)
            Warning{} << "chooseRandomCameraPose(): Frustum is strange, the result might not be correct";

        Vector3 leftLine{frustum[0].x(), frustum[0].z(), frustum[0].w()};
        Vector3 rightLine{frustum[1].x(), frustum[1].z(), frustum[1].w()};

        Vector3 intersection = Math::cross(leftLine, rightLine);
        if(std::abs(intersection.z()) < 1e-3)
        {
            Warning{} << "chooseRandomCameraPose(): Frustum is strange, the result might not be correct";
            intersection = {0.0f, 0.0f, 1.0f};
        }

        lr_x = intersection[0] / intersection[2];
        lr_z = intersection[1] / intersection[2];
    }

    float tb_y = 0.0f;
    float tb_z = 0.0f;
    {
        // Simply solve in 2D as we know that the frustum is aligned
        if(std::abs(frustum[2].x()) > 1e-3)
            Warning{} << "chooseRandomCameraPose(): Frustum is strange, the result might not be correct";

        Vector3 topLine{frustum[2].y(), frustum[2].z(), frustum[2].w()};
        Vector3 bottomLine{frustum[3].y(), frustum[3].z(), frustum[3].w()};

        Vector3 intersection = Math::cross(topLine, bottomLine);
        if(std::abs(intersection.z()) < 1e-3)
        {
            Warning{} << "chooseRandomCameraPose(): Frustum is strange, the result might not be correct";
            intersection = {0.0f, 0.0f, 1.0f};
        }

        tb_y = intersection[0] / intersection[2];
        tb_z = intersection[1] / intersection[2];
    }

    Vector3 camPosition{lr_x, tb_y, std::min(lr_z, tb_z)};

    setCameraPose(cameraRot * Matrix4::translation(camPosition));
}

void Scene::simulateTableTopScene(const std::function<void(int)>& visCallback)
{
    loadPhysics();

    // What kind of objects do we have in the scene?
    float maxDiameter = 0.0f;
    for(auto& obj : m_objects)
    {
        float diameter = obj->mesh()->bbox().size().length();
        maxDiameter = std::max(maxDiameter, diameter);
    }

    // Define the plane. It always goes through the origin.
    constexpr Magnum::Vector3 normal{0.0f, 0.0f, 1.0f};

    // Add it to the physics scene
    Vector3 xAxis = Vector3::xAxis();
    Vector3 zAxis = normal;
    Vector3 yAxis = Math::cross(zAxis, xAxis);

    Matrix3 rot{xAxis, yAxis, zAxis};

    constexpr Magnum::Vector3 BOX_HALF_EXTENTS{30.0, 30.0, 0.04};

    Matrix4 T = Matrix4::from(rot, Vector3{});

    auto& physics = m_ctx->physxPhysics();

    physx::PxBoxGeometry boxGeom{BOX_HALF_EXTENTS.x(), BOX_HALF_EXTENTS.y(), BOX_HALF_EXTENTS.z()};
    PhysXHolder<physx::PxMaterial> material{physics.createMaterial(0.5f, 0.5f, 0.0f)};
    PhysXHolder<physx::PxShape> shape{physics.createShape(boxGeom, *material, true)};
    PhysXHolder<physx::PxRigidStatic> actor{physics.createRigidStatic(physx::PxTransform{T})};
    actor->attachShape(*shape);

    // Call setBackgroundPlanePose() with the correct pose so that the visual
    // matches up with the physics simulation
    {
        std::uniform_real_distribution<float> planeRotDist(-M_PI, M_PI);
        Magnum::Rad planeRot{planeRotDist(m_randomGenerator)};

        Matrix4 topSidePlanePose = T * Matrix4::rotationZ(planeRot) * Matrix4::translation({0.0, 0.0, BOX_HALF_EXTENTS.z()});
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

        Matrix4 pose = Matrix4::from(q.toMatrix(), pos) * Matrix4::translation(-obj->mesh()->bbox().center());
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

    chooseRandomCameraPose();
}

void Scene::serialize(Corrade::Utility::ConfigurationGroup& group) const
{
    group.setValue("viewport", m_camera->viewport());
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
    if(group.hasValue("viewport"))
        m_camera->setViewport(group.value<Magnum::Vector2i>("viewport"));

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
