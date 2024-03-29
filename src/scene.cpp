// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/scene.h>

#include <stillleben/context.h>
#include <stillleben/object.h>
#include <stillleben/light_map.h>
#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>
#include <stillleben/physx_impl.h>

#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/StaticArray.h>
#include <Corrade/Utility/ConfigurationGroup.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Format.h>

#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include <Magnum/Math/ConfigurationValue.h>
#pragma GCC diagnostic pop

#include <sstream>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

namespace
{
    struct FilterShaderData
    {
        bool reportContacts = false;
    };

    using namespace physx;
    PxFilterFlags filterShader(
        PxFilterObjectAttributes attributes0,
        PxFilterData filterData0,
        PxFilterObjectAttributes attributes1,
        PxFilterData filterData1,
        PxPairFlags& pairFlags,
        const void* constantBlock,
        PxU32 constantBlockSize)
    {
        const auto& params = *reinterpret_cast<const FilterShaderData*>(constantBlock);

        if(params.reportContacts)
            pairFlags |= PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
        else
            pairFlags |= PxPairFlag::eCONTACT_DEFAULT;

        return PxFilterFlag::eDEFAULT;
    }
}

class Scene::SimulationCallback : public physx::PxSimulationEventCallback
{
public:
    void onContact(
        const physx::PxContactPairHeader&,
        const physx::PxContactPair* pairs, physx::PxU32 nbPairs) override
    {
        using namespace physx;

        for(PxU32 i=0; i < nbPairs; i++)
        {
            const PxContactPair& cp = pairs[i];

            if(cp.flags & (PxContactPairFlag::eACTOR_PAIR_LOST_TOUCH | PxContactPairFlag::eREMOVED_SHAPE_0 | PxContactPairFlag::eREMOVED_SHAPE_1))
                continue;

            auto obj1 = reinterpret_cast<sl::Object*>(cp.shapes[0]->getActor()->userData);
            auto obj2 = reinterpret_cast<sl::Object*>(cp.shapes[1]->getActor()->userData);

            if(!obj1 || !obj2)
                continue;

            if(!cp.contactPatches)
                continue;

            PxContactStreamIterator iter{
                cp.contactPatches, cp.contactPoints,
                cp.getInternalFaceIndices(),
                cp.patchCount, cp.contactCount
            };

            float minSep = std::numeric_limits<float>::infinity();

            while(iter.hasNextPatch())
            {
                iter.nextPatch();
                while(iter.hasNextContact())
                {
                    iter.nextContact();
                    minSep = std::min(iter.getSeparation(), minSep);
                }
            }

            obj1->m_separation = std::min(obj1->m_separation, minSep);
            obj2->m_separation = std::min(obj2->m_separation, minSep);
        }
    }

    void onWake(physx::PxActor ** actors, physx::PxU32 count) override
    {}

    void onAdvance(const physx::PxRigidBody *const * bodyBuffer, const physx::PxTransform * poseBuffer, const physx::PxU32 count) override
    {}

    void onConstraintBreak(physx::PxConstraintInfo * constraints, physx::PxU32 count) override
    {}

    void onSleep(physx::PxActor ** actors, physx::PxU32 count) override
    {}

    void onTrigger(physx::PxTriggerPair * pairs, physx::PxU32 count) override
    {}
};

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

    FilterShaderData shaderParams{true};

    physx::PxSceneDesc desc(physics.getTolerancesScale());
    desc.gravity = physx::PxVec3(0.0f, 0.0f, -9.81f);
    desc.cpuDispatcher = m_physicsDispatcher.get();
    desc.filterShader = filterShader;
//     desc.flags |= physx::PxSceneFlag::eENABLE_STABILIZATION;
    desc.filterShaderData = &shaderParams;
    desc.filterShaderDataSize = sizeof(FilterShaderData);
    desc.flags |= physx::PxSceneFlag::eENABLE_STABILIZATION;

    m_physicsScene.reset(physics.createScene(desc));

    if(auto client = m_physicsScene->getScenePvdClient())
        client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);

    // Enable contact reporting
    m_simCallback = std::make_unique<SimulationCallback>();
    m_physicsScene->setSimulationEventCallback(m_simCallback.get());
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

void Scene::removeObject(const std::shared_ptr<Object>& obj)
{
    auto it = std::find(m_objects.begin(), m_objects.end(), obj);

    if(it == m_objects.end())
        return;

    m_objects.erase(std::remove(m_objects.begin(), m_objects.end(), obj), m_objects.end());
    obj->setParentSceneObject(nullptr);
    obj->setPhysicsScene(nullptr);
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

void Scene::setLightDirections(const Containers::ArrayView<const Vector3>& positions)
{
    if(positions.size() > m_lightDirections.size())
        throw std::invalid_argument{"Cannot support that many lights"};

    for(std::size_t i = 0; i < positions.size(); ++i)
        m_lightDirections[i] = positions[i];
    for(std::size_t i = positions.size(); i < m_lightDirections.size(); ++i)
        m_lightDirections[i] = {};
}

Containers::StaticArrayView<NumLights, Vector3> Scene::lightDirections()
{
    return m_lightDirections;
}

void Scene::setLightColors(const Containers::ArrayView<const Color3>& colors)
{
    if(colors.size() > m_lightColors.size())
        throw std::invalid_argument{"Cannot support that many lights"};

    for(std::size_t i = 0; i < colors.size(); ++i)
        m_lightColors[i] = colors[i];
    for(std::size_t i = colors.size(); i < m_lightColors.size(); ++i)
        m_lightColors[i] = Color3{0.0f};
}

Containers::StaticArrayView<NumLights, Color3> Scene::lightColors()
{
    return m_lightColors;
}

void Scene::setAmbientLight(const Magnum::Color3& color)
{
    m_ambientLight = color;
}

void Scene::chooseRandomLightDirection()
{
    // We want to have the light coming from above, but not from behind the
    // objects. We first determine the light position relative to the camera.

    std::normal_distribution<float> normalDist;
    Magnum::Vector3 randomDirection = Magnum::Vector3{
        normalDist(m_randomGenerator),
        -std::abs(normalDist(m_randomGenerator)), // always from above
        -std::abs(normalDist(m_randomGenerator)) // always on camera side
    }.normalized();

    Magnum::Vector3 lightDirectionInCam = -randomDirection.normalized();

    Vector3 lightDirectionInWorld = m_camera->cameraMatrix().invertedRigid().transformVector(lightDirectionInCam);

    setLightDirections({lightDirectionInWorld});
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

    std::vector<std::shared_ptr<Object>> dynamicObjects;
    std::copy_if(m_objects.begin(), m_objects.end(), std::back_inserter(dynamicObjects), [](auto& obj){
        return !obj->isStatic();
    });

    // Define the plane. It always goes through the origin.
    constexpr Magnum::Vector3 normal{0.0f, 0.0f, 1.0f};

    PhysXHolder<physx::PxRigidStatic> actor;
    auto remover = finally([&](){ if(actor) m_physicsScene->removeActor(*actor); });

    float z = 0.4f;

    if(m_objects.size() == dynamicObjects.size())
    {
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
        actor.reset(physics.createRigidStatic(physx::PxTransform{T}));
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

        z = BOX_HALF_EXTENTS.z();
    }

    m_physicsScene->setGravity(physx::PxVec3{-9.81f * normal});

    for(auto& obj : dynamicObjects)
    {
        double diameter = obj->mesh()->bbox().size().length();
        z += diameter/2.0f;
        Magnum::Vector3 pos = z*normal;
        z += diameter/2.0f;

        Magnum::Quaternion q = randomQuaternion(m_randomGenerator);

        Matrix4 pose = Matrix4::from(q.toMatrix(), pos) * Matrix4::translation(-obj->mesh()->bbox().center());
        obj->setPose(pose);
        obj->rigidBody().wakeUp();
    }

    constexpr unsigned int FPS = 25;
    constexpr float TIME_PER_FRAME = 1.0f / FPS;
    constexpr unsigned int SUBSTEPS_PRECISE = 4;
    constexpr unsigned int SUBSTEPS_FAST = 4;

    auto redropObject = [&](std::shared_ptr<sl::Object>& obj){
        float maxZ = 0.0f;

        for(auto& o : dynamicObjects)
        {
            if(o == obj)
                continue;

            auto bboxCenter = o->pose().transformPoint(o->mesh()->bbox().center());
            auto bboxR = o->mesh()->bbox().size().length()/2;
            maxZ = std::max(maxZ, bboxCenter.z() + bboxR);
        }

        Vector3 bboxOffsetInGlobal = obj->pose().transformVector(obj->mesh()->bbox().center());
        bboxOffsetInGlobal.z() -= obj->mesh()->bbox().size().length()/2;

        Matrix4 pose = obj->pose();
        pose.translation() = Vector3{0.0f, 0.0f, maxZ - bboxOffsetInGlobal.z()};

        obj->m_stuckCounter = 0;

        obj->setPose(pose);
        obj->rigidBody().clearForce(physx::PxForceMode::eACCELERATION);
        obj->rigidBody().setLinearVelocity(physx::PxVec3{Vector3{}});
        obj->rigidBody().setAngularVelocity(physx::PxVec3{Vector3{}});
    };

    for(auto& obj : dynamicObjects)
    {
        obj->rigidBody().wakeUp();
        m_physicsScene->resetFiltering(obj->rigidBody());
        obj->m_stuckCounter = 0;
    }

    const int maxIterations = 100;
    for(int i = 0; i < maxIterations; ++i)
    {
        if(visCallback)
            visCallback(i);

        const unsigned int SUBSTEPS = (i < 40) ? SUBSTEPS_PRECISE : SUBSTEPS_FAST;

        for(unsigned int j = 0; j < SUBSTEPS; ++j)
        {
            for(auto& obj : dynamicObjects)
            {
                obj->m_separation = std::numeric_limits<float>::infinity();

                if(obj->rigidBody().isSleeping())
                    continue;
            }

            m_physicsScene->simulate(TIME_PER_FRAME / SUBSTEPS);
            m_physicsScene->fetchResults(true);
        }

        for(auto& obj : dynamicObjects)
        {
            obj->updateFromPhysics();

            if(obj->pose().translation().z() < -0.5)
                redropObject(obj);
            else if(obj->m_separation < -0.01f)
            {
                if(++obj->m_stuckCounter > 0.4f / TIME_PER_FRAME)
                    redropObject(obj);
            }
            else if(obj->m_stuckCounter > 0)
                obj->m_stuckCounter--;
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

    for(UnsignedInt i = 0; i < m_lightDirections.size(); ++i)
    {
        auto lightGroup = group.addGroup("light");

        lightGroup->setValue("direction", m_lightDirections[i]);
        lightGroup->setValue("color", m_lightColors[i]);
    }

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

    group.setValue("manualExposure", m_manualExposure);
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
    {
        setLightDirections({-group.value<Magnum::Vector3>("lightPosition").normalized()});
        setLightColors({Color3{0.0f, 0.8f, 0.0f}});
    }
    else
    {
        auto lightGroups = group.groups("light");

        Containers::Array<Vector3> directions;
        Containers::Array<Color3> colors;

        for(const auto& lightGroup : lightGroups)
        {
            Containers::arrayAppend(directions, lightGroup->value<Magnum::Vector3>("direction"));
            Containers::arrayAppend(colors, lightGroup->value<Magnum::Color3>("color"));
        }

        setLightDirections(directions);
        setLightColors(colors);
    }

    if(group.hasValue("ambientLight"))
        m_ambientLight = group.value<Magnum::Color3>("ambientLight");

    if(group.hasValue("lightMap"))
        m_lightMap = std::make_shared<LightMap>(group.value("lightMap"), m_ctx);

    if(group.hasValue("backgroundPlanePose"))
        m_backgroundPlanePose = group.value<Magnum::Matrix4>("backgroundPlanePose");
    if(group.hasValue("backgroundPlaneSize"))
        m_backgroundPlaneSize = group.value<Magnum::Vector2>("backgroundPlaneSize");

    if(group.hasValue("manualExposure"))
        m_manualExposure = group.value<Float>("manualExposure");

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

void Scene::simulate(float dt)
{
    loadPhysics();

    m_physicsScene->simulate(dt);
    m_physicsScene->fetchResults(true);

    for(auto& obj : m_objects)
        obj->updateFromPhysics();
}

void Scene::checkCollisions()
{
    loadPhysics();

    for(auto& obj : m_objects)
    {
        if(isObjectColliding(*obj))
            obj->m_separation = -std::numeric_limits<float>::max();
        else
            obj->m_separation = 0.0f;
    }
}

void Scene::setManualExposure(Float exposure)
{
    m_manualExposure = exposure;
}

}
