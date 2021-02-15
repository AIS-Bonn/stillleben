// Simulate a manipulator interacting with the scene
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/manipulation_sim.h>

#include <stillleben/context.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/physx.h>

#include "physx_impl.h"

using namespace physx;
using namespace Magnum;

namespace sl
{

class ManipulationSim::Private
{
public:
    std::shared_ptr<sl::Scene> scene;
    std::shared_ptr<sl::Object> manipulator;

    Matrix4 initialPose;
    PhysXHolder<PxD6Joint> joint;
};

ManipulationSim::ManipulationSim(const std::shared_ptr<sl::Scene>& scene, const std::shared_ptr<sl::Object>& manipulator, const Magnum::Matrix4& initialPose)
 : m_d{Containers::pointer<Private>()}
{
    m_d->scene = scene;
    m_d->manipulator = manipulator;
    m_d->initialPose = initialPose;

    // Make sure the object is added to the scene
    if(std::find(scene->objects().begin(), scene->objects().end(), manipulator) == scene->objects().end())
        scene->addObject(manipulator);

    manipulator->setPose(initialPose);

    scene->loadPhysics();

    auto& physics = m_d->scene->context()->physxPhysics();

    // Add a 6D joint which we use to control the manipulator
    m_d->joint.reset(PxD6JointCreate(physics,
        nullptr, PxTransform{initialPose}, // at current pose relative to world (null)
        &m_d->manipulator->rigidBody(), PxTransform{PxIDENTITY::PxIdentity} // in origin of manipulator
    ));

    // By default rotation is locked
    lockRotationAxes(true, true, true);

    // Setup default spring parameters
    setSpringParameters(600.0f, 0.1f, 60.0f);
}

ManipulationSim::~ManipulationSim()
{
}

void ManipulationSim::lockRotationAxes(bool x, bool y, bool z)
{
    m_d->joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
    m_d->joint->setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
    m_d->joint->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
    m_d->joint->setMotion(PxD6Axis::eTWIST, x ? PxD6Motion::eLOCKED : PxD6Motion::eFREE);
    m_d->joint->setMotion(PxD6Axis::eSWING1, y ? PxD6Motion::eLOCKED : PxD6Motion::eFREE);
    m_d->joint->setMotion(PxD6Axis::eSWING2, z ? PxD6Motion::eLOCKED : PxD6Motion::eFREE);
}

void ManipulationSim::setSpringParameters(float stiffness, float damping, float forceLimit)
{
    PxD6JointDrive drive(stiffness, damping, forceLimit);

    m_d->joint->setDrive(PxD6Drive::eX, drive);
    m_d->joint->setDrive(PxD6Drive::eY, drive);
    m_d->joint->setDrive(PxD6Drive::eZ, drive);
}

void ManipulationSim::step(const Magnum::Matrix4& goalPose, float dt)
{
    m_d->joint->setDrivePosition(PxTransform{m_d->initialPose.invertedRigid() * goalPose});

    auto scene = m_d->manipulator->rigidBody().getScene();
    scene->simulate(dt);
    scene->fetchResults(true);

    for(auto& obj : m_d->scene->objects())
        obj->updateFromPhysics();
}


}
