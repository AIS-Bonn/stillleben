// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <BulletCollision/CollisionShapes/btCompoundShape.h>
#include <btBulletDynamicsCommon.h>

#include <stillleben/object.h>

#include <stillleben/mesh.h>

#include <limits>

#include <Magnum/DebugTools/ObjectRenderer.h>

#include <Magnum/Math/Range.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Color.h>

#include <Magnum/Mesh.h>
#include <Magnum/MeshTools/Compile.h>

#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/MeshObjectData3D.h>

#include <Magnum/BulletIntegration/Integration.h>
#include <Magnum/BulletIntegration/MotionState.h>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

void Drawable::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera)
{
    auto& cb = *m_cb;
    if(cb)
        cb(transformationMatrix, camera, this);
}

Object::Object()
 : m_collisionShape{std::make_unique<btCompoundShape>()}
{
}

Object::~Object()
{
}

void Object::load()
{
    // pretransform is handled differently for Magnum & Bullet:
    // For Magnum, we just use the pretransform directly, while for Bullet
    // we have to split it into the rigid part (see addMeshObject()) and
    // the scaling part below.
    m_meshObject.setTransformation(m_mesh->pretransform());

    // Load the scene
    auto& importer = m_mesh->importer();
    if(importer.defaultScene() != -1)
    {
        Containers::Optional<Trade::SceneData> sceneData = importer.scene(importer.defaultScene());
        if(!sceneData)
        {
            throw Exception("Could not load scene data");
        }

        // Recursively add all children
        for(UnsignedInt objectId : sceneData->children3D())
            addMeshObject(m_meshObject, objectId);
    }
    else if(!m_mesh->meshes().empty() && m_mesh->meshes()[0])
    {
        // The format has no scene support, display just the first loaded mesh with
        // a default material and be done with it
        addMeshObject(m_meshObject, 0);
    }

    new DebugTools::ObjectRenderer3D{m_sceneObject, {}, &m_debugDrawables};

    // Setup bullet integration

    // NOTE: This call has to be after the child shapes are added.
    // Bullet is horrible...
    m_collisionShape->setLocalScaling(btVector3{Vector3{m_mesh->pretransformScale()}});

    auto motionState = new Magnum::BulletIntegration::MotionState{m_sceneObject};

    const auto mass = 0.2f;

    btVector3 bInertia(0.0f, 0.0f, 0.0f);
    m_collisionShape->calculateLocalInertia(mass, bInertia);

    btRigidBody::btRigidBodyConstructionInfo info(
        mass, &motionState->btMotionState(), m_collisionShape.get(), bInertia
    );
    m_rigidBody = std::make_unique<btRigidBody>(info);
    m_rigidBody->forceActivationState(DISABLE_DEACTIVATION);
}

void Object::addMeshObject(Object3D& parent, UnsignedInt i)
{
    std::unique_ptr<Trade::ObjectData3D> objectData = m_mesh->importer().object3D(i);
    if(!objectData)
    {
        Error{} << "Cannot import object, skipping";
        return;
    }

    // Add the object to the scene and set its transformation
    auto* object = new Object3D{&parent};
    object->setTransformation(objectData->transformation());

    // Add a drawable if the object has a mesh and the mesh is loaded
    if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && objectData->instance() != -1 && m_mesh->meshes()[objectData->instance()])
    {
        const Int materialId = static_cast<Trade::MeshObjectData3D*>(objectData.get())->material();

        auto drawable = new Drawable{*object, m_drawables, m_mesh->meshes()[objectData->instance()], &m_cb};

        if(materialId == -1 || !m_mesh->materials()[materialId])
        {
            // Material not available / not loaded, use a default material
            drawable->setColor(0xffffff_rgbf);
        }
        else if(m_mesh->materials()[materialId]->flags() & Trade::PhongMaterialData::Flag::DiffuseTexture)
        {
            // Textured material. If the texture failed to load, again just use a
            // default colored material.
            Containers::Optional<GL::Texture2D>& texture = m_mesh->textures()[m_mesh->materials()[materialId]->diffuseTexture()];
            if(texture)
                drawable->setTexture(&*texture);
            else
                drawable->setColor(0xffffff_rgbf);
        }
        else
        {
            // Color-only material
            drawable->setColor(m_mesh->materials()[materialId]->diffuseColor());
        }

        // This is a bit tricky: Bullet can only handle rigid transforms
        // here. So we ask for the relative transform to m_meshObject
        // (which is rigid), and then apply the rigid part of the pretransform.
        // Scaling is then handled by scaling the entire bullet collision shape.
        btTransform bulletTransform(
            m_mesh->pretransformRigid()
             * m_meshObject.absoluteTransformationMatrix().inverted()
             * object->absoluteTransformationMatrix()
        );

        m_collisionShape->addChildShape(
            bulletTransform,
            m_mesh->collisionShapes()[objectData->instance()].get()
        );
    }

    // Recursively add children
    for(std::size_t id: objectData->children())
        addMeshObject(*object, id);
}

std::shared_ptr<Object> Object::instantiate(const std::shared_ptr<Mesh>& mesh)
{
    auto object = std::make_shared<Object>();

    object->m_mesh = mesh;

    object->load();

    return object;
}

void Object::draw(Magnum::SceneGraph::Camera3D& camera, const DrawCallback& cb)
{
    m_cb = cb;
    camera.draw(m_drawables);
}

void Object::setParentSceneObject(Object3D* parent)
{
    m_sceneObject.setParent(parent);
}

void Object::setPhysicsWorld(btDiscreteDynamicsWorld* world)
{
    world->addRigidBody(m_rigidBody.get());
}

void Object::setPose(const PoseMatrix& matrix)
{
    m_sceneObject.setTransformation(matrix);

    // Stupid, but hey, it works!
    btTransform transform;
    m_rigidBody->getMotionState()->getWorldTransform(transform);
    m_rigidBody->setWorldTransform(transform);
}

void Object::setInstanceIndex(unsigned int index)
{
    if(index > std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument("Object::setInstanceIndex(): out of range");

    m_instanceIndex = index;
}

btRigidBody& Object::rigidBody()
{
    if(!m_rigidBody)
        throw std::logic_error("Access of rigidBody() before object is loaded");

    return *m_rigidBody;
}

}
