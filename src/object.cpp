// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/object.h>

#include <stillleben/context.h>
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

#include "physx_impl.h"

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
{
}

Object::~Object()
{
}

void Object::load()
{
    m_rigidBody.reset(
        m_mesh->context()->physxPhysics().createRigidDynamic(physx::PxTransform(physx::PxIdentity))
    );
    m_rigidBody->userData = this;

    // pretransform is handled differently for Magnum & PhysX:
    // For Magnum, we just use the pretransform directly, while for PhysX
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

    // Calculate mass & inertia
    physx::PxRigidBodyExt::updateMassAndInertia(*m_rigidBody, 500.0f);
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

        auto physxMesh = m_mesh->physXMeshes()[objectData->instance()];

        if(physxMesh)
        {
            auto& physics = m_mesh->context()->physxPhysics();

            PhysXHolder<physx::PxMaterial> material{
                physics.createMaterial(0.5f, 0.5f, 0.0f)
            };
            physx::PxMeshScale meshScale(m_mesh->pretransformScale());

            physx::PxConvexMeshGeometry geometry(physxMesh->get(), meshScale);

            PhysXHolder<physx::PxShape> shape{
                physics.createShape(geometry, *material, true)
            };

            // Where are we relative to m_meshObject?
            Matrix4 poseInMeshObject = m_meshObject.absoluteTransformationMatrix().inverted() * object->absoluteTransformationMatrix();

            // The transformation between m_sceneObject and m_meshObject is
            // composed of a rigid part and a scale.
            Matrix4 poseInSceneObjectRigid = m_mesh->pretransformRigid() * poseInMeshObject;

            // PhysX applies scale locally in the shape.
            // => This is pretransformScale(), but we need to adjust the
            //    translation part of poseInSceneObjectRigid from scaled
            //    coordinates to unscaled ones.

            Matrix4 pose = Matrix4::from(
                poseInSceneObjectRigid.rotationScaling(),
                m_mesh->pretransformScale() * poseInSceneObjectRigid.translation()
            );

            shape->setLocalPose(physx::PxTransform{pose});

            m_rigidBody->attachShape(*shape);
        }
    }

    // Recursively add children
    for(std::size_t id: objectData->children())
        addMeshObject(*object, id);
}

std::shared_ptr<Object> Object::instantiate(const std::shared_ptr<Mesh>& mesh)
{
    if(!mesh)
        throw std::invalid_argument("Got nullptr mesh");

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

void Object::setPhysicsScene(physx::PxScene* scene)
{
    if(m_physicsScene)
        m_physicsScene->removeActor(*m_rigidBody);

    if(scene)
        scene->addActor(*m_rigidBody);

    m_physicsScene = scene;
}

void Object::setPose(const Magnum::Matrix4& matrix)
{
    m_sceneObject.setTransformation(matrix);

    m_rigidBody->setGlobalPose(physx::PxTransform{matrix});
}

void Object::updateFromPhysics()
{
    m_sceneObject.setTransformation(Matrix4{m_rigidBody->getGlobalPose()});
}

void Object::setInstanceIndex(unsigned int index)
{
    if(index > std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument("Object::setInstanceIndex(): out of range");

    m_instanceIndex = index;
}

}
