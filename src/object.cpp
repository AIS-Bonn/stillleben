// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/object.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>

#include <limits>

#include <Corrade/Utility/ConfigurationGroup.h>

#include <Magnum/DebugTools/ObjectRenderer.h>

#include <Magnum/Math/Range.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/ConfigurationValue.h>

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

void Object::setMesh(const std::shared_ptr<sl::Mesh>& mesh)
{
    if(m_mesh)
        throw std::logic_error("Re-setting the mesh via setMesh() is currently not supported");

    m_mesh = mesh;
}

void Object::setInstantiationOptions(const InstantiationOptions& opts)
{
    m_options = opts;
}

void Object::populateParts()
{
    if(!m_parts.empty())
        return; // already done

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
            addPart(m_meshObject, objectId);
    }
    else if(!m_mesh->meshes().empty() && m_mesh->meshes()[0])
    {
        // The format has no scene support, display just the first loaded mesh with
        // a default material and be done with it
        addPart(m_meshObject, 0);
    }
}

void Object::loadVisual()
{
    if(m_visualLoaded)
        return;

    m_mesh->loadVisual();
    populateParts();

    for(auto part : m_parts)
    {
        auto objectData = m_mesh->importer().object3D(part->index());

        // Add a drawable if the object has a mesh and the mesh is loaded
        if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && objectData->instance() != -1 && m_mesh->meshes()[objectData->instance()])
        {
            auto meshObjectData = static_cast<Trade::MeshObjectData3D*>(objectData.get());
            auto mesh = m_mesh->meshes()[objectData->instance()];
            auto meshFlags = m_mesh->meshFlags()[objectData->instance()];
            const Int materialId = meshObjectData->material();

            auto drawable = new Drawable{*part, m_drawables, mesh, &m_cb};

            if(m_options.forceColor || materialId == -1 || !m_mesh->materials()[materialId])
            {
                // Material not available / not loaded, use a default material
                drawable->setColor(m_options.color);
            }
            else if(m_mesh->materials()[materialId]->flags() & Trade::PhongMaterialData::Flag::DiffuseTexture)
            {
                // Textured material. If the texture failed to load, again just use a
                // default colored material.
                Containers::Optional<GL::Texture2D>& texture = m_mesh->textures()[m_mesh->materials()[materialId]->diffuseTexture()];
                if(texture)
                    drawable->setTexture(&*texture);
                else
                    drawable->setColor(m_options.color);
            }
            else
            {
                // Color-only material
                drawable->setColor(m_mesh->materials()[materialId]->diffuseColor());
            }

            drawable->setHasVertexColors(!m_options.forceColor && (meshFlags & Mesh::MeshFlag::HasVertexColors));
        }
    }

    new DebugTools::ObjectRenderer3D{m_sceneObject, {}, &m_debugDrawables};

    m_visualLoaded = true;
}

void Object::loadPhysics()
{
    if(m_rigidBody)
        return;

    m_mesh->loadPhysics();
    populateParts();

    m_rigidBody.reset(
        m_mesh->context()->physxPhysics().createRigidDynamic(physx::PxTransform(physx::PxIdentity))
    );
    m_rigidBody->userData = this;
    m_rigidBody->setGlobalPose(physx::PxTransform{m_sceneObject.transformation()});

    if(m_physicsScene)
        m_physicsScene->addActor(*m_rigidBody);

    for(auto part : m_parts)
    {
        auto objectData = m_mesh->importer().object3D(part->index());

        if(!objectData || objectData->instanceType() != Trade::ObjectInstanceType3D::Mesh)
            continue;

        if(objectData->instance() == -1)
            continue;

        auto physxMesh = m_mesh->physXMeshes()[objectData->instance()];

        if(!physxMesh)
            continue;

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
        Matrix4 poseInMeshObject = m_meshObject.absoluteTransformationMatrix().inverted() * part->absoluteTransformationMatrix();

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

    // Calculate mass & inertia
    physx::PxRigidBodyExt::updateMassAndInertia(*m_rigidBody, 500.0f);
}

void Object::addPart(Object3D& parent, UnsignedInt i)
{
    std::unique_ptr<Trade::ObjectData3D> objectData = m_mesh->importer().object3D(i);
    if(!objectData)
    {
        Error{} << "Cannot import object, skipping";
        return;
    }

    // Add the object to the scene and set its transformation
    auto* object = new Part{i, &parent};
    object->setTransformation(objectData->transformation());

    m_parts.push_back(object);

    // Recursively add children
    for(std::size_t id: objectData->children())
        addPart(*object, id);
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
    if(m_rigidBody)
    {
        if(m_physicsScene)
            m_physicsScene->removeActor(*m_rigidBody);

        if(scene)
            scene->addActor(*m_rigidBody);
    }

    m_physicsScene = scene;
}

void Object::setPose(const Magnum::Matrix4& matrix)
{
    m_sceneObject.setTransformation(matrix);

    if(m_rigidBody)
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

void Object::serialize(Corrade::Utility::ConfigurationGroup& group)
{
    auto meshGroup = group.addGroup("mesh");
    m_mesh->serialize(*meshGroup);

    group.setValue("pose", pose());
    group.setValue("instanceIndex", m_instanceIndex);
    group.setValue("specularColor", m_specularColor);
    group.setValue("shininess", m_shininess);
    group.setValue("roughness", m_roughness);
    group.setValue("metalness", m_metalness);
}

void Object::deserialize(const Corrade::Utility::ConfigurationGroup& group, MeshCache& cache)
{
    auto meshGroup = group.group("mesh");
    if(!meshGroup)
        throw std::runtime_error("Did not find mesh subgroup in object");

    setMesh(cache.load(*meshGroup));

    // FIXME: support InstantiationOptions
    populateParts();

    if(group.hasValue("pose"))
        setPose(group.value<Magnum::Matrix4>("pose"));

    if(group.hasValue("instanceIndex"))
        setInstanceIndex(group.value<unsigned int>("instanceIndex"));

    if(group.hasValue("specularColor"))
        setSpecularColor(group.value<Magnum::Color4>("specularColor"));

    if(group.hasValue("shininess"))
        setShininess(group.value<float>("shininess"));

    if(group.hasValue("roughness"))
        setRoughness(group.value<float>("roughness"));

    if(group.hasValue("metalness"))
        setMetalness(group.value<float>("metalness"));
}

void Object::setSpecularColor(const Magnum::Color4& color)
{
    m_specularColor = color;
}

void Object::setShininess(float shininess)
{
    m_shininess = shininess;
}

void Object::setRoughness(float roughness)
{
    m_roughness = roughness;
}

void Object::setMetalness(float metalness)
{
    m_metalness = metalness;
}

}
