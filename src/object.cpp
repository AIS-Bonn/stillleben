// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/object.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>
#include <stillleben/physx_impl.h>

#include <limits>
#include <sstream>

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
    const auto& sceneData = m_mesh->sceneData();
    if(sceneData)
    {
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
        const auto& objectData = m_mesh->objects()[part->index()];

        // Add a drawable if the object has a mesh and the mesh is loaded
        if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && m_mesh->meshes()[part->index()])
        {
            auto meshObjectData = static_cast<const Trade::MeshObjectData3D*>(objectData.get());
            auto mesh = m_mesh->meshes()[part->index()];
            auto meshFlags = m_mesh->meshFlags()[part->index()];
            const Int materialId = meshObjectData->material();

            auto drawable = new Drawable{*part, m_drawables, mesh, &m_cb};
            drawable->setHasVertexColors(!m_options.forceColor && (meshFlags & Mesh::MeshFlag::HasVertexColors));

            if(m_options.forceColor || materialId == -1 || !m_mesh->materials()[materialId])
            {
                // Material not available / not loaded, use a default material
                drawable->setColor(m_options.color);
                continue;
            }

            const auto& material = m_mesh->materials()[materialId];

            drawable->setMetallicRoughness(material->metallic(), material->roughness());

            if(material->flags() & PBRMaterialData::Flag::BaseColorTexture)
            {
                // Textured material. If the texture failed to load, again just use a
                // default colored material.
                Containers::Optional<GL::Texture2D>& texture = m_mesh->textures()[material->baseColorTexture()];
                if(texture)
                    drawable->setTexture(&*texture);
                else
                    drawable->setColor(m_options.color);
            }
            else
            {
                // Color-only material
                drawable->setColor(m_mesh->materials()[materialId]->baseColor());
            }
        }
    }

    new DebugTools::ObjectRenderer3D{m_mesh->context()->debugResourceManager(), m_sceneObject, {}, &m_debugDrawables};

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
    m_rigidBody->setRigidBodyFlag(physx::PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);

    try
    {
        setPhysicsPose(); // may throw if the pose is not proper
    }
    catch(...)
    {
        // so make sure we reset m_rigidBody if that happens so the user may retry.
        m_rigidBody.reset();
        throw;
    }

    if(m_physicsScene)
        m_physicsScene->addActor(*m_rigidBody);

    auto& physxMeshes = m_mesh->physXMeshes();
    auto& physics = m_mesh->context()->physxPhysics();

    // The transformation between m_sceneObject and m_meshObject is
    // composed of a rigid part and a scale.
    Matrix4 poseInSceneObjectRigid = m_mesh->pretransformRigid();

    // PhysX applies scale locally in the shape.
    // => This is pretransformScale(), but we need to adjust the
    //    translation part of poseInSceneObjectRigid from scaled
    //    coordinates to unscaled ones.

    Matrix4 pose = Matrix4::from(
        poseInSceneObjectRigid.rotationScaling(),
        m_mesh->pretransformScale() * poseInSceneObjectRigid.translation()
    );

    physx::PxMaterial& material = m_mesh->context()->physxDefaultMaterial();

    for(auto& physxMesh : physxMeshes)
    {
        physx::PxMeshScale meshScale(m_mesh->pretransformScale());

        // FIXME: Ugly const_cast
        physx::PxConvexMeshGeometry geometry(const_cast<physx::PxConvexMesh*>(physxMesh.get()), meshScale);

        PhysXHolder<physx::PxShape> shape{
            physics.createShape(geometry, material, true)
        };

        shape->setLocalPose(physx::PxTransform{pose});
        shape->setRestOffset(0.0015);

        m_rigidBody->attachShape(*shape);
    }

    // Calculate mass & inertia
    setDensity(m_density);

    // Synchronize static flag
    setStatic(isStatic());
}

void Object::setDensity(float density)
{
    loadPhysics();

    physx::PxRigidBodyExt::updateMassAndInertia(*m_rigidBody, density);
    m_density = density;
}

void Object::setMass(float mass)
{
    loadPhysics();

    float factor = mass / m_rigidBody->getMass();
    setDensity(m_density * factor);
}

float Object::mass()
{
    loadPhysics();

    return m_rigidBody->getMass();
}

float Object::volume()
{
    loadPhysics();

    return m_rigidBody->getMass() / m_density;
}

void Object::setLinearVelocityLimit(float limit)
{
    loadPhysics();

    m_rigidBody->setMaxLinearVelocity(limit);
}

float Object::linearVelocityLimit()
{
    loadPhysics();

    return m_rigidBody->getMaxLinearVelocity();
}



void Object::loadPhysicsVisualization()
{
    if(m_physicsVisLoaded)
        return;

    m_mesh->loadPhysicsVisualization();

    populateParts();

    auto& meshes = m_mesh->physXVisualizationMeshes();

    for(auto mesh : meshes)
        new Drawable{m_meshObject, m_physXDrawables, mesh, &m_cb};

    m_physicsVisLoaded = true;
}

void Object::addPart(Object3D& parent, UnsignedInt i)
{
    const auto& objectData = m_mesh->objects()[i];
    if(!objectData)
    {
        Error{} << "Cannot import object, skipping";
        return;
    }

    auto* object = new Part{i, &parent};

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

void Object::drawPhysics(Magnum::SceneGraph::Camera3D& camera, const DrawCallback& cb)
{
    m_cb = cb;
    camera.draw(m_physXDrawables);
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
    setPhysicsPose();
}

void Object::setPhysicsPose()
{
    if(!m_rigidBody)
        return;

    float det = m_sceneObject.transformation().rotationScaling().determinant();

    if(std::abs(det - 1.0f) > 0.01f)
    {
        std::stringstream ss;
        Debug dbg{&ss};
        dbg << "You provided a pose which is not a pure rotation / translation:\n";
        dbg << m_sceneObject.transformation();
        dbg << "\n(determinant:" << det << dbg.nospace << ")\n";
        dbg << "This is not supported when using the physics engine.";

        throw std::runtime_error{ss.str()};
    }

    m_rigidBody->setGlobalPose(physx::PxTransform{m_sceneObject.transformation()});
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
    group.setValue("metallic", m_metallic);

    // FIXME: What about stickerTexture?
    group.setValue("stickerRange", m_stickerRange);
    group.setValue("stickerRotation", m_stickerRotation);

    group.setValue("static", m_static);
    group.setValue("density", m_density);
    group.setValue("linear_velocity_limit", linearVelocityLimit());
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

    if(group.hasValue("metallic"))
        setMetallic(group.value<float>("metallic"));

    if(group.hasValue("stickerRange"))
        setStickerRange(group.value<Magnum::Range2D>("stickerRange"));
    if(group.hasValue("stickerRotation"))
        setStickerRotation(group.value<Magnum::Quaternion>("stickerRotation"));

    if(group.hasValue("static"))
        setStatic(group.value<bool>("static"));

    if(group.hasValue("density"))
        setDensity(group.value<float>("density"));

    if(group.hasValue("linearVelocityLimit"))
        setLinearVelocityLimit(group.value<float>("linearVelocityLimit"));
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

void Object::setMetallic(float metalness)
{
    m_metallic = metalness;
}

void Object::setStickerTexture(const std::shared_ptr<Magnum::GL::RectangleTexture>& texture)
{
    m_stickerTexture = texture;
}

void Object::setStickerRange(const Magnum::Range2D& range)
{
    m_stickerRange = range;
}

void Object::setStickerRotation(const Magnum::Quaternion& q)
{
    m_stickerRotation = q;
}

Magnum::Matrix4 Object::stickerViewProjection() const
{
    const float diagonal = m_mesh->bbox().size().length();

    // NOTE: This is a crude approximation, it does not guarantee that the
    // object fits inside the projection frustum.
    // (or at least I haven't thought about it further)
    const Magnum::Matrix4 proj{
        {2.0f / diagonal,            0.0f, 0.0f, 0.0f},
        {           0.0f, 2.0f / diagonal, 0.0f, 0.0f},
        {           0.0f,            0.0f, 1.0f, 0.0f},
        {           0.0f,            0.0f, 1.0f, 1.0f},
    };

    constexpr Magnum::Matrix4 trans = Magnum::Matrix4::translation(
        {0.0f, 0.0f, 1.0f}
    );

    return proj * trans * Magnum::Matrix4{m_stickerRotation.toMatrix()};
}

void Object::setStatic(bool isStatic)
{
    m_static = isStatic;
    if(m_rigidBody)
        m_rigidBody->setRigidBodyFlag(physx::PxRigidBodyFlag::eKINEMATIC, isStatic);
}

void Object::setLinearVelocity(const Magnum::Vector3& velocity)
{
    loadPhysics();
    m_rigidBody->setLinearVelocity(physx::PxVec3{velocity});
}

Magnum::Vector3 Object::linearVelocity()
{
    loadPhysics();
    return Vector3{m_rigidBody->getLinearVelocity()};
}


void Object::setAngularVelocity(const Magnum::Vector3& velocity)
{
    loadPhysics();
    m_rigidBody->setAngularVelocity(physx::PxVec3{velocity});
}

Magnum::Vector3 Object::angularVelocity()
{
    loadPhysics();
    return Vector3{m_rigidBody->getAngularVelocity()};
}

}
