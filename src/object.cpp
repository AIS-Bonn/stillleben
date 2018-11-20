// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/object.h>

#include <stillleben/mesh.h>

#include <limits>

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

void Object::load()
{
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
            addMeshObject(m_sceneObject, objectId);
    }
    else if(!m_mesh->meshes().empty() && m_mesh->meshes()[0])
    {
        // The format has no scene support, display just the first loaded mesh with
        // a default material and be done with it
        addMeshObject(m_sceneObject, 0);
    }
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

        auto mesh = MeshTools::compile(*m_mesh->meshes()[objectData->instance()]);

        auto drawable = new Drawable{*object, m_drawables, std::move(mesh), &m_cb};

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

        // Update bbox
        auto trans = object->absoluteTransformation();

        for(const auto& point : *m_mesh->meshPoints()[objectData->instance()])
        {
            auto transformed = trans.transformPoint(point);

            m_bbox.min().x() = std::min(m_bbox.min().x(), transformed.x());
            m_bbox.min().y() = std::min(m_bbox.min().y(), transformed.y());
            m_bbox.min().z() = std::min(m_bbox.min().z(), transformed.z());

            m_bbox.max().x() = std::max(m_bbox.max().x(), transformed.x());
            m_bbox.max().y() = std::max(m_bbox.max().y(), transformed.y());
            m_bbox.max().z() = std::max(m_bbox.max().z(), transformed.z());
        }
    }

    // Recursively add children
    for(std::size_t id: objectData->children())
        addMeshObject(*object, id);
}


Object::Object()
{
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

}
