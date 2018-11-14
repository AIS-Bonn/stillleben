// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/object.h>

#include <stillleben/mesh.h>
#include <stillleben/impl/mesh.h>

#include <limits>

#include <Magnum/Math/Range.h>
#include <Magnum/Math/Vector3.h>

#include <Magnum/Mesh.h>
#include <Magnum/MeshTools/Compile.h>

#include <Magnum/SceneGraph/SceneGraph.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/MeshObjectData3D.h>

using namespace Magnum;
using namespace Math::Literals;
typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;

namespace sl
{

namespace
{
    class Drawable : public SceneGraph::Drawable3D
    {
    public:
        Drawable(Object3D& object, SceneGraph::DrawableGroup3D& group, GL::Mesh&& mesh)
         : SceneGraph::Drawable3D{object, &group}
         , m_mesh{std::move(mesh)}
        {
        }

        void setTexture(const GL::Texture2D* texture)
        { m_texture = texture; }

        void setColor(const Color4& color)
        { m_color = color; }

        void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) override;
    private:
        GL::Mesh m_mesh;
        const GL::Texture2D* m_texture = nullptr;
        Color4 m_color = 0xffffff_rgbf;
    };

    void Drawable::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera)
    {

    }
}

class Object::Private
{
public:
    void load();
    void addMeshObject(Object3D& parent, UnsignedInt i);

    std::shared_ptr<Mesh> mesh;
    const Mesh::Private* mesh_d;

    Object3D sceneObject;
    SceneGraph::DrawableGroup3D drawables;
    Range3D bbox{Vector3(std::numeric_limits<float>::infinity()), Vector3(-std::numeric_limits<float>::infinity())};
};

void Object::Private::load()
{
    // Load the scene
    const auto& importer = mesh_d->importer;
    if(importer->defaultScene() != -1)
    {
        Containers::Optional<Trade::SceneData> sceneData = importer->scene(importer->defaultScene());
        if(!sceneData)
        {
            throw Exception("Could not load scene data");
        }

        // Recursively add all children
        for(UnsignedInt objectId : sceneData->children3D())
            addMeshObject(sceneObject, objectId);
    }
    else if(!mesh_d->meshes.empty() && mesh_d->meshes[0])
    {
        // The format has no scene support, display just the first loaded mesh with
        // a default material and be done with it
        addMeshObject(sceneObject, 0);
    }
}

void Object::Private::addMeshObject(Object3D& parent, UnsignedInt i)
{
    std::unique_ptr<Trade::ObjectData3D> objectData = mesh_d->importer->object3D(i);
    if(!objectData)
    {
        Error{} << "Cannot import object, skipping";
        return;
    }

    // Add the object to the scene and set its transformation
    auto* object = new Object3D{&parent};
    object->setTransformation(objectData->transformation());

    // Add a drawable if the object has a mesh and the mesh is loaded
    if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && objectData->instance() != -1 && mesh_d->meshes[objectData->instance()])
    {
        const Int materialId = static_cast<Trade::MeshObjectData3D*>(objectData.get())->material();

        auto mesh = MeshTools::compile(*mesh_d->meshes[objectData->instance()]);

        auto drawable = new Drawable{*object, drawables, std::move(mesh)};

        if(materialId == -1 || !mesh_d->materials[materialId])
        {
            // Material not available / not loaded, use a default material
            drawable->setColor(0xffffff_rgbf);
        }
        else if(mesh_d->materials[materialId]->flags() & Trade::PhongMaterialData::Flag::DiffuseTexture)
        {
            // Textured material. If the texture failed to load, again just use a
            // default colored material.
            const Containers::Optional<GL::Texture2D>& texture = mesh_d->textures[mesh_d->materials[materialId]->diffuseTexture()];
            if(texture)
                drawable->setTexture(&*texture);
            else
                drawable->setColor(0xffffff_rgbf);
        }
        else
        {
            // Color-only material
            drawable->setColor(mesh_d->materials[materialId]->diffuseColor());
        }

        // Update bbox
        auto trans = object->absoluteTransformation();

        for(const auto& point : *mesh_d->meshPoints[objectData->instance()])
        {
            auto transformed = trans.transformPoint(point);

            bbox.min().x() = std::min(bbox.min().x(), transformed.x());
            bbox.min().y() = std::min(bbox.min().y(), transformed.y());
            bbox.min().z() = std::min(bbox.min().z(), transformed.z());

            bbox.max().x() = std::max(bbox.max().x(), transformed.x());
            bbox.max().y() = std::max(bbox.max().y(), transformed.y());
            bbox.max().z() = std::max(bbox.max().z(), transformed.z());
        }
    }

    // Recursively add children
    for(std::size_t id: objectData->children())
        addMeshObject(*object, id);
}


Object::Object()
 : m_d(std::make_unique<Private>())
{
}

std::shared_ptr<Object> Object::instantiate(const std::shared_ptr<Mesh>& mesh)
{
    auto object = std::make_shared<Object>();

    object->m_d->mesh = mesh;
    object->m_d->mesh_d = &mesh->impl();

    object->m_d->load();

    return object;
}

}
