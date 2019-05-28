// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_OBJECT_H
#define STILLLEBEN_OBJECT_H

#include <stillleben/common.h>
#include <stillleben/math.h>
#include <stillleben/physx.h>

#include <Magnum/Math/Range.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/GL/Mesh.h>

#include <memory>

namespace physx
{
    class PxScene;
    class PxRigidDynamic;
}

namespace sl
{

class Context;
class Mesh;
class MeshCache;

class Drawable;
typedef std::function<void(const Magnum::Matrix4& transformationMatrix, Magnum::SceneGraph::Camera3D& camera, Drawable* drawable)> DrawCallback;

class ObjectPart : public Object3D
{
public:

};

class Drawable : public Magnum::SceneGraph::Drawable3D
{
public:
    Drawable(Object3D& object, Magnum::SceneGraph::DrawableGroup3D& group, const std::shared_ptr<Magnum::GL::Mesh>& mesh, DrawCallback* cb)
     : Magnum::SceneGraph::Drawable3D{object, &group}
     , m_mesh{mesh}
     , m_cb(cb)
    {
    }

    inline Magnum::GL::Texture2D* texture()
    { return m_texture; }
    void setTexture(Magnum::GL::Texture2D* texture)
    { m_texture = texture; }

    Magnum::Color4 color()
    { return m_color; }
    void setColor(const Magnum::Color4& color)
    { m_color = color; }

    Magnum::GL::Mesh& mesh()
    { return *m_mesh; }

    void setHasVertexColors(bool hasVertexColors)
    { m_hasVertexColors = hasVertexColors; }

    inline bool hasVertexColors() const
    { return m_hasVertexColors; }

    void draw(const Magnum::Matrix4& transformationMatrix, Magnum::SceneGraph::Camera3D& camera) override;
private:
    std::shared_ptr<Magnum::GL::Mesh> m_mesh;
    Magnum::GL::Texture2D* m_texture = nullptr;
    Magnum::Color4 m_color{};
    DrawCallback* m_cb = nullptr;
    bool m_hasVertexColors = false;
};

struct InstantiationOptions
{
    /**
     * Default color if the mesh does not have texture or vertex colors
     **/
    Magnum::Color4 color{1.0f, 1.0f, 1.0f, 1.0f};

    /**
     * If true, always render the mesh with this color, regardless of texture
     **/
    bool forceColor = false;
};

class Object
{
public:
    class Part : public Object3D
    {
    public:
        explicit Part(Magnum::UnsignedInt index, Object3D* parent)
         : Object3D{parent}
         , m_index{index}
        {}

        ~Part() = default;

        constexpr Magnum::UnsignedInt index()
        { return m_index; }
    private:
        Magnum::UnsignedInt m_index;
    };


    Object();
    ~Object();

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    void setMesh(const std::shared_ptr<Mesh>& mesh);
    void setInstantiationOptions(const InstantiationOptions& options);

    void loadVisual();
    void loadPhysics();

    void serialize(Corrade::Utility::ConfigurationGroup& group);
    void deserialize(const Corrade::Utility::ConfigurationGroup& group, MeshCache& meshCache);

    void setPose(const Magnum::Matrix4& pose);
    Magnum::Matrix4 pose() const
    { return m_sceneObject.transformationMatrix(); }

    void setParentSceneObject(Object3D* parent);
    void setPhysicsScene(physx::PxScene* scene);

    void draw(Magnum::SceneGraph::Camera3D& camera, const DrawCallback& cb);

    void setInstanceIndex(unsigned int instanceIndex);
    unsigned int instanceIndex() const
    { return m_instanceIndex; }

    std::shared_ptr<Mesh> mesh()
    { return m_mesh; }

    Magnum::SceneGraph::DrawableGroup3D& debugDrawables()
    { return m_debugDrawables; }

    physx::PxRigidDynamic& rigidBody()
    { return *m_rigidBody; }

    void updateFromPhysics();

private:
    void populateParts();
    void addPart(Object3D& parent, Magnum::UnsignedInt i);

    std::shared_ptr<Mesh> m_mesh;
    InstantiationOptions m_options;

    // This is the scene object that contains everything in this object.
    // setPose() acts upon this object.
    Object3D m_sceneObject;

    // This holds the actual mesh.
    // Mesh::scaleToBBoxDiagonal() acts upon this object.
    Object3D m_meshObject{&m_sceneObject};

    std::vector<Part*> m_parts;

    Magnum::SceneGraph::DrawableGroup3D m_drawables;
    Magnum::SceneGraph::DrawableGroup3D m_debugDrawables;

    DrawCallback m_cb;

    float m_scale = 1.0f;

    unsigned int m_instanceIndex = 0;

    physx::PxScene* m_physicsScene = nullptr;
    PhysXHolder<physx::PxRigidDynamic> m_rigidBody;

    bool m_visualLoaded = false;
};

}

#endif
