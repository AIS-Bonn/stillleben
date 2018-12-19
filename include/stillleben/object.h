// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_OBJECT_H
#define STILLLEBEN_OBJECT_H

#include <stillleben/math.h>

#include <memory>
#include <limits>

#include <stillleben/common.h>

#include <Magnum/Math/Range.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/GL/Mesh.h>

namespace sl
{

class Mesh;

class Drawable;
typedef std::function<void(const Magnum::Matrix4& transformationMatrix, Magnum::SceneGraph::Camera3D& camera, Drawable* drawable)> DrawCallback;

class Drawable : public Magnum::SceneGraph::Drawable3D
{
public:
    Drawable(Object3D& object, Magnum::SceneGraph::DrawableGroup3D& group, Magnum::GL::Mesh&& mesh, DrawCallback* cb)
     : Magnum::SceneGraph::Drawable3D{object, &group}
     , m_mesh{std::move(mesh)}
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
    { return m_mesh; }

    void draw(const Magnum::Matrix4& transformationMatrix, Magnum::SceneGraph::Camera3D& camera) override;
private:
    Magnum::GL::Mesh m_mesh;
    Magnum::GL::Texture2D* m_texture = nullptr;
    Magnum::Color4 m_color{};
    DrawCallback* m_cb = nullptr;
};

class Object
{
public:
    Object();
    ~Object();

    static std::shared_ptr<Object> instantiate(const std::shared_ptr<Mesh>& mesh);

    void setPose(const PoseMatrix& pose);
    PoseMatrix pose() const
    { return m_sceneObject.transformationMatrix(); }

    void setParentSceneObject(Object3D* parent);

    void draw(Magnum::SceneGraph::Camera3D& camera, const DrawCallback& cb);

    void setInstanceIndex(unsigned int instanceIndex);
    unsigned int instanceIndex() const
    { return m_instanceIndex; }

    std::shared_ptr<Mesh> mesh()
    { return m_mesh; }

    Magnum::SceneGraph::DrawableGroup3D& debugDrawables()
    { return m_debugDrawables; }

private:
    void load();
    void addMeshObject(Object3D& parent, Magnum::UnsignedInt i);

    std::shared_ptr<Mesh> m_mesh;

    // This is the scene object that contains everything in this object.
    // setPose() acts upon this object.
    Object3D m_sceneObject;

    // This holds the actual mesh.
    // scaleToBBoxDiagonal() acts upon this object.
    Object3D m_meshObject{&m_sceneObject};

    Magnum::SceneGraph::DrawableGroup3D m_drawables;
    Magnum::SceneGraph::DrawableGroup3D m_debugDrawables;
    Magnum::Range3D m_bbox{
        Magnum::Vector3(std::numeric_limits<float>::infinity()),
        Magnum::Vector3(-std::numeric_limits<float>::infinity())
    };

    DrawCallback m_cb;

    float m_scale = 1.0f;

    unsigned int m_instanceIndex = 0;
};

}

#endif
