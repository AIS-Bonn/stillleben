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
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/RectangleTexture.h>

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
class Scene;

class Drawable;
typedef std::function<void(const Magnum::Matrix4& transformationMatrix, Magnum::SceneGraph::Camera3D& camera, Drawable* drawable)> DrawCallback;

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

    Magnum::Float metallic() const
    { return m_metallic; }

    Magnum::Float roughness() const
    { return m_roughness; }

    void setMetallicRoughness(Magnum::Float metallic, Magnum::Float roughness)
    { m_metallic = metallic; m_roughness = roughness; }

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
    Magnum::Float m_metallic = 0.5f;
    Magnum::Float m_roughness = 0.04f;
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
    void loadPhysicsVisualization();

    void serialize(Corrade::Utility::ConfigurationGroup& group);
    void deserialize(const Corrade::Utility::ConfigurationGroup& group, MeshCache& meshCache);

    void setPose(const Magnum::Matrix4& pose);
    Magnum::Matrix4 pose() const
    { return m_sceneObject.transformationMatrix(); }

    void setParentSceneObject(Object3D* parent);
    void setPhysicsScene(physx::PxScene* scene);

    void draw(Magnum::SceneGraph::Camera3D& camera, const DrawCallback& cb);

    void drawPhysics(Magnum::SceneGraph::Camera3D& camera, const DrawCallback& cb);

    float mass();
    void setMass(float mass);

    constexpr float density()
    { return m_density; }
    void setDensity(float density);

    float volume();

    void setLinearVelocityLimit(float velLimit);
    float linearVelocityLimit();

    void setLinearVelocity(const Magnum::Vector3& velocity);
    Magnum::Vector3 linearVelocity();

    void setAngularVelocity(const Magnum::Vector3& velocity);
    Magnum::Vector3 angularVelocity();


    void setInstanceIndex(unsigned int instanceIndex);
    unsigned int instanceIndex() const
    { return m_instanceIndex; }

    void setSpecularColor(const Magnum::Color4& color);
    constexpr Magnum::Color4 specularColor() const
    { return m_specularColor; }

    void setShininess(float shininess);
    constexpr float shininess() const
    { return m_shininess; }

    void setRoughness(float roughness);
    constexpr float roughness() const
    { return m_roughness; }

    void setMetallic(float metalness);
    constexpr float metallic() const
    { return m_metallic; }

    std::shared_ptr<Mesh> mesh()
    { return m_mesh; }

    Magnum::SceneGraph::DrawableGroup3D& debugDrawables()
    { return m_debugDrawables; }

    physx::PxRigidDynamic& rigidBody()
    { return *m_rigidBody; }

    void updateFromPhysics();

    void setStickerTexture(const std::shared_ptr<Magnum::GL::RectangleTexture>& color);
    std::shared_ptr<Magnum::GL::RectangleTexture> stickerTexture() const
    { return m_stickerTexture; }

    void setStickerRange(const Magnum::Range2D& range);
    constexpr Magnum::Range2D stickerRange() const
    { return m_stickerRange; }

    void setStickerRotation(const Magnum::Quaternion& q);
    constexpr Magnum::Quaternion stickerRotation() const
    { return m_stickerRotation; }

    Magnum::Matrix4 stickerViewProjection() const;

    void setStatic(bool isStatic);
    constexpr bool isStatic() const
    { return m_static; }

    constexpr float separation() const
    { return m_separation; }

private:
    friend class Scene;

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
    Magnum::SceneGraph::DrawableGroup3D m_physXDrawables;
    Magnum::SceneGraph::DrawableGroup3D m_simplifiedDrawables;

    DrawCallback m_cb;

    float m_scale = 1.0f;

    unsigned int m_instanceIndex = 0;

    physx::PxScene* m_physicsScene = nullptr;
    PhysXHolder<physx::PxRigidDynamic> m_rigidBody;

    bool m_visualLoaded = false;
    bool m_physicsVisLoaded = false;

    // By default, we have a fully specular object
    Magnum::Color4 m_specularColor{1.0f};

    // With sharp specular highlights
    float m_shininess = 80.0f;

    // These are factors onto the individual drawables
    float m_roughness = -1.0f;
    float m_metallic = -1.0f;

    // Sticker simulation
    std::shared_ptr<Magnum::GL::RectangleTexture> m_stickerTexture{};
    Magnum::Range2D m_stickerRange{};
    Magnum::Quaternion m_stickerRotation{};

    bool m_static = false;

    float m_density = 1000.0f;

    float m_separation = 0.0f;
    unsigned int m_stuckCounter = 0;
};

}

#endif
