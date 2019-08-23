// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_MESH_H
#define STILLLEBEN_MESH_H

#include <limits>
#include <memory>

#include <stillleben/exception.h>

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/PluginManager.h>

#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Range.h>

#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>

#include <Magnum/GL/Texture.h>

class btCollisionShape;

namespace physx
{
    class PxConvexMesh;
}

namespace sl
{

class Context;
class PhysXOutputBuffer;

template<class T>
class PhysXHolder;

class Mesh
{
public:
    enum class MeshFlag
    {
        HasVertexColors = (1 << 0),
    };
    using MeshFlags = Corrade::Containers::EnumSet<MeshFlag>;

    using MeshArray = Corrade::Containers::Array<std::shared_ptr<Magnum::GL::Mesh>>;
    using MeshFlagArray = Corrade::Containers::Array<MeshFlags>;
    using PointArray = Corrade::Containers::Array<Corrade::Containers::Optional<std::vector<Magnum::Vector3>>>;
    using CollisionArray = Corrade::Containers::Array<std::shared_ptr<btCollisionShape>>;
    using TextureArray = Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::GL::Texture2D>>;
    using MaterialArray = Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::PhongMaterialData>>;
    using SimplifiedMeshArray = Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::MeshData3D>>;

    using CookedPhysXMeshArray = Corrade::Containers::Array<Corrade::Containers::Optional<PhysXOutputBuffer>>;
    using PhysXMeshArray = Corrade::Containers::Array<Corrade::Containers::Optional<PhysXHolder<physx::PxConvexMesh>>>;

    static constexpr std::size_t DefaultPhysicsTriangles = 2000;

    class LoadException : public Exception
    {
        using Exception::Exception;
    };

    enum class Scale
    {
        Exact,
        OrderOfMagnitude,
    };

    Mesh(const std::string& filename, const std::shared_ptr<Context>& ctx);
    Mesh(const Mesh& other) = delete;
    Mesh(Mesh&& other);
    ~Mesh();

    std::shared_ptr<Context>& context()
    { return m_ctx; }

    void serialize(Corrade::Utility::ConfigurationGroup& group);
    void deserialize(const Corrade::Utility::ConfigurationGroup& group);

    //! @name Loading mesh data
    //@{

    /**
     * @brief Load everything
     *
     * This is your one-catch-all method: Loads visual & collision meshes.
     **/
    void load(std::size_t maxPhysicsTriangles = DefaultPhysicsTriangles, bool visual = true, bool physics = true);

    /**
     * @brief Open input file & preprocess
     **/
    void openFile();

    /**
     * @brief Load visual meshes onto the GPU
     *
     * This needs to be called from the main thread.
     **/
    void loadVisual();

    /**
     * @brief Load physics meshes
     **/
    void loadPhysics(std::size_t maxPhysicsTriangles = DefaultPhysicsTriangles);

    static std::vector<std::shared_ptr<Mesh>> loadThreaded(
        const std::shared_ptr<Context>& ctx,
        const std::vector<std::string>& filenames,
        bool visual = true, bool physics = true,
        std::size_t maxPhysicsTriangles = DefaultPhysicsTriangles
    );

    //@}

    Magnum::Range3D bbox() const;

    void centerBBox();
    void scaleToBBoxDiagonal(float targetDiagonal, Scale mode = Scale::Exact);

    const Magnum::Matrix4& pretransformRigid() const
    { return m_pretransformRigid; }

    float pretransformScale() const
    { return m_scale; }

    const Magnum::Matrix4& pretransform() const
    { return m_pretransform; }

    /**
     * @brief Set pretransform from matrix
     *
     * Since internally the pretransform is represented as a rigid
     * transformation plus scaling, the 4x4 matrix is decomposed into these
     * components.
     **/
    void setPretransform(const Magnum::Matrix4& m);

    Magnum::Trade::AbstractImporter& importer()
    { return *m_importer; }

    MeshArray& meshes()
    { return m_meshes; }

    MeshFlagArray& meshFlags()
    { return m_meshFlags; }

    PointArray& meshPoints()
    { return m_meshPoints; }

    CollisionArray& collisionShapes()
    { return m_collisionShapes; }

    PhysXMeshArray& physXMeshes()
    { return m_physXMeshes; }

    TextureArray& textures()
    { return m_textures; }

    MaterialArray& materials()
    { return m_materials; }

    void setClassIndex(unsigned int index);
    unsigned int classIndex() const
    { return m_classIndex; }

    std::string filename() const
    { return m_filename; }

private:
    void updateBoundingBox(const Magnum::Matrix4& transform, unsigned int meshObjectIdx);
    void updatePretransform();

    void loadPretransform(const std::string& filename);

    std::shared_ptr<Context> m_ctx;

    std::string m_filename;

    std::unique_ptr<Magnum::Trade::AbstractImporter> m_importer;

    bool m_visualLoaded = false;
    bool m_physicsLoaded = false;

    MeshArray m_meshes;
    MeshFlagArray m_meshFlags;
    PointArray m_meshPoints;
    CollisionArray m_collisionShapes;
    TextureArray m_textures;
    MaterialArray m_materials;
    SimplifiedMeshArray m_simplifiedMeshes;
    CookedPhysXMeshArray m_physXBuffers;
    PhysXMeshArray m_physXMeshes;

    Magnum::Range3D m_bbox{
        Magnum::Vector3(std::numeric_limits<float>::infinity()),
        Magnum::Vector3(-std::numeric_limits<float>::infinity())
    };

    float m_scale = 1.0f;

    Magnum::Matrix4 m_pretransformRigid;
    Magnum::Matrix4 m_pretransform;

    unsigned int m_classIndex = 1;
};

}

#endif
