// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_MESH_H
#define STILLLEBEN_MESH_H

#include <limits>
#include <memory>

#include <stillleben/exception.h>

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/PluginManager.h>

#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Range.h>

#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>

#include <Magnum/GL/Texture.h>
#include <Magnum/GL/Buffer.h>

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
    using NormalArray = Corrade::Containers::Array<Corrade::Containers::Optional<std::vector<Magnum::Vector3>>>;
    using FaceArray = Corrade::Containers::Array<Corrade::Containers::Optional<std::vector<Magnum::UnsignedInt>>>;
    using ColorArray = Corrade::Containers::Array<Corrade::Containers::Optional<std::vector<Magnum::Color4>>>;
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

    /**
     * @brief Update vertex positions
     **/
    void updateVertexPositions(
        Corrade::Containers::ArrayView<int>& verticesIndex,
        Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate
    );

    /**
     * @brief Update vertex colors
     **/
    void updateVertexColors(
        Corrade::Containers::ArrayView<int>& verticesIndex,
        Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
    );

    /**
     * @brief Update vertex positions and colors
    **/
    void updateVertexPositionsAndColors(
        Corrade::Containers::ArrayView<int>& verticesIndex,
        Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate,
        Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
    );

    /**
     * @brief Set vertex positions
     **/
    void setVertexPositions(
        Corrade::Containers::ArrayView<Magnum::Vector3>& newVertices
    );

    /**
     * @brief Set vertex colors
     **/
    void setVertexColors(
        Corrade::Containers::ArrayView<Magnum::Color4>& newColors
    );

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

    const std::vector<Magnum::Vector3>& meshPoints()
    {
        if(m_meshPoints.size() == 0)
            throw Exception{"sl::Mesh contains multiple meshes, but sl::Mesh::meshPoints supports only single meshes"};

        if(!m_meshPoints[0])
            throw Exception{"Submesh 0 has no points"};

        return *m_meshPoints[0];
    }

    const std::vector<Magnum::Vector3>& meshNormals()
    {
        if(m_meshPoints.size() == 0)
            throw Exception{"sl::Mesh contains multiple meshes, but sl::Mesh::meshNormals supports only single meshes"};

        if(!m_meshNormals[0])
            throw Exception{"Submesh 0 has no points"};

        return *m_meshNormals[0];
    }

    const std::vector<Magnum::UnsignedInt>& meshFaces()
    {
        if(m_meshPoints.size() == 0)
            throw Exception{"sl::Mesh contains multiple meshes, but sl::Mesh::meshFaces supports only single meshes"};

        if(!m_meshFaces[0])
            throw Exception{"Submesh 0 has no points"};

        return *m_meshFaces[0];
    }

    const std::vector<Magnum::Color4>& meshColors()
    {
        if(m_meshPoints.size() == 0)
            throw Exception{"sl::Mesh contains multiple meshes, but sl::Mesh::meshColors supports only single meshes"};

        if(!m_meshFaces[0])
            throw Exception{"Submesh 0 has no points"};

        return *m_meshColors[0];
    }

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

    unsigned int numVertices() const
    { return m_numVertices; }


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
    NormalArray m_meshNormals;
    FaceArray m_meshFaces;
    ColorArray m_meshColors;
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

    Magnum::GL::Buffer m_vertexIndexBuf{Magnum::NoCreate};

    Magnum::Containers::Optional<Magnum::Trade::MeshData3D> m_meshData;
    unsigned int m_numVertices;

    Corrade::Containers::Array<Magnum::UnsignedInt> m_vertexIndices;
};

}

#endif
