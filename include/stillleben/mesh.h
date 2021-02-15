// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_MESH_H
#define STILLLEBEN_MESH_H

#include <limits>
#include <memory>

#include <stillleben/exception.h>
#include <stillleben/mesh_tools/pbr_material_data.h>

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/Manager.h>

#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Range.h>

#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>
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

class ConsolidatedMesh;
class Context;
class PhysXOutputBuffer;

template<class T>
class PhysXHolder;

class Mesh
{
public:
    // Some imports from Corrade to keep the types shorter
    template<class T>
    using Array = Corrade::Containers::Array<T>;

    template<class T>
    using StridedArrayView1D = Corrade::Containers::StridedArrayView1D<T>;

    template<class T>
    using Optional = Corrade::Containers::Optional<T>;

    template<class T>
    using Pointer = Corrade::Containers::Pointer<T>;

    enum class MeshFlag
    {
        HasVertexColors = (1 << 0),
    };
    using MeshFlags = Corrade::Containers::EnumSet<MeshFlag>;

    using ObjectDataArray = Array<Pointer<Magnum::Trade::ObjectData3D>>;
    using MeshDataArray = Array<Optional<Magnum::Trade::MeshData>>;
    using MeshArray = Array<std::shared_ptr<Magnum::GL::Mesh>>;
    using MeshFlagArray = Array<MeshFlags>;
    using ImageDataArray = Array<Optional<Magnum::Trade::ImageData2D>>;
    using TextureDataArray = Array<Optional<Magnum::Trade::TextureData>>;
    using TextureArray = Array<Optional<Magnum::GL::Texture2D>>;
    using MaterialArray = Array<Optional<PBRMaterialData>>;

    using CookedPhysXMeshArray = Array<Optional<PhysXOutputBuffer>>;
    using PhysXMeshArray = Array<PhysXHolder<physx::PxConvexMesh>>;
    using PhysXVisArray = Array<std::shared_ptr<Magnum::GL::Mesh>>;

    class LoadException : public Exception
    {
        using Exception::Exception;
    };

    enum class Scale
    {
        Exact,
        OrderOfMagnitude,
    };

    enum class Flag
    {
        PhysicsForceConvexHull = 1 << 0
    };

    using Flags = Corrade::Containers::EnumSet<Flag>;

    Mesh(const std::string& filename, const std::shared_ptr<Context>& ctx, Flags flags = {});
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
    void load(bool visual = true, bool physics = true);

    static std::vector<std::shared_ptr<Mesh>> loadThreaded(
        const std::shared_ptr<Context>& ctx,
        const std::vector<std::string>& filenames,
        bool visual = true, bool physics = true
    );

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
    void loadPhysics();

    /**
     * @brief Load physics visualization meshes
     **/
    void loadPhysicsVisualization();

    /**
     * @brief Update vertex positions
     **/
    void updateVertexPositions(
        const Corrade::Containers::ArrayView<int>& verticesIndex,
        const Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate
    );

    /**
     * @brief Update vertex colors
     **/
    void updateVertexColors(
        const Corrade::Containers::ArrayView<int>& verticesIndex,
        const Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
    );

    /**
     * @brief Update vertex positions and colors
    **/
    void updateVertexPositionsAndColors(
        const Corrade::Containers::ArrayView<int>& verticesIndex,
        const Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate,
        const Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
    );

    /**
     * @brief Set vertex positions
     **/
    void setVertexPositions(
        const Corrade::Containers::ArrayView<Magnum::Vector3>& newVertices
    );

    /**
     * @brief Set vertex colors
     **/
    void setVertexColors(
        const Corrade::Containers::ArrayView<Magnum::Color4>& newColors
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

    const MeshArray& meshes() const
    { return m_meshes; }

    const MeshFlagArray& meshFlags() const
    { return m_meshFlags; }

    StridedArrayView1D<Magnum::Vector3> meshPoints(int subMesh = -1);
    StridedArrayView1D<Magnum::Vector3> meshNormals(int subMesh = -1);
    StridedArrayView1D<Magnum::UnsignedInt> meshFaces(int subMesh = -1);
    StridedArrayView1D<Magnum::Color4> meshColors(int subMesh = -1);

    const PhysXMeshArray& physXMeshes() const
    { return m_physXMeshes; }

    const PhysXVisArray& physXVisualizationMeshes() const
    { return m_physXVisMeshes; }

    // TextureArray can't be const because texture binding needs non-const reference.
    TextureArray& textures()
    { return m_textures; }

    const MaterialArray& materials() const
    { return m_materials; }

    const ObjectDataArray& objects() const
    { return m_objectData; }

    const Optional<Magnum::Trade::SceneData>& sceneData() const
    { return m_sceneData; }

    void setClassIndex(unsigned int index);
    unsigned int classIndex() const
    { return m_classIndex; }

    std::string filename() const
    { return m_filename; }


private:
    void updateBoundingBox();

    void updatePretransform();

    void loadPretransform(const std::string& filename);

    void recomputeNormals();
    void recompileMesh();

    std::shared_ptr<Context> m_ctx;

    std::string m_filename;

    Flags m_flags;

    bool m_opened = false;
    bool m_visualLoaded = false;
    bool m_physicsLoaded = false;
    bool m_physicsVisLoaded = false;

    Optional<Magnum::Trade::SceneData> m_sceneData;
    ObjectDataArray m_objectData;
    MeshArray m_meshes;
    Pointer<ConsolidatedMesh> m_consolidated;
    MeshFlagArray m_meshFlags;
    ImageDataArray m_imageData;
    TextureDataArray m_textureData;
    TextureArray m_textures;
    MaterialArray m_materials;
    Pointer<sl::PhysXOutputBuffer> m_physXBuffer;
    PhysXMeshArray m_physXMeshes;
    PhysXVisArray m_physXVisMeshes;

    Magnum::Range3D m_bbox{
        Magnum::Vector3(std::numeric_limits<float>::infinity()),
        Magnum::Vector3(-std::numeric_limits<float>::infinity())
    };

    float m_scale = 1.0f;

    Magnum::Matrix4 m_pretransformRigid;
    Magnum::Matrix4 m_pretransform;

    unsigned int m_classIndex = 1;

    Magnum::GL::Buffer m_vertexBuffer{Magnum::NoCreate};
    Magnum::GL::Buffer m_indexBuffer{Magnum::NoCreate};
};

CORRADE_ENUMSET_OPERATORS(Mesh::Flags)

}

#endif
