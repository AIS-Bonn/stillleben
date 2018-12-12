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

namespace sl
{

class Context;

class Mesh
{
public:
    using MeshArray = Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::MeshData3D>>;
    using PointArray = Corrade::Containers::Array<Corrade::Containers::Optional<std::vector<Magnum::Vector3>>>;
    using TextureArray = Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::GL::Texture2D>>;
    using MaterialArray = Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::PhongMaterialData>>;

    class LoadException : public Exception
    {
        using Exception::Exception;
    };

    Mesh(const std::shared_ptr<Context>& ctx);
    Mesh(const Mesh& other) = delete;
    Mesh(Mesh&& other);
    ~Mesh();

    void load(const std::string& filename);

    Magnum::Range3D bbox() const;

    void centerBBox();
    void scaleToBBoxDiagonal(float targetDiagonal);

    const Magnum::Matrix4& pretransform() const
    { return m_pretransform; }

    Magnum::Trade::AbstractImporter& importer()
    { return *m_importer; }

    MeshArray& meshes()
    { return m_meshes; }

    PointArray& meshPoints()
    { return m_meshPoints; }

    TextureArray& textures()
    { return m_textures; }

    MaterialArray& materials()
    { return m_materials; }

    void setClassIndex(unsigned int index);
    unsigned int classIndex() const
    { return m_classIndex; }

private:
    void updateBoundingBox(const Magnum::Matrix4& transform, unsigned int meshObjectIdx);
    void updatePretransform();

    std::shared_ptr<Context> m_ctx;

    std::unique_ptr<Magnum::Trade::AbstractImporter> m_importer;

    MeshArray m_meshes;
    PointArray m_meshPoints;
    TextureArray m_textures;
    MaterialArray m_materials;

    Magnum::Range3D m_bbox{
        Magnum::Vector3(std::numeric_limits<float>::infinity()),
        Magnum::Vector3(-std::numeric_limits<float>::infinity())
    };

    Magnum::Vector3 m_translation{Magnum::Math::ZeroInit};
    float m_scale = 1.0f;

    Magnum::Matrix4 m_pretransform;

    unsigned int m_classIndex = 1;
};

}

#endif
