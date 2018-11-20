// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_MESH_H
#define STILLLEBEN_MESH_H

#include <memory>

#include <stillleben/exception.h>

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/PluginManager.h>

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

private:
    std::shared_ptr<Context> m_ctx;

    std::unique_ptr<Magnum::Trade::AbstractImporter> m_importer;

    MeshArray m_meshes;
    PointArray m_meshPoints;
    TextureArray m_textures;
    MaterialArray m_materials;
};

}

#endif
