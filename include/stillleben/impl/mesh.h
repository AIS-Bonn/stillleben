// Mesh implementation
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_IMPL_MESH_H
#define STILLLEBEN_IMPL_MESH_H

#include <stillleben/mesh.h>

#include <memory>

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

class Mesh::Private
{
public:
    Private(const std::shared_ptr<Context>& ctx)
     : ctx(ctx)
    {}

    std::shared_ptr<Context> ctx;


    std::unique_ptr<Magnum::Trade::AbstractImporter> importer;

    Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::MeshData3D>> meshes;
    Corrade::Containers::Array<Corrade::Containers::Optional<std::vector<Magnum::Vector3>>> meshPoints;
    Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::GL::Texture2D>> textures;
    Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::PhongMaterialData>> materials;
};

}

#endif
