// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh.h>
#include <stillleben/impl/mesh.h>
#include <stillleben/context.h>

#include <Corrade/Utility/Configuration.h>

#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Mesh.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/SceneGraph.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>
#include <Magnum/Image.h>

using namespace Magnum;

namespace sl
{

Mesh::Mesh(const std::shared_ptr<Context>& ctx)
 : m_d(std::make_unique<Private>(ctx))
{
}

Mesh::Mesh(sl::Mesh && other) = default;

Mesh::~Mesh()
{
}

void Mesh::load(const std::string& filename)
{
    // Load a scene importer plugin
    auto pluginManager = m_d->ctx->importerPluginManager();
    m_d->importer = pluginManager->loadAndInstantiate("AssimpImporter");
    if(!m_d->importer)
        std::exit(1);

    // Set up postprocess options if using AssimpImporter
    auto group = m_d->importer->configuration().group("postprocess");
    if(group)
    {
        group->setValue("JoinIdenticalVertices", true);
        group->setValue("Triangulate", true);
        group->setValue("GenSmoothNormals", true);
        group->setValue("PreTransformVertices", true);
        group->setValue("SortByPType", true);
        group->setValue("GenUVCoords", true);
        group->setValue("TransformUVCoords", true);
    }

    // Load file
    if(!m_d->importer->openFile(filename))
    {
        throw LoadException("Could not load file");
    }

    // Load all textures. Textures that fail to load will be NullOpt.
    m_d->textures = Containers::Array<Containers::Optional<GL::Texture2D>>{m_d->importer->textureCount()};
    for(UnsignedInt i = 0; i != m_d->importer->textureCount(); ++i)
    {
        Containers::Optional<Trade::TextureData> textureData = m_d->importer->texture(i);
        if(!textureData || textureData->type() != Trade::TextureData::Type::Texture2D)
        {
            Warning{} << "Cannot load texture properties, skipping";
            continue;
        }

        Containers::Optional<Trade::ImageData2D> imageData = m_d->importer->image2D(textureData->image());
        GL::TextureFormat format;
        if(imageData && imageData->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(imageData && imageData->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
        {
            Warning{} << "Cannot load texture image, skipping";
            continue;
        }

        // Configure the texture
        GL::Texture2D texture;
        texture
            .setMagnificationFilter(textureData->magnificationFilter())
            .setMinificationFilter(textureData->minificationFilter(), textureData->mipmapFilter())
            .setWrapping(textureData->wrapping().xy())
            .setStorage(Math::log2(imageData->size().max()) + 1, format, imageData->size())
            .setSubImage(0, {}, *imageData)
            .generateMipmap();

        m_d->textures[i] = std::move(texture);
    }

    // Load all materials. Materials that fail to load will be NullOpt. The
    // data will be stored directly in objects later, so save them only
    // temporarily.
    m_d->materials = Containers::Array<Containers::Optional<Trade::PhongMaterialData>>{m_d->importer->materialCount()};
    for(UnsignedInt i = 0; i != m_d->importer->materialCount(); ++i)
    {
        std::unique_ptr<Trade::AbstractMaterialData> materialData = m_d->importer->material(i);
        if(!materialData || materialData->type() != Trade::MaterialType::Phong)
        {
            Warning{} << "Cannot load material, skipping";
            continue;
        }

        m_d->materials[i] = std::move(static_cast<Trade::PhongMaterialData&>(*materialData));
    }

    // Load all meshes. Meshes that fail to load will be NullOpt.
    m_d->meshes = Containers::Array<Containers::Optional<Trade::MeshData3D>>{m_d->importer->mesh3DCount()};
    m_d->meshPoints = Containers::Array<Containers::Optional<std::vector<Vector3>>>{m_d->importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_d->importer->mesh3DCount(); ++i)
    {
        Containers::Optional<Trade::MeshData3D> meshData = m_d->importer->mesh3D(i);
        if(!meshData || !meshData->hasNormals() || meshData->primitive() != MeshPrimitive::Triangles)
        {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        std::vector<Vector3> points;
        for(std::size_t j = 0; j < meshData->positionArrayCount(); ++j)
        {
            auto array = meshData->positions(j);
            std::copy(array.begin(), array.end(), std::back_inserter(points));
        }

        // Compile the mesh
        m_d->meshes[i] = std::move(meshData);
        m_d->meshPoints[i] = points;
    }
}

}
