// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh.h>
#include <stillleben/context.h>

#include <btBulletDynamicsCommon.h>

#include <Corrade/Utility/Configuration.h>

#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/SceneGraph.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>
#include <Magnum/Image.h>

#include <sstream>

using namespace Magnum;

namespace sl
{

/**
 * @brief Create bullet collision shape from Trade::MeshData3D
 *
 * @warning The resulting collision shape references the original mesh data,
 *   so the MeshData3D instance needs to be kept around!
 **/
static std::shared_ptr<btCollisionShape> collisionShapeFromMeshData(
    const Trade::MeshData3D& meshData, bool convexHull = true)
{
    // Source: https://github.com/mosra/magnum-integration/issues/20#issuecomment-246951535

    if(meshData.primitive() != MeshPrimitive::Triangles)
    {
        Error() << "Cannot load collision mesh, skipping";
        return {};
    }

    if(convexHull)
    {
        auto shape = std::make_shared<btConvexHullShape>(
            reinterpret_cast<const float*>(meshData.positions(0).data()),
            meshData.positions(0).size(),
            sizeof(Vector3)
        );

        shape->optimizeConvexHull();

        return shape;
    }
    else
    {
        btIndexedMesh bulletMesh;
        bulletMesh.m_numTriangles = meshData.indices().size()/3;
        bulletMesh.m_triangleIndexBase = reinterpret_cast<const unsigned char *>(meshData.indices().data());
        bulletMesh.m_triangleIndexStride = 3 * sizeof(UnsignedInt);
        bulletMesh.m_numVertices = meshData.positions(0).size();
        bulletMesh.m_vertexBase = reinterpret_cast<const unsigned char *>(meshData.positions(0).data());
        bulletMesh.m_vertexStride = sizeof(Vector3);
        bulletMesh.m_indexType = PHY_INTEGER;
        bulletMesh.m_vertexType = PHY_FLOAT;

        Debug{} << "Creating btBvhTriangleMeshShape with numTriangles:"
                << bulletMesh.m_numTriangles
                << "and numVertices:"
                << bulletMesh.m_numVertices
        ;

        auto tivArray = new btTriangleIndexVertexArray();
        tivArray->addIndexedMesh(bulletMesh, PHY_INTEGER);

        return std::shared_ptr<btBvhTriangleMeshShape>(
            new btBvhTriangleMeshShape(tivArray, true),
            [&](btBvhTriangleMeshShape* b) {
                delete b;
                delete tivArray;
            }
        );
    }
}

Mesh::Mesh(const std::shared_ptr<Context>& ctx)
 : m_ctx{ctx}
{
}

Mesh::Mesh(sl::Mesh && other) = default;

Mesh::~Mesh()
{
}

void Mesh::load(const std::string& filename)
{
    // Load a scene importer plugin
    m_importer = m_ctx->instantiateImporter();
    if(!m_importer)
        std::abort();

    // Set up postprocess options if using AssimpImporter
    auto group = m_importer->configuration().group("postprocess");
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
    {
        std::stringstream ss;
        Corrade::Utility::Error redirect{&ss};
        if(!m_importer->openFile(filename))
        {
            throw LoadException(ss.str() + "\nCould not load file");
        }
    }

    // Load all textures. Textures that fail to load will be NullOpt.
    m_textures = Containers::Array<Containers::Optional<GL::Texture2D>>{m_importer->textureCount()};
    for(UnsignedInt i = 0; i != m_importer->textureCount(); ++i)
    {
        Containers::Optional<Trade::TextureData> textureData = m_importer->texture(i);
        if(!textureData || textureData->type() != Trade::TextureData::Type::Texture2D)
        {
            Warning{} << "Cannot load texture properties, skipping";
            continue;
        }

        Containers::Optional<Trade::ImageData2D> imageData = m_importer->image2D(textureData->image());
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

        m_textures[i] = std::move(texture);
    }

    // Load materials.
    m_materials = Containers::Array<Containers::Optional<Trade::PhongMaterialData>>{m_importer->materialCount()};
    for(UnsignedInt i = 0; i != m_importer->materialCount(); ++i)
    {
        std::unique_ptr<Trade::AbstractMaterialData> materialData = m_importer->material(i);
        if(!materialData || materialData->type() != Trade::MaterialType::Phong)
        {
            Warning{} << "Cannot load material, skipping";
            continue;
        }

        m_materials[i] = std::move(static_cast<Trade::PhongMaterialData&>(*materialData));
    }

    // Load all meshes. Meshes that fail to load will be NullOpt.
    m_meshes = Containers::Array<std::shared_ptr<GL::Mesh>>{m_importer->mesh3DCount()};
    m_meshPoints = Containers::Array<Containers::Optional<std::vector<Vector3>>>{m_importer->mesh3DCount()};
    m_collisionShapes = CollisionArray{m_importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_importer->mesh3DCount(); ++i)
    {
        Containers::Optional<Trade::MeshData3D> meshData = m_importer->mesh3D(i);
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

        m_meshes[i] = std::make_shared<GL::Mesh>(
            MeshTools::compile(*meshData)
        );
        m_meshPoints[i] = points;

        m_collisionShapes[i] = collisionShapeFromMeshData(*meshData);
    }

    // Update the bounding box
    {
        // Inspect the scene if available
        if(m_importer->defaultScene() != -1)
        {
            auto sceneData = m_importer->scene(m_importer->defaultScene());
            if(!sceneData)
                throw Exception("Could not load scene data");

            // Recursively inspect all children
            for(UnsignedInt objectId : sceneData->children3D())
                updateBoundingBox(Matrix4{}, objectId);
        }
        else if(!m_meshes.empty() && m_meshes[0])
        {
            // The format has no scene support, use first mesh
            updateBoundingBox(Matrix4{}, 0);
        }
    }
}

void Mesh::updateBoundingBox(const Magnum::Matrix4& parentTransform, unsigned int meshObjectIdx)
{
    std::unique_ptr<Trade::ObjectData3D> objectData = m_importer->object3D(meshObjectIdx);
    if(!objectData)
        return;

    Magnum::Matrix4 transform = parentTransform * objectData->transformation();

    if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && objectData->instance() != -1 && m_meshes[objectData->instance()])
    {
        for(const auto& point : *m_meshPoints[objectData->instance()])
        {
            auto transformed = transform.transformPoint(point);

            m_bbox.min().x() = std::min(m_bbox.min().x(), transformed.x());
            m_bbox.min().y() = std::min(m_bbox.min().y(), transformed.y());
            m_bbox.min().z() = std::min(m_bbox.min().z(), transformed.z());

            m_bbox.max().x() = std::max(m_bbox.max().x(), transformed.x());
            m_bbox.max().y() = std::max(m_bbox.max().y(), transformed.y());
            m_bbox.max().z() = std::max(m_bbox.max().z(), transformed.z());
        }
    }

    // Recurse
    for(std::size_t idx : objectData->children())
        updateBoundingBox(transform, idx);
}

void Mesh::centerBBox()
{
    m_translation = -m_bbox.center();
    updatePretransform();
}

void Mesh::scaleToBBoxDiagonal(float targetDiagonal, Scale mode)
{
    float diagonal = m_bbox.size().length();

    float scale = targetDiagonal / diagonal;

    switch(mode)
    {
        case Scale::Exact:
            m_scale = scale;
            break;
        case Scale::OrderOfMagnitude:
            m_scale = std::pow(10, std::round(std::log10(scale)));
            break;
    }

    updatePretransform();
}

void Mesh::updatePretransform()
{
    m_pretransformRigid = Matrix4::translation(m_translation);
    m_pretransform = Matrix4::scaling(Vector3(m_scale)) * m_pretransformRigid;
}

Magnum::Range3D Mesh::bbox() const
{
    Vector3 lower = m_pretransform.transformPoint(m_bbox.min());
    Vector3 upper = m_pretransform.transformPoint(m_bbox.max());

    return Range3D{lower, upper};
}

void Mesh::setClassIndex(unsigned int index)
{
    if(index > std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument("Mesh::setClassIndex(): out of range");

    m_classIndex = index;
}

}
