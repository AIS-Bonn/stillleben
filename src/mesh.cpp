// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh.h>
#include <stillleben/context.h>
#include <stillleben/contrib/ctpl_stl.h>
#include <stillleben/mesh_tools/simplify_mesh.h>
#include <stillleben/physx.h>

#include <Corrade/Utility/Configuration.h>

#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
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

#include "physx_impl.h"

using namespace Magnum;

namespace sl
{

/**
 * @brief Create PhysX collision shape from Trade::MeshData3D
 * */
static Corrade::Containers::Optional<PhysXOutputBuffer> cookForPhysX(physx::PxCooking& cooking, const Trade::MeshData3D& meshData)
{
    if(meshData.primitive() != MeshPrimitive::Triangles)
    {
        Error{} << "Cannot load collision mesh, skipping";
        return {};
    }

    if(meshData.positionArrayCount() > 1)
    {
        Warning{} << "Mesh has more than one position array, this is unsupported";
    }

    Corrade::Containers::Optional<PhysXOutputBuffer> out;

    out.emplace();

    static_assert(sizeof(decltype(*meshData.positions(0).data())) == sizeof(physx::PxVec3));

    physx::PxConvexMeshDesc meshDesc;
    meshDesc.points.count = meshData.positions(0).size();
    meshDesc.points.stride = sizeof(physx::PxVec3);
    meshDesc.points.data = meshData.positions(0).data();
    meshDesc.flags = physx::PxConvexFlag::eCOMPUTE_CONVEX;

    physx::PxConvexMeshCookingResult::Enum result;
    bool status = cooking.cookConvexMesh(meshDesc, *out, &result);
    if(!status)
    {
        Error{} << "PhysX cooking failed, ignoring mesh";
        return {};
    }

    return out;
}

Mesh::Mesh(const std::shared_ptr<Context>& ctx)
 : m_ctx{ctx}
{
}

Mesh::Mesh(sl::Mesh && other) = default;

Mesh::~Mesh()
{
}

void Mesh::load(const std::string& filename, std::size_t maxPhysicsTriangles)
{
    loadNonGL(filename, maxPhysicsTriangles);
    loadGL();
}

void Mesh::loadNonGL(const std::string& filename, std::size_t maxPhysicsTriangles)
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
        if(!m_importer->openFile(filename))
        {
            throw LoadException("Could not load file");
        }
    }

    // Simplify meshes if possible
    m_simplifiedMeshes = SimplifiedMeshArray{m_importer->mesh3DCount()};
    m_collisionShapes = CollisionArray{m_importer->mesh3DCount()};
    m_physXBuffers = CookedPhysXMeshArray{m_importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_importer->mesh3DCount(); ++i)
    {
        Containers::Optional<Trade::MeshData3D> meshData = m_importer->mesh3D(i);
        if(!meshData || meshData->primitive() != MeshPrimitive::Triangles)
        {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        Trade::MeshData3D simplifiedMesh{
            MeshPrimitive::Triangles,
            meshData->indices(),
            {meshData->positions(0)},
            {}, {}, {}
        };

        if(meshData->indices().size()/3 > maxPhysicsTriangles)
        {
            // simplify in-place
            mesh_tools::QuadricEdgeSimplification<Magnum::Vector3> simplification{
                simplifiedMesh.indices(), simplifiedMesh.positions(0)
            };

            simplification.simplify(maxPhysicsTriangles);
        }

        m_simplifiedMeshes[i] = std::move(simplifiedMesh);
//         m_collisionShapes[i] = collisionShapeFromMeshData(*m_simplifiedMeshes[i]);
        m_physXBuffers[i] = cookForPhysX(m_ctx->physxCooking(), *m_simplifiedMeshes[i]);
    }
}

void Mesh::loadGL()
{
    if(!m_importer)
        throw std::logic_error("You need to call loadNonGL() first");

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
    m_physXMeshes = PhysXMeshArray{m_importer->mesh3DCount()};
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

        if(auto& buf = m_physXBuffers[i])
        {
            physx::PxDefaultMemoryInputData stream(buf->data(), buf->size());
            m_physXMeshes[i] = PhysXHolder<physx::PxConvexMesh>{
                m_ctx->physxPhysics().createConvexMesh(stream)
            };
        }
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

namespace
{
    std::shared_ptr<Mesh> loadHelper(
        const std::shared_ptr<Context>& ctx,
        const std::string& filename,
        std::size_t maxPhysicsTriangles)
    {
        auto mesh = std::make_shared<Mesh>(ctx);
        mesh->loadNonGL(filename, maxPhysicsTriangles);

        return mesh;
    }
}

std::vector<std::shared_ptr<Mesh>> Mesh::loadThreaded(
    const std::shared_ptr<Context>& ctx,
    const std::vector<std::string>& filenames,
    std::size_t maxPhysicsTriangles)
{
    using Future = std::future<std::shared_ptr<sl::Mesh>>;

    ctpl::thread_pool pool(std::thread::hardware_concurrency());

    std::vector<Future> results;
    for(const auto& filename : filenames)
    {
        results.push_back(pool.push(std::bind(&loadHelper,
            ctx, filename, maxPhysicsTriangles
        )));
    }

    std::vector<std::shared_ptr<sl::Mesh>> ret;
    for(auto& future : results)
    {
        auto mesh = future.get(); // may throw if we had a load error

        mesh->loadGL();

        ret.push_back(std::move(mesh));
    }

    return ret;
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
