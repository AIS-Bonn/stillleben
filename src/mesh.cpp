// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh.h>
#include <stillleben/context.h>
#include <stillleben/contrib/ctpl_stl.h>
#include <stillleben/mesh_tools/simplify_mesh.h>
#include <stillleben/physx.h>

#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Format.h>
#include <Corrade/Utility/FormatStl.h>

#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/ConfigurationValue.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Math/Algorithms/Svd.h>
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
#include <fstream>
#include <experimental/filesystem>

#include <unistd.h>
#include <sys/stat.h>

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

Mesh::Mesh(const std::string& filename, const std::shared_ptr<Context>& ctx)
 : m_ctx{ctx}
 , m_filename{filename}
{
}

Mesh::Mesh(sl::Mesh && other) = default;

Mesh::~Mesh()
{
}

void Mesh::load(std::size_t maxPhysicsTriangles, bool visual, bool physics)
{
    openFile();

    if(physics)
        loadPhysics(maxPhysicsTriangles);

    if(visual)
        loadVisual();
}

void Mesh::openFile()
{
    if(m_importer)
        return;

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
        if(!m_importer->openFile(m_filename))
        {
            m_importer.reset();
            throw LoadException("Could not load file: " + m_filename);
        }
    }

    // Load pretransform, if available
    loadPretransform(m_filename + ".pretransform");

    // Compute bounding box
    m_meshPoints = Containers::Array<Containers::Optional<std::vector<Vector3>>>{m_importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_importer->mesh3DCount(); ++i)
    {
        Containers::Optional<Trade::MeshData3D> meshData = m_importer->mesh3D(i);
        if(!meshData || !meshData->hasNormals() || meshData->primitive() != MeshPrimitive::Triangles)
            continue; // we print a proper warning in loadVisual()

        std::vector<Vector3> points;
        for(std::size_t j = 0; j < meshData->positionArrayCount(); ++j)
        {
            auto array = meshData->positions(j);
            std::copy(array.begin(), array.end(), std::back_inserter(points));
        }

        m_meshPoints[i] = std::move(points);
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
        else if(!m_meshPoints.empty() && m_meshPoints[0])
        {
            // The format has no scene support, use first mesh
            updateBoundingBox(Matrix4{}, 0);
        }
    }
}

void Mesh::loadPhysics(std::size_t maxPhysicsTriangles)
{
    namespace fs = std::experimental::filesystem;

    if(m_physicsLoaded)
        return;

    if(!m_importer)
        throw std::logic_error("You need to call openFile() first");

    // Simplify meshes if possible
    m_simplifiedMeshes = SimplifiedMeshArray{m_importer->mesh3DCount()};
    m_physXBuffers = CookedPhysXMeshArray{m_importer->mesh3DCount()};
    m_physXMeshes = PhysXMeshArray{m_importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_importer->mesh3DCount(); ++i)
    {
        Containers::Optional<Trade::MeshData3D> meshData = m_importer->mesh3D(i);
        if(!meshData || meshData->primitive() != MeshPrimitive::Triangles)
        {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        Corrade::Containers::Optional<PhysXOutputBuffer> buf;

        // We want to cache the simplified & cooked PhysX mesh.
        fs::path cacheFile(Corrade::Utility::formatString("{}.mesh{}", m_filename, i));
        if(fs::exists(cacheFile))
        {
            std::ifstream stream(cacheFile.string(), std::ios::binary);
            if(!stream)
                throw std::runtime_error("Cannot read cache file: " + cacheFile.string());

            // copies all data into buffer
            std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(stream), {});

            buf = PhysXOutputBuffer{std::move(buffer)};
        }
        else
        {
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
            buf = cookForPhysX(m_ctx->physxCooking(), *m_simplifiedMeshes[i]);

            // In order to make this atomic, we create a temporary file, fill
            // it, and move it to the destination.
            std::string TEMPLATE = cacheFile.string() + ".temp-XXXXXX";
            std::vector<char> filename{TEMPLATE.begin(), TEMPLATE.end()+1};

            int fd = mkstemp(filename.data());
            if(fd < 0)
            {
                throw std::runtime_error(Corrade::Utility::formatString(
                    "Could not create temporary cache file {}: {}",
                    filename.data(), strerror(errno)
                ));
            }

            // Make it world-readable
            fchmod(fd, 0644);

            if(write(fd, buf->data(), buf->size()) != static_cast<ssize_t>(buf->size()))
            {
                throw std::runtime_error(Corrade::Utility::formatString(
                    "Could not write to file {}: {}",
                    filename.data(), strerror(errno)
                ));
            }

            close(fd);

            // Now move it to the right location (this is atomic)
            if(rename(filename.data(), cacheFile.c_str()) != 0)
            {
                if(unlink(filename.data()) != 0)
                {
                    Warning{} << "Could not delete temporary file " << filename.data();
                }
            }
        }

        if(buf)
        {
            physx::PxDefaultMemoryInputData stream(buf->data(), buf->size());
            m_physXMeshes[i] = PhysXHolder<physx::PxConvexMesh>{
                m_ctx->physxPhysics().createConvexMesh(stream)
            };
        }

        m_physXBuffers[i] = std::move(buf);
    }

    m_physicsLoaded = true;
}

void Mesh::loadVisual()
{
    if(m_visualLoaded)
        return;

    if(!m_importer)
        throw std::logic_error("You need to call openFile() first");

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
    m_meshFlags = Containers::Array<MeshFlags>{m_importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_importer->mesh3DCount(); ++i)
    {
        Containers::Optional<Trade::MeshData3D> meshData = m_importer->mesh3D(i);
        if(!meshData || !meshData->hasNormals() || meshData->primitive() != MeshPrimitive::Triangles)
        {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        m_meshes[i] = std::make_shared<GL::Mesh>(
            MeshTools::compile(*meshData)
        );

        if(meshData->hasColors())
            m_meshFlags[i] |= MeshFlag::HasVertexColors;
    }

    m_visualLoaded = true;
}

void Mesh::loadPretransform(const std::string& filename)
{
    std::ifstream stream(filename);

    // If there is no file, we silently ignore that and use identity
    // pretransform.
    if(!stream)
        return;

    Matrix4 pretransform;
    for(int i = 0; i < 4; ++i)
    {
        std::string line;
        if(!std::getline(stream, line))
        {
            Error{} << "Short pretransform file" << filename;
            return;
        }

        std::stringstream ss(line);
        ss.imbue(std::locale::classic());

        for(int j = 0; j < 4; ++j)
        {
            if(!(ss >> pretransform[j][i]))
            {
                Error{} << "Could not read number from pretransform file" << filename;
                return;
            }
        }
    }

    setPretransform(pretransform);
}

namespace
{
    std::shared_ptr<Mesh> loadHelper(
        const std::shared_ptr<Context>& ctx,
        const std::string& filename,
        bool physics,
        std::size_t maxPhysicsTriangles)
    {
        auto mesh = std::make_shared<Mesh>(filename, ctx);

        auto t1 = std::chrono::high_resolution_clock::now();
        mesh->openFile();
        auto t2 = std::chrono::high_resolution_clock::now();

        if(physics)
            mesh->loadPhysics(maxPhysicsTriangles);

        auto t3 = std::chrono::high_resolution_clock::now();

        Debug{}
            << "openFile:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << "physics:" << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count()
        ;

        return mesh;
    }
}

std::vector<std::shared_ptr<Mesh>> Mesh::loadThreaded(
    const std::shared_ptr<Context>& ctx,
    const std::vector<std::string>& filenames,
    bool visual, bool physics,
    std::size_t maxPhysicsTriangles)
{
    using Future = std::future<std::shared_ptr<sl::Mesh>>;

    ctpl::thread_pool pool(std::thread::hardware_concurrency());

    std::vector<Future> results;
    for(const auto& filename : filenames)
    {
        results.push_back(pool.push(std::bind(&loadHelper,
            ctx, filename, physics, maxPhysicsTriangles
        )));
    }

    std::vector<std::shared_ptr<sl::Mesh>> ret;
    for(auto& future : results)
    {
        auto mesh = future.get(); // may throw if we had a load error

        if(visual)
            mesh->loadVisual();

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

    if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && objectData->instance() != -1 && m_meshPoints[objectData->instance()])
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
    m_pretransformRigid.translation() = -m_pretransformRigid.rotationScaling() * m_bbox.center();
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
    m_pretransform = Matrix4::scaling(Vector3(m_scale)) * m_pretransformRigid;
}

void Mesh::setPretransform(const Magnum::Matrix4& m)
{
    // We need to separate the transformation matrix into a homogenous scaling
    // and a rotation+translation.

    Matrix3x3 u{Math::NoInit};
    Vector3 w{Math::NoInit};
    Matrix3x3 v{Math::NoInit};
    std::tie(u, w, v) = Math::Algorithms::svd(m.rotationScaling());

    float minScale = w.min();
    float maxScale = w.max();

    if(maxScale - minScale > 1e-5f)
    {
        Error{} << "Scaling is not uniform:" << w;
        throw std::invalid_argument("Scaling is not uniform");
    }

    m_scale = (maxScale + minScale) / 2.0f;
    m_pretransformRigid = Matrix4::from(u*v.transposed(), (1.0f / m_scale) * m.translation());

    updatePretransform();
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

void Mesh::serialize(Corrade::Utility::ConfigurationGroup& group)
{
    group.setValue("filename", m_filename);
    group.setValue("classIndex", m_classIndex);
    group.setValue("scale", m_scale);
    group.setValue("rigidPretransform", m_pretransformRigid);
}

void Mesh::deserialize(const Corrade::Utility::ConfigurationGroup& group)
{
    m_filename = group.value("filename");

    openFile();

    if(group.hasValue("classIndex"))
        setClassIndex(group.value<unsigned int>("classIndex"));

    if(group.hasValue("scale"))
        m_scale = group.value<float>("scale");

    if(group.hasValue("rigidPretransform"))
        m_pretransformRigid = group.value<Magnum::Matrix4>("rigidPretransform");
}

}
