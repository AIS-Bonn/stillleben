// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh.h>
#include <stillleben/context.h>
#include <stillleben/mesh_tools/tangents.h>
#include <stillleben/mesh_tools/consolidate.h>
#include <stillleben/physx.h>
#include <stillleben/physx_impl.h>

#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/ScopeGuard.h>

#include <Corrade/PluginManager/Manager.h>

#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/Format.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/MurmurHash2.h>
#include <Corrade/Utility/String.h>

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
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Trade/ImageData.h>

#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>
#include <Magnum/Trade/AbstractSceneConverter.h>
#include <Magnum/Image.h>

#include <VHACD.h>

#include <sstream>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

#include "shaders/render_shader.h"
#include "utils/os.h"

using namespace Magnum;

namespace sl
{

namespace
{
    /**
     * @brief Create PhysX collision shape convex hull
     * */
    bool cookForPhysX(physx::PxCooking& cooking,
        const Containers::ArrayView<Vector3>& vertices,
        PhysXOutputBuffer& buffer)
    {
        static_assert(sizeof(decltype(*vertices.data())) == sizeof(physx::PxVec3));

        physx::PxConvexMeshDesc meshDesc;
        meshDesc.points.count = vertices.size();
        meshDesc.points.stride = sizeof(physx::PxVec3);
        meshDesc.points.data = vertices.data();
        meshDesc.flags = physx::PxConvexFlag::eCOMPUTE_CONVEX | physx::PxConvexFlag::eSHIFT_VERTICES;

        physx::PxConvexMeshCookingResult::Enum result;
        bool status = cooking.cookConvexMesh(meshDesc, buffer, &result);
        if(!status)
        {
            Error{} << "PhysX cooking failed: " << result;
            return false;
        }

        return true;
    }

    using MeshHash = Corrade::Utility::MurmurHash2;
    constexpr Magnum::UnsignedInt FILE_FORMAT_VERSION = 5;

    template<class T>
    MeshHash::Digest hashArray(const Corrade::Containers::Array<T>& vec)
    {
        MeshHash hash{}; // use default seed
        const auto& view = Corrade::Containers::arrayCast<const char>(vec);
        return hash(view.data(), view.size());
    }

    Corrade::Containers::Optional<std::vector<uint8_t>> readCacheFile(const std::string& cacheFile, const std::string& sourceFile, const Magnum::Trade::MeshData& meshData, Mesh::Flags flags)
    {
        auto readHash = [](std::istream& stream) -> Corrade::Containers::Optional<MeshHash::Digest>{
            std::array<char, MeshHash::DigestSize> hashBytes;
            stream.read(hashBytes.data(), hashBytes.size());
            if(static_cast<std::size_t>(stream.gcount()) != hashBytes.size())
                return {};

            return MeshHash::Digest::fromByteArray(hashBytes.data());
        };
        auto readVersion = [](std::istream& stream) -> Magnum::UnsignedInt {
            Magnum::UnsignedInt version;
            stream.read(reinterpret_cast<char*>(&version), sizeof(version));
            if(stream.gcount() != sizeof(version))
                return 0;
            return version;
        };
        auto readFlags = [](std::istream& stream) -> Mesh::Flags {
            Magnum::UnsignedInt flags;
            stream.read(reinterpret_cast<char*>(&flags), sizeof(flags));
            if(stream.gcount() != sizeof(flags))
                return {};
            return Mesh::Flags{Mesh::Flag(flags)};
        };

        if(!Corrade::Utility::Directory::exists(cacheFile))
            return {};

        if(os::modificationTime(cacheFile) <= os::modificationTime(sourceFile))
        {
            Debug{} << "Cache file is stale";
            return {};
        }

        std::ifstream stream(cacheFile, std::ios::binary);
        if(!stream)
            return {};

        auto version = readVersion(stream);
        if(version != FILE_FORMAT_VERSION)
            return {};

        if(readFlags(stream) != flags)
            return {};

        auto cacheVertexHash = readHash(stream);
        auto cacheIndicesHash = readHash(stream);

        auto vertexHash = hashArray(meshData.positions3DAsArray(0));
        if(vertexHash != cacheVertexHash)
        {
            Debug{} << "Vertex hash does not match: cached=" << cacheVertexHash << ", actual=" << vertexHash;
            return {};
        }

        auto indicesHash = hashArray(meshData.indicesAsArray());
        if(indicesHash != cacheIndicesHash)
        {
            Debug{} << "Index hash does not match: cached=" << cacheIndicesHash << ", actual=" << indicesHash;
            return {};
        }

        // copies all data into buffer
        // TODO: Is this inefficient?
        std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(stream), {});

        return buffer;
    }

    bool meshIsWatertight(const Containers::ArrayView<Vector3>& vertices, const Containers::ArrayView<UnsignedInt>& indices)
    {
        static_assert(2 * sizeof(UnsignedInt) == sizeof(std::uint64_t));
        std::unordered_map<std::uint64_t, UnsignedInt> counts;

        // Loop over all edges
        for(std::size_t i = 0; i < indices.size()/3; ++i)
        {
            auto triangle = indices.slice(i*3, i*3+3);
            for(std::size_t j = 0; j < 3; ++j)
            {
                auto v0 = triangle[j];
                auto v1 = triangle[(j+1)%3];

                if(v0 < v1)
                    counts[(static_cast<uint64_t>(v0) << 32) | v1]++;
                else
                    counts[(static_cast<uint64_t>(v1) << 32) | v0]++;
            }
        }

        return std::all_of(counts.begin(), counts.end(), [](auto edge){
            return edge.second == 2;
        });
    }
}

Mesh::Mesh(const std::string& filename, const std::shared_ptr<Context>& ctx, Flags flags)
 : m_ctx{ctx}
 , m_filename{filename}
 , m_flags{flags}
{
}

Mesh::Mesh(sl::Mesh && other) = default;

Mesh::~Mesh()
{
}

void Mesh::load(bool visual, bool physics)
{
    // If we don't load the visuals, hide Magnum importer warnings
    std::stringstream warnings;
    Warning redirect(visual ? &std::cerr : &warnings);

    openFile();

    if(physics)
        loadPhysics();

    if(visual)
        loadVisual();
}

void Mesh::openFile()
{
    if(m_opened)
        return;

    Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter> manager(m_ctx->importerPluginPath());
    Pointer<Magnum::Trade::AbstractImporter> importer;

    bool haveTinyGltf = false;

    // Load a scene importer plugin
    if(Utility::String::endsWith(m_filename, ".gltf") || Utility::String::endsWith(m_filename, ".glb"))
    {
        importer = manager.loadAndInstantiate("TinyGltfImporter");
        if(!importer)
            std::abort();

        haveTinyGltf = true;
    }
    else
    {
        importer = manager.loadAndInstantiate("AssimpImporter");
        if(!importer)
            std::abort();

        // Set up postprocess options if using AssimpImporter
        auto group = importer->configuration().group("postprocess");
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
    }

    // Load file
    {
        if(!importer->openFile(m_filename))
        {
            importer.reset();
            throw LoadException("Could not load file: " + m_filename);
        }
    }

    // Load pretransform, if available
    loadPretransform(m_filename + ".pretransform");

    // Load scene data
    if(importer->defaultScene() != -1)
    {
        m_sceneData = importer->scene(importer->defaultScene());
    }

    // Load objects
    if(importer->object3DCount() != 0)
    {
        m_objectData = ObjectDataArray{importer->object3DCount()};
        for(UnsignedInt i = 0; i < importer->object3DCount(); ++i)
            m_objectData[i] = importer->object3D(i);
    }
    else
    {
        // Format has no support for objects, create a dummy one.
        m_objectData = ObjectDataArray{1};
        m_objectData[0] = Containers::Pointer<Trade::ObjectData3D>(new Trade::MeshObjectData3D{{}, {}, 0, 0, 0});
    }

    // Consolidate mesh data into one buffer
    auto consolidated = consolidateMesh(*importer);
    if(!consolidated)
    {
        Error{} << "Could not consolidate mesh";
        return;
    }
    m_consolidated.emplace(std::move(*consolidated));

    // Load textures
    m_textureData = Array<Optional<Magnum::Trade::TextureData>>{importer->textureCount()};
    for(UnsignedInt i = 0; i < importer->textureCount(); ++i)
        m_textureData[i] = importer->texture(i);

    // Load images
    m_imageData = ImageDataArray{importer->image2DCount()};
    for(UnsignedInt i = 0; i < importer->image2DCount(); ++i)
        m_imageData[i] = importer->image2D(i);

    // Load materials.
    m_materials = MaterialArray{importer->materialCount()};
    for(UnsignedInt i = 0; i != importer->materialCount(); ++i)
        m_materials[i] = importer->material(i);

    // Compute bounding box
    updateBoundingBox();

    m_opened = true;
}

void Mesh::loadPhysics()
{
    if(m_physicsLoaded)
        return;

    if(!m_consolidated)
        throw std::runtime_error{"No mesh found"};

    auto& meshData = m_consolidated->data;
    if(meshData.primitive() != MeshPrimitive::Triangles)
    {
        Warning{} << "loadPhysics: unsupported primitive";
        return;
    }

    PhysXOutputBuffer buf;

    // We want to cache the simplified & cooked PhysX mesh.
    std::string cacheFile = Corrade::Utility::formatString("{}.sl_mesh", m_filename);

    auto buffer = readCacheFile(cacheFile, m_filename, meshData, m_flags);

    if(buffer)
    {
        buf = PhysXOutputBuffer{std::move(*buffer)};
    }
    else
    {
        Debug{} << "Simplifying mesh and writing cache file...";
        Array<Vector3> vertices = meshData.positions3DAsArray();
        Array<UnsignedInt> indices = meshData.indicesAsArray();

        if(!meshIsWatertight(vertices, indices))
            Warning{} << "Mesh is not watertight!";

        // First compute a single convex hull (we abuse V-HACD here because
        // it's nice and robust)
        auto singleHull = VHACD::CreateVHACD();
        Containers::ScopeGuard _singleHullDeleter{singleHull, [](VHACD::IVHACD* vhacd){
            vhacd->Clean();
            vhacd->Release();
        }};

        bool giveRawMeshToPhysX = false;
        {
            // Call V-HACD for convex decomposition
            VHACD::IVHACD::Parameters params;
            params.m_concavity = 1.0;
            params.m_asyncACD = false; // hangs sometimes
            params.m_convexhullApproximation = false;
            params.m_maxConvexHulls = 1;

            singleHull->Compute(
                reinterpret_cast<const float*>(vertices.data()),
                vertices.size(),
                indices.data(),
                indices.size()/3,
                params
            );

            double volume = 0;
            for(std::size_t i = 0; i < singleHull->GetNConvexHulls(); ++i)
            {
                VHACD::IVHACD::ConvexHull hull;
                singleHull->GetConvexHull(i, hull);
                volume += hull.m_volume;
            }

            if(volume < 1e-9)
            {
                // We could not compute a sensible convex hull using V-HACD, let
                // PhysX try from the raw vertices.
                giveRawMeshToPhysX = true;
            }
        }

        VHACD::IVHACD* source = singleHull;

        // Now compute a real convex decomposition and see if it's better
        auto vhacd = VHACD::CreateVHACD();
        Containers::ScopeGuard deleter{vhacd, [](VHACD::IVHACD* vhacd){
            vhacd->Clean();
            vhacd->Release();
        }};

        if(!giveRawMeshToPhysX && !(m_flags & Flag::PhysicsForceConvexHull))
        {
            {
                // Call V-HACD for convex decomposition
                VHACD::IVHACD::Parameters params;
                params.m_concavity = 0.002;
                params.m_asyncACD = false; // hangs sometimes

                vhacd->Compute(
                    reinterpret_cast<const float*>(vertices.data()),
                    vertices.size(),
                    indices.data(),
                    indices.size()/3,
                    params
                );
            }

            double decompositionVolume = 0.0;
            for(Magnum::UnsignedInt j = 0; j < vhacd->GetNConvexHulls(); ++j)
            {
                VHACD::IVHACD::ConvexHull hull;
                vhacd->GetConvexHull(j, hull);
                decompositionVolume += hull.m_volume;
            }

            double singleHullVolume = 0.0;
            {
                VHACD::IVHACD::ConvexHull hull;

                CORRADE_INTERNAL_ASSERT(singleHull->GetNConvexHulls() == 1);
                singleHull->GetConvexHull(0, hull);
                singleHullVolume += hull.m_volume;
            }

            // If the convexity is high enough, we can use a single convex hull,
            // which makes physics simulation *much* faster.
            double convexity = decompositionVolume / singleHullVolume;

            if(convexity < 0.75)
                source = vhacd;
        }

        if(!giveRawMeshToPhysX)
        {
            Magnum::UnsignedInt n = source->GetNConvexHulls();
            buf.write(&n, sizeof(n));

            Magnum::UnsignedInt numValid = 0;

            for(Magnum::UnsignedInt j = 0; j < n; ++j)
            {
                VHACD::IVHACD::ConvexHull hull;
                source->GetConvexHull(j, hull);
                auto hullVerticesD = Containers::arrayView(
                    reinterpret_cast<const Vector3d*>(hull.m_points),
                    hull.m_nPoints
                );

                Array<Vector3> hullVertices{hullVerticesD.size()};
                for(std::size_t k = 0; k < hullVertices.size(); ++k)
                    hullVertices[k] = Vector3{hullVerticesD[k]};

                bool ok = cookForPhysX(
                    m_ctx->physxCooking(),
                    hullVertices,
                    buf
                );

                if(!ok)
                    continue;

                numValid++;
            }

            if(numValid != n)
            {
                Warning{} << "Some parts of the convex decomposition of mesh" << m_filename << "failed to load, the physics simulation might not be accurate (got" << numValid << "of" << n << ").";

                // Patch the number of actually written hulls
                std::memcpy(buf.data(), &numValid, sizeof(numValid));
            }
        }
        else
        {
            Magnum::UnsignedInt n = 1;
            buf.write(&n, sizeof(n));

            bool ok = cookForPhysX(
                m_ctx->physxCooking(),
                vertices,
                buf
            );

            if(!ok)
            {
                Warning{} << "Could not compute convex hull at all for mesh" << m_filename << ". This object will not participate in physics simulation.";
                return;
            }
        }

        os::AtomicFileStream ostream(cacheFile);

        // Write version
        ostream.write(reinterpret_cast<const char*>(&FILE_FORMAT_VERSION), sizeof(FILE_FORMAT_VERSION));

        // Write flags
        Magnum::UnsignedInt flagsValue = Containers::enumCastUnderlyingType(m_flags);
        ostream.write(reinterpret_cast<const char*>(&flagsValue), sizeof(flagsValue));

        // Write hashes
        MeshHash::Digest vertexHash = hashArray(meshData.positions3DAsArray());
        vertexHash.byteArray();
        ostream.write(vertexHash.byteArray(), MeshHash::DigestSize);

        MeshHash::Digest indicesHash = hashArray(meshData.indicesAsArray());
        ostream.write(indicesHash.byteArray(), MeshHash::DigestSize);

        // Write buf data
        ostream.write(reinterpret_cast<char*>(buf.data()), buf.size());
    }

    physx::PxDefaultMemoryInputData stream(buf.data(), buf.size());
    Magnum::UnsignedInt n;
    if(stream.read(&n, sizeof(n)) != sizeof(n))
    {
        Error{} << "Invalid buffer, could not read number of convex hulls";
        return;
    }

    m_physXMeshes = Array<PhysXHolder<physx::PxConvexMesh>>{n};
    for(Magnum::UnsignedInt j = 0; j < n; ++j)
    {
        m_physXMeshes[j] = PhysXHolder<physx::PxConvexMesh>{
            m_ctx->physxPhysics().createConvexMesh(stream)
        };
    }

    m_physXBuffer.emplace(std::move(buf));

    m_physicsLoaded = true;
}

void Mesh::loadPhysicsVisualization()
{
    if(m_physicsVisLoaded)
        return;

    loadPhysics();

    m_physXVisMeshData = PhysXVisDataArray{};
    m_physXVisMeshes = Array<std::shared_ptr<GL::Mesh>>{m_physXMeshes.size()};

    for(std::size_t j = 0; j < m_physXMeshes.size(); ++j)
    {
        auto& physXMesh = m_physXMeshes[j];

        auto physXVertices = Corrade::Containers::arrayView(physXMesh->getVertices(), physXMesh->getNbVertices());
        auto vertices = Corrade::Containers::arrayCast<const Magnum::Vector3>(physXVertices);

        const Magnum::UnsignedByte* indexSrcData = physXMesh->getIndexBuffer();

        // Triangulate the polygons
        Corrade::Containers::Array<UnsignedInt> triangles;
        for(std::size_t k = 0; k < physXMesh->getNbPolygons(); ++k)
        {
            physx::PxHullPolygon poly;
            physXMesh->getPolygonData(k, poly);

            UnsignedInt v0 = indexSrcData[poly.mIndexBase];

            for(std::size_t l = 1; l + 1 < poly.mNbVerts; ++l)
            {
                UnsignedInt v1 = indexSrcData[poly.mIndexBase + l];
                UnsignedInt v2 = indexSrcData[poly.mIndexBase + l + 1];

                Containers::arrayAppend(triangles, v0);
                Containers::arrayAppend(triangles, v1);
                Containers::arrayAppend(triangles, v2);
            }
        }

        auto triangleView = Containers::arrayCast<char>(triangles);
        Containers::Array<char> indexData(triangleView.size());
        std::memcpy(indexData.data(), triangleView.data(), indexData.size());
        auto indexView = Containers::arrayCast<UnsignedInt>(indexData);

        auto vertexSrc = Containers::arrayCast<const char>(vertices);
        Containers::Array<char> vertexData(vertexSrc.size());
        std::memcpy(vertexData.data(), vertexSrc.data(), vertexData.size());
        auto vertexView = Containers::arrayCast<Vector3>(vertexData);

        Trade::MeshData meshData{MeshPrimitive::Triangles,
            std::move(indexData), Trade::MeshIndexData{indexView},
            std::move(vertexData), {
            Trade::MeshAttributeData{Trade::MeshAttribute::Position,
                Containers::StridedArrayView1D<const Vector3>{
                    vertexView, &vertexView[0],
                    vertexView.size(), sizeof(Magnum::Vector3)}}
        }};

        m_physXVisMeshes[j] = std::make_shared<GL::Mesh>(MeshTools::compile(meshData));
        Containers::arrayAppend(m_physXVisMeshData, std::move(meshData));
    }
}

Mesh::PhysXVisDataArray& Mesh::physicsMeshData()
{
    loadPhysicsVisualization();
    return m_physXVisMeshData;
}

void Mesh::dumpPhysicsMeshes(const std::string& path)
{
    loadPhysicsVisualization();

    if(!Utility::Directory::isDirectory(path))
        throw std::runtime_error{Utility::formatString("Path {} is not a directory", path)};

    PluginManager::Manager<Trade::AbstractSceneConverter> manager{m_ctx->sceneConverterPluginPath()};

    auto converter = manager.loadAndInstantiate("StanfordSceneConverter");
    if(!converter)
        throw std::runtime_error{"Could not load StanfordSceneConverter"};

    unsigned int idx = 0;
    for(auto& mesh : m_physXVisMeshData)
    {
        converter->convertToFile(mesh, Utility::Directory::join(path, Utility::formatString("{:.4}.ply", idx++)));
    }
}

void Mesh::loadVisual()
{
    if(m_visualLoaded)
        return;

    if(!m_consolidated)
        throw std::runtime_error{"No mesh found"};

    // Load all textures. Textures that fail to load will be NullOpt.
    m_textures = Containers::Array<Containers::Optional<GL::Texture2D>>{m_textureData.size()};
    for(UnsignedInt i = 0; i != m_textureData.size(); ++i)
    {
        const auto& textureData = m_textureData[i];
        if(!textureData || textureData->type() != Trade::TextureData::Type::Texture2D)
        {
            Warning{} << "Cannot load texture properties, skipping";
            continue;
        }

        const auto& imageData = m_imageData[textureData->image()];
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

    // Load all meshes. Meshes that fail to load will be NullOpt.
    std::size_t numMeshes = m_consolidated->meshes.size();
    m_meshes = MeshArray{numMeshes};
    m_meshFlags = MeshFlagArray{numMeshes};

    m_vertexBuffer = Magnum::GL::Buffer{};
    m_indexBuffer = Magnum::GL::Buffer{};

    m_vertexBuffer.setData(m_consolidated->data.vertexData(), GL::BufferUsage::StaticDraw);
    m_indexBuffer.setData(m_consolidated->data.indexData(), GL::BufferUsage::StaticDraw);

    for(UnsignedInt i = 0; i != numMeshes; ++i)
    {
        const auto& meshData = m_consolidated->meshes[i];
        if(!meshData || !meshData->hasAttribute(Trade::MeshAttribute::Normal) || meshData->primitive() != MeshPrimitive::Triangles)
            continue;

        GL::Mesh mesh;
        mesh
            .setPrimitive(MeshPrimitive::Triangles)
            .setCount(meshData->indexCount())
            .setIndexBuffer(m_indexBuffer, m_consolidated->indexOffsets[i] * sizeof(UnsignedInt), MeshIndexType::UnsignedInt)
        ;

        for(std::size_t attr = 0; attr < meshData->attributeCount(); ++attr)
        {
            Containers::Optional<GL::DynamicAttribute> attribute;

            const VertexFormat format = meshData->attributeFormat(attr);
            switch(meshData->attributeName(attr))
            {
                case Trade::MeshAttribute::Position:
                    attribute.emplace(Shaders::Generic3D::Position{}, format);
                    break;
                case Trade::MeshAttribute::TextureCoordinates:
                    attribute.emplace(Shaders::Generic2D::TextureCoordinates{}, format);
                    break;
                case Trade::MeshAttribute::Color:
                    m_meshFlags[i] |= MeshFlag::HasVertexColors;
                    attribute.emplace(Shaders::Generic2D::Color4{}, format);
                    break;
                case Trade::MeshAttribute::Tangent:
                    attribute.emplace(Shaders::Generic3D::Tangent4{}, format);
                    break;
                case Trade::MeshAttribute::Bitangent:
                    attribute.emplace(Shaders::Generic3D::Bitangent{}, format);
                    break;
                case Trade::MeshAttribute::Normal:
                    attribute.emplace(Shaders::Generic3D::Normal{}, format);
                    break;
                case Trade::MeshAttribute::ObjectId:
                    attribute.emplace(Shaders::Generic3D::ObjectId{}, format);
                    break;

                default:
                    Warning{} << "Ignoring" << meshData->attributeName(attr);
            }

            if(!attribute)
                continue;

            mesh.addVertexBuffer(
                m_vertexBuffer,
                meshData->attributeOffset(attr),
                meshData->attributeStride(attr),
                *attribute
            );
        }

        m_meshes[i] = std::make_shared<GL::Mesh>(std::move(mesh));
    }

    // Nobody needs these anymore
    m_textureData = {};
    m_imageData = {};

    m_visualLoaded = true;
}

void Mesh::updateVertexPositions(
    const Corrade::Containers::ArrayView<int>& verticesIndex,
    const Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate
)
{
    updateVertexPositionsAndColors(verticesIndex, positionsUpdate, {});
}

void Mesh::updateVertexColors(
    const Corrade::Containers::ArrayView<int>& verticesIndex,
    const Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
)
{
    updateVertexPositionsAndColors(verticesIndex, {}, colorsUpdate);
}

void Mesh::recomputeNormals()
{
    const auto& vertices = meshPoints();
    auto numVertices = vertices.size();

    const auto& indices = meshFaces();
    auto numIndices = indices.size();

    std::vector<std::vector<int>> vertexFacesMap(numVertices); // Faces a vertex is associated with.
    std::vector<float> facesArea(numIndices/3);
    std::vector<Vector3> facesNormal(numIndices/3);

    for(std::size_t i = 0; i < numIndices; i+=3)
    {
        auto face = i / 3;
        auto& vertexOne = indices[i ];
        auto& vertexTwo = indices[i+1];
        auto& vertexThree = indices[i+2];
        vertexFacesMap[vertexOne].push_back(face);
        vertexFacesMap[vertexTwo].push_back(face);
        vertexFacesMap[vertexThree].push_back(face);

        // area of the face
        Vector3 v1 = vertices[vertexOne];
        Vector3 v2 = vertices[vertexTwo];
        Vector3 v3 = vertices[vertexThree];

        auto v1v2 = v1 - v2;
        auto v1v3 = v1 - v3;

        Vector3 crossProduct = Math::cross(v1v2, v1v3);

        float area = crossProduct.length();
        facesArea[face] = area;

        Vector3 normal = crossProduct.normalized();

        facesNormal[face] = normal;
    }

    // compute new normals weight
    auto normals = meshNormals();
    for(std::size_t i = 0; i < numVertices; ++i)
    {
        Vector3 normal = Vector3(0., 0., 0.);

        for(const auto& face : vertexFacesMap[i])
            normal += facesNormal[face] * facesArea[face];

        normal = normal.normalized();

        normals[i] = normal;
    }
}

void Mesh::recompileMesh()
{
    m_vertexBuffer.setData(m_consolidated->data.vertexData(), GL::BufferUsage::DynamicDraw);
}

void Mesh::updateVertexPositionsAndColors(
    const Corrade::Containers::ArrayView<int>& verticesIndex,
    const Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate,
    const Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
)
{
    if(!positionsUpdate.empty())
    {
        auto vertices = meshPoints();
        for(std::size_t vi = 0; vi < verticesIndex.size(); ++vi)
        {
            Vector3& point = vertices[verticesIndex[vi] - 1];
            Vector3 update = positionsUpdate[vi];
            point = point + update;
        }

        // recompute normals
        recomputeNormals();
    }

    if(!colorsUpdate.empty())
    {
        auto colors = meshColors();
        for(std::size_t vi = 0; vi < verticesIndex.size(); ++vi)
        {
            Color4& color = colors[verticesIndex[vi] - 1];
            Color4 cUpdate = colorsUpdate[vi];
            color = color + cUpdate;
        }
    }

    recompileMesh();
}

void Mesh::setVertexPositions(
    const Corrade::Containers::ArrayView<Magnum::Vector3>& newVertices
)
{
    auto vertices = meshPoints();

    if(vertices.size() != newVertices.size())
        throw std::invalid_argument{"Number of new vertices should match the existing mesh vertices"};

    for(std::size_t i = 0; i < vertices.size(); ++i)
        vertices[i] = newVertices[i];

    recomputeNormals();
    recompileMesh();
}

void Mesh::setVertexColors(
    const Corrade::Containers::ArrayView<Magnum::Color4>& newColors
)
{
    auto colors = meshColors();

    if(colors.size() != newColors.size())
        throw std::invalid_argument{"Number of new vertices should match the existing mesh vertices for vertex color update"};

    for(std::size_t i = 0; i < colors.size(); ++i)
        colors[i] = newColors[i];

    recompileMesh();
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

std::vector<std::shared_ptr<Mesh>> Mesh::loadThreaded(
    const std::shared_ptr<Context>& ctx,
    const std::vector<std::string>& filenames,
    bool visual, bool physics,
    const std::vector<Flags>& flags)
{
    std::queue<int> workQueue;
    std::mutex workQueueMutex;

    {
        for(std::size_t i = 0; i < filenames.size(); ++i)
            workQueue.push(i);
    }

    std::vector<std::shared_ptr<Mesh>> meshes{filenames.size()};

    auto worker = [&](){
        // If we don't load the visuals, hide Magnum importer warnings
        std::stringstream warnings;
        Warning redirect(visual ? &std::cerr : &warnings);

        while(1)
        {
            int index;

            {
                std::unique_lock<std::mutex> lock(workQueueMutex);
                if(workQueue.empty())
                    return;

                index = workQueue.front();
                workQueue.pop();
            }

            std::string filename = filenames[index];

            Mesh::Flags meshFlags{};
            if(!flags.empty())
                meshFlags = flags[index];

            try
            {
                auto mesh = std::make_shared<Mesh>(filename, ctx, meshFlags);

                mesh->openFile();

                if(physics)
                    mesh->loadPhysics();

                meshes[index] = std::move(mesh);
            }
            catch(LoadException& e)
            {
                Error{} << "Could not load file" << filename << ":" << e.what();
            }
        }
    };

    Containers::Array<std::thread> workers{DirectInit,
        std::thread::hardware_concurrency(),
        worker
    };

    for(auto& worker : workers)
        worker.join();

    for(auto& mesh : meshes)
    {
        if(!mesh)
            throw std::runtime_error{"Could not load one of the meshes"};

        if(visual)
            mesh->loadVisual();
    }

    return meshes;
}

void Mesh::updateBoundingBox()
{
    m_bbox = {
        Magnum::Vector3(std::numeric_limits<float>::infinity()),
        Magnum::Vector3(-std::numeric_limits<float>::infinity())
    };

    for(auto& point : m_consolidated->data.attribute<Vector3>(Trade::MeshAttribute::Position))
    {
        m_bbox.min().x() = std::min(m_bbox.min().x(), point.x());
        m_bbox.min().y() = std::min(m_bbox.min().y(), point.y());
        m_bbox.min().z() = std::min(m_bbox.min().z(), point.z());

        m_bbox.max().x() = std::max(m_bbox.max().x(), point.x());
        m_bbox.max().y() = std::max(m_bbox.max().y(), point.y());
        m_bbox.max().z() = std::max(m_bbox.max().z(), point.z());
    }
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

    Matrix3x3 u{NoInit};
    Vector3 w{NoInit};
    Matrix3x3 v{NoInit};
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

    updatePretransform();
}

Mesh::StridedArrayView1D<Vector3> Mesh::meshPoints(int subMesh)
{
    if(!m_consolidated)
        throw Exception{"Mesh was not loaded correctly"};

    auto buf = m_consolidated->data.mutableAttribute<Vector3>(Magnum::Trade::MeshAttribute::Position);

    if(subMesh == -1)
    {
        // Just return the entire buffer
        return buf;
    }
    else
    {
        if(subMesh < 0 || static_cast<UnsignedInt>(subMesh) >= m_consolidated->meshes.size())
            throw Exception{"sl::Mesh::meshPoints() invalid index"};

        return buf.slice(m_consolidated->vertexOffsets[subMesh], m_consolidated->vertexOffsets[subMesh+1]);
    }
}

Mesh::StridedArrayView1D<Magnum::Vector3> Mesh::meshNormals(int subMesh)
{
    if(!m_consolidated)
        throw Exception{"Mesh was not loaded correctly"};

    auto buf = m_consolidated->data.mutableAttribute<Vector3>(Magnum::Trade::MeshAttribute::Normal);

    if(subMesh == -1)
    {
        // Just return the entire buffer
        return buf;
    }
    else
    {
        if(subMesh < 0 || static_cast<UnsignedInt>(subMesh) >= m_consolidated->meshes.size())
            throw Exception{"sl::Mesh::meshPoints() invalid index"};

        return buf.slice(m_consolidated->vertexOffsets[subMesh], m_consolidated->vertexOffsets[subMesh+1]);
    }
}

Mesh::StridedArrayView1D<Magnum::UnsignedInt> Mesh::meshFaces(int subMesh)
{
    if(!m_consolidated)
        throw Exception{"Mesh was not loaded correctly"};

    auto indices = m_consolidated->data.mutableIndices<UnsignedInt>();

    if(subMesh == -1)
        return indices;
    else
    {
        if(subMesh < 0 || static_cast<UnsignedInt>(subMesh) >= m_consolidated->meshes.size())
            throw Exception{"sl::Mesh::meshPoints() invalid index"};

        return indices.slice(m_consolidated->indexOffsets[subMesh], m_consolidated->indexOffsets[subMesh+1]);
    }
}

Mesh::StridedArrayView1D<Magnum::Color4> Mesh::meshColors(int subMesh)
{
    if(!m_consolidated)
        throw Exception{"Mesh was not loaded correctly"};

    auto buf = m_consolidated->data.mutableAttribute<Color4>(Magnum::Trade::MeshAttribute::Color);

    if(subMesh == -1)
    {
        // Just return the entire buffer
        return buf;
    }
    else
    {
        if(subMesh < 0 || static_cast<UnsignedInt>(subMesh) >= m_consolidated->meshes.size())
            throw Exception{"sl::Mesh::meshPoints() invalid index"};

        return buf.slice(m_consolidated->vertexOffsets[subMesh], m_consolidated->vertexOffsets[subMesh+1]);
    }
}

}
