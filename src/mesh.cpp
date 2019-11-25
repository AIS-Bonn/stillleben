// Mesh
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh.h>
#include <stillleben/context.h>
#include <stillleben/contrib/ctpl_stl.h>
#include <stillleben/mesh_tools/simplify_mesh.h>
#include <stillleben/physx.h>

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
#include <Magnum/Trade/ImageData.h>

#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>
#include <Magnum/Image.h>

#include <sstream>
#include <fstream>

#include "shaders/render_shader.h"
#include "physx_impl.h"
#include "utils/os.h"

using namespace Magnum;

namespace sl
{

namespace
{
    /**
     * @brief Create PhysX collision shape from Trade::MeshData3D
     * */
    Corrade::Containers::Optional<PhysXOutputBuffer> cookForPhysX(physx::PxCooking& cooking, const Trade::MeshData3D& meshData)
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

    using MeshHash = Corrade::Utility::MurmurHash2;

    template<class T>
    MeshHash::Digest hashVector(const std::vector<T>& vec)
    {
        MeshHash hash{}; // use default seed
        const auto& view = Corrade::Containers::arrayCast<const char>(
            Corrade::Containers::arrayView(vec)
        );
        return hash(view.data(), view.size());
    }

    Corrade::Containers::Optional<std::vector<uint8_t>> readCacheFile(const std::string& cacheFile, const std::string& sourceFile, const Magnum::Trade::MeshData3D& meshData)
    {
        auto readHash = [](std::istream& stream) -> Corrade::Containers::Optional<MeshHash::Digest>{
            std::array<char, MeshHash::DigestSize> hashBytes;
            stream.read(hashBytes.data(), hashBytes.size());
            if(stream.gcount() != hashBytes.size())
                return {};

            return MeshHash::Digest::fromByteArray(hashBytes.data());
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

        auto cacheVertexHash = readHash(stream);
        auto cacheIndicesHash = readHash(stream);

        auto vertexHash = hashVector(meshData.positions(0));
        if(vertexHash != cacheVertexHash)
        {
            Debug{} << "Vertex hash does not match: cached=" << cacheVertexHash << ", actual=" << vertexHash;
            return {};
        }

        auto indicesHash = hashVector(meshData.indices());
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
    if(m_opened)
        return;

    Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter> manager(m_ctx->importerPluginPath());
    Pointer<Magnum::Trade::AbstractImporter> importer;

    // Load a scene importer plugin
    if(Utility::String::endsWith(m_filename, ".gltf") || Utility::String::endsWith(m_filename, ".glb"))
    {
        importer = manager.loadAndInstantiate("TinyGltfImporter");
        if(!importer)
            std::abort();
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
    m_objectData = ObjectDataArray{importer->object3DCount()};
    for(UnsignedInt i = 0; i < importer->object3DCount(); ++i)
        m_objectData[i] = importer->object3D(i);

    // Load meshes
    m_meshData = Array<Optional<Magnum::Trade::MeshData3D>>{importer->mesh3DCount()};
    for(UnsignedInt i = 0; i < importer->mesh3DCount(); ++i)
        m_meshData[i] = importer->mesh3D(i);

    // Load textures
    m_textureData = Array<Optional<Magnum::Trade::TextureData>>{importer->textureCount()};
    for(UnsignedInt i = 0; i < importer->textureCount(); ++i)
        m_textureData[i] = importer->texture(i);

    // Load images
    m_imageData = ImageDataArray{importer->image2DCount()};
    for(UnsignedInt i = 0; i < importer->image2DCount(); ++i)
        m_imageData[i] = importer->image2D(i);

    // Load materials.
    m_materials = Containers::Array<Containers::Optional<Trade::PhongMaterialData>>{importer->materialCount()};
    for(UnsignedInt i = 0; i != importer->materialCount(); ++i)
    {
        std::unique_ptr<Trade::AbstractMaterialData> materialData = importer->material(i);
        if(!materialData || materialData->type() != Trade::MaterialType::Phong)
        {
            Warning{} << "Cannot load material, skipping";
            continue;
        }

        m_materials[i] = std::move(static_cast<Trade::PhongMaterialData&>(*materialData));
    }

    // Compute bounding box
    m_meshPoints = PointArray{importer->mesh3DCount()};
    m_meshNormals = NormalArray{importer->mesh3DCount()};
    m_meshFaces = FaceArray{importer->mesh3DCount()};
    m_meshColors = ColorArray{importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != m_meshData.size(); ++i)
    {
        auto& meshData = m_meshData[i];
        if(!meshData || !meshData->hasNormals() || meshData->primitive() != MeshPrimitive::Triangles)
            continue; // we print a proper warning in loadVisual()

        std::vector<Vector3> points;
        std::vector<Vector3> normals;
        std::vector<UnsignedInt> faces;
        std::vector<Color4> colors;

        auto colorArraySize = meshData->colorArrayCount();
        for(std::size_t j = 0; j < meshData->positionArrayCount(); ++j)
        {
            auto array = meshData->positions(j);
            points.reserve(points.size() + array.size());
            std::copy(array.begin(), array.end(), std::back_inserter(points));

            auto normalsArray = meshData->normals(j);
            normals.reserve(normals.size() + normalsArray.size());
            std::copy(normalsArray.begin(), normalsArray.end(), std::back_inserter(normals));

            if(colorArraySize != 0)
            {
                // mesh has per-vertex coloring
                auto colorArray = meshData->colors(j);
                colors.reserve(colors.size() + colorArray.size());
                std::copy(colorArray.begin(), colorArray.end(), std::back_inserter(colors));
            }
            else
            {
                // mesh has no per-vertex coloring
                // fill all vertices to white color
                colors.resize(colors.size() + array.size(), Color4{1.0f, 1.0f, 1.0f, 1.0f});
            }
        }

        auto facesArray = meshData->indices();
        faces = std::vector<UnsignedInt>{facesArray.begin(), facesArray.end()};

        m_meshPoints[i] = std::move(points);
        m_meshNormals[i] = std::move(normals);
        m_meshFaces[i] = std::move(faces);
        m_meshColors[i] = std::move(colors);
    }

    // Update the bounding box
    {
        // Inspect the scene if available
        if(m_sceneData)
        {
            // Recursively inspect all children
            for(UnsignedInt objectId : m_sceneData->children3D())
                updateBoundingBox(Matrix4{}, objectId);
        }
        else if(!m_meshPoints.empty() && m_meshPoints[0])
        {
            // The format has no scene support, use first mesh
            updateBoundingBox(Matrix4{}, 0);
        }
    }

    m_opened = true;
}

void Mesh::loadPhysics(std::size_t maxPhysicsTriangles)
{
    if(m_physicsLoaded)
        return;

    if(m_meshData.empty())
        throw std::runtime_error{"No mesh found"};

    // Simplify meshes if possible
    m_simplifiedMeshes = SimplifiedMeshArray{m_meshData.size()};
    m_physXBuffers = CookedPhysXMeshArray{m_meshData.size()};
    m_physXMeshes = PhysXMeshArray{m_meshData.size()};
    for(UnsignedInt i = 0; i != m_meshData.size(); ++i)
    {
        auto& meshData = m_meshData[i];
        if(!meshData || meshData->primitive() != MeshPrimitive::Triangles)
        {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        Corrade::Containers::Optional<PhysXOutputBuffer> buf;

        // We want to cache the simplified & cooked PhysX mesh.
        std::string cacheFile = Corrade::Utility::formatString("{}.mesh{}", m_filename, i);

        auto buffer = readCacheFile(cacheFile, m_filename, *meshData);

        if(buffer)
        {
            Debug{} << "Using mesh cache file";
            buf = PhysXOutputBuffer{std::move(*buffer)};
        }
        else
        {
            Debug{} << "Simplifying mesh and writing cache file...";
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

            os::AtomicFileStream ostream(cacheFile);

            // Write hashes
            MeshHash::Digest vertexHash = hashVector(meshData->positions(0));
            vertexHash.byteArray();
            ostream.write(vertexHash.byteArray(), MeshHash::DigestSize);

            MeshHash::Digest indicesHash = hashVector(meshData->indices());
            ostream.write(indicesHash.byteArray(), MeshHash::DigestSize);

            ostream.write(reinterpret_cast<char*>(buf->data()), buf->size());
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

    if(m_meshData.empty())
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
    m_meshes = MeshArray{m_meshData.size()};
    m_meshFlags = Containers::Array<MeshFlags>{m_meshData.size()};
    for(UnsignedInt i = 0; i != m_meshData.size(); ++i)
    {
        const auto& meshData = m_meshData[i];
        if(!meshData || !meshData->hasNormals() || meshData->primitive() != MeshPrimitive::Triangles)
        {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        m_meshes[i] = std::make_shared<GL::Mesh>(
            MeshTools::compile(*meshData)
        );

        if(m_meshPoints[i])
        {
            unsigned int numPoints = m_meshPoints[i]->size();

            // Vertex index starts from 1
            // this is done to avoid the confusion of 0 being the default value in the texel fetch sampling.
            // FIXME: This produces garbage in the case of multi-submesh meshes.

            m_vertexIndices =  Corrade::Containers::Array<Magnum::UnsignedInt>(Corrade::Containers::NoInit, numPoints);
            for(std::size_t j = 0; j < numPoints; ++j)
                m_vertexIndices[j] = j + 1;

            m_vertexIndexBuf = Magnum::GL::Buffer{};
            m_vertexIndexBuf.setData(m_vertexIndices);
            m_meshes[i]->addVertexBuffer(m_vertexIndexBuf, 0, RenderShader::VertexIndex());
        }

        if(meshData->hasColors())
            m_meshFlags[i] |= MeshFlag::HasVertexColors;
    }

    // Nobody needs these anymore
    m_meshData = {};
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
    if(m_meshData.size() != 1)
        throw Exception{"Mesh::recomputeNormals() assumes a single sub-mesh"};

    auto& meshData = m_meshData.front();

    std::vector<std::vector<int>> vertexFacesMap(meshData->positions(0).size()); // Faces a vertex is associated with.
    std::vector<float> facesArea(meshData->indices().size());
    std::vector<Vector3> facesNormal(meshData->indices().size());

    for(std::size_t i = 0; i < meshData->indices().size(); i+=3)
    {
        auto face = i / 3;
        auto& vertexOne = meshData->indices()[i ];
        auto& vertexTwo = meshData->indices()[i+1];
        auto& vertexThree = meshData->indices()[i+2];
        vertexFacesMap[vertexOne].push_back(face);
        vertexFacesMap[vertexTwo ].push_back(face);
        vertexFacesMap[vertexThree].push_back(face);

        // area of the face
        Vector3 v1 = meshData->positions(0)[vertexOne];
        Vector3 v2 = meshData->positions(0)[vertexTwo];
        Vector3 v3 = meshData->positions(0)[vertexThree];

        auto v1v2 = v1 - v2;
        auto v1v3 = v1 - v3;

        Vector3 crossProduct = Math::cross(v1v2, v1v3);

        float area = crossProduct.length();
        facesArea[face] = area;

        Vector3 normal = crossProduct.normalized();

        facesNormal[face] = normal;
    }

    // compute new normals weight
    for(std::size_t i = 0; i < m_meshPoints[0]->size(); ++i)
    {
        Vector3 normal = Vector3(0., 0., 0.);

        for(const auto& face : vertexFacesMap[i])
            normal += facesNormal[face] * facesArea[face];

        normal = normal.normalized();

        Vector3& oldNormal = meshData->normals(0)[i];
        oldNormal = normal;
    }
}

void Mesh::recompileMesh()
{
    if(m_meshData.size() != 1)
        throw Exception{"Mesh::recompileMesh() assumes a single sub-mesh"};

    *m_meshes[0] = MeshTools::compile(*m_meshData.front());

    // add an index to each vertex in the mesh
    {
        // use the already created m_vertexIndices
        m_vertexIndexBuf = Magnum::GL::Buffer{};
        m_vertexIndexBuf.setData(m_vertexIndices);
        m_meshes[0]->addVertexBuffer(m_vertexIndexBuf, 0, RenderShader::VertexIndex());
    }
}

void Mesh::updateVertexPositionsAndColors(
    const Corrade::Containers::ArrayView<int>& verticesIndex,
    const Corrade::Containers::ArrayView<Magnum::Vector3>& positionsUpdate,
    const Corrade::Containers::ArrayView<Magnum::Color4>& colorsUpdate
)
{
    if(m_meshPoints.size() != 1 || m_meshData.size() != 1)
        throw Exception{"Mesh::updateVertexPositionsAndColors() assumes a single sub-mesh"};

    auto& meshData = m_meshData.front();

    // update
    if(!positionsUpdate.empty())
    {
        for(std::size_t vi = 0; vi < verticesIndex.size(); ++vi)
        {
            Vector3& point = meshData->positions(0)[verticesIndex[vi] - 1];
            Vector3 update = positionsUpdate[vi];
            point = point + update;
        }

        // recompute normals
        recomputeNormals();
    }

    if(!colorsUpdate.empty())
    {
        for(std::size_t vi = 0; vi < verticesIndex.size(); ++vi)
        {
            Color4& color = meshData->colors(0)[verticesIndex[vi] - 1];
            Color4 cUpdate = colorsUpdate[vi];
            color = color + cUpdate;
        }
    }

    m_meshColors[0] = meshData->colors(0);
    m_meshPoints[0] = meshData->positions(0);
    m_meshNormals[0] = meshData->normals(0);

    recompileMesh();
}

void Mesh::setVertexPositions(
    const Corrade::Containers::ArrayView<Magnum::Vector3>& newVertices
)
{
    if(m_meshPoints.size() != 1 || m_meshData.size() != 1)
        throw Exception{"Mesh::setVertexPositions() assumes a single sub-mesh"};

    auto& meshData = m_meshData.front();

    if(meshData->positions(0).size() != newVertices.size())
        throw std::invalid_argument{"Number of new vertices should match the existing mesh vertices"};

    std::copy(newVertices.begin(), newVertices.end(), meshData->positions(0).begin());

    recomputeNormals();

    m_meshPoints[0] = meshData->positions(0);
    m_meshNormals[0] = meshData->normals(0);

    recompileMesh();
}

void Mesh::setVertexColors(
    const Corrade::Containers::ArrayView<Magnum::Color4>& newColors
)
{
    if(m_meshPoints.size() != 1 || m_meshData.size() != 1)
        throw Exception{"Mesh::setVertexColors() assumes a single sub-mesh"};

    auto& meshData = m_meshData.front();

    if(meshData->colorArrayCount() == 0)
    {
        throw Exception("MeshData does not contain colors attribute."
            "This could happend if the mesh file does not contain per vertex coloring.");
    }

    if(meshData->positions(0).size() != newColors.size())
        throw std::invalid_argument{"Number of new vertices should match the existing mesh vertices for vertex color update"};

    std::copy(newColors.begin(), newColors.end(), meshData->colors(0).begin());

    m_meshColors[0] = meshData->colors(0);

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
    const auto& objectData = m_objectData[meshObjectIdx];
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
