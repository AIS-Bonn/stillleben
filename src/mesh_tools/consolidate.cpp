// Consolidate multiple MeshData instances into one buffer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>
// adapted from Magnum::MeshTools::concatenate()

#include <stillleben/mesh_tools/consolidate.h>

#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/Utility/Algorithms.h>

#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/SceneData.h>

#include <numeric>
#include <unordered_map>

namespace sl
{

using namespace Magnum;

namespace
{
    template<class T, class Callable>
    Containers::Optional<Trade::MeshAttributeData> transferAttribute(Trade::MeshData& input, Trade::MeshData& output, UnsignedInt vertexOffset, Trade::MeshAttribute attr, Callable&& transformer = [](Containers::StridedArrayView2D<T>& input, Containers::StridedArrayView2D<T>& output){ Utility::copy(input, output); })
    {
        if(!input.hasAttribute(attr))
            return {};

        auto src = input.attribute<T>(attr);
        auto dst = output.mutableAttribute<T>(attr).slice(vertexOffset, vertexOffset + src.size());

        for(std::size_t i = 0; i < src.size(); ++i)
            dst[i] = transformer(src[i]);

        return Trade::MeshAttributeData{
            attr,
            output.mutableAttribute<T>(attr)
        };
    }
}

Containers::Optional<ConsolidatedMesh> consolidateMesh(Magnum::Trade::AbstractImporter& importer)
{
    struct Vertex
    {
        Vector3 position;
        Vector2 textureCoords;
        Color4 color{{1.0f, 1.0f, 1.0f}};
        Vector4 tangent;
        UnsignedInt vertexIndex;
        Vector3 normal;
    };

    // Load meshes
    Containers::Array<Containers::Optional<Magnum::Trade::MeshData>> meshData{importer.meshCount()};
    for(UnsignedInt i = 0; i < importer.meshCount(); ++i)
    {
        auto mesh = importer.mesh(i);

        if(!mesh)
            continue;

        if(mesh->primitive() != MeshPrimitive::Triangles || !mesh->isIndexed())
        {
            Warning{} << "Ignoring non-triangle (or non-indexed) sub-mesh" << i << "/" << importer.meshCount();
            continue;
        }

        meshData[i] = std::move(mesh);
    }

    MeshPrimitive primitive;
    bool valid = false;
    for(auto& mesh : meshData)
    {
        if(mesh)
        {
            primitive = mesh->primitive();
            valid = true;
            break;
        }
    }

    if(!valid)
        return {};

    if(primitive != Magnum::MeshPrimitive::Triangles)
    {
        Error{} << "consolidateMesh: mesh primitive type" << primitive << "is not supported";
        return {};
    }

    UnsignedInt indexCount = 0;
    UnsignedInt vertexCount = 0;

    // Load objects
    using ObjectDataArray = Containers::Array<Containers::Pointer<Trade::ObjectData3D>>;
    ObjectDataArray objects;
    if(importer.object3DCount() != 0)
    {
        objects = ObjectDataArray{importer.object3DCount()};
        for(UnsignedInt i = 0; i < importer.object3DCount(); ++i)
            objects[i] = importer.object3D(i);
    }
    else
    {
        // Format has no support for objects, create a dummy one.
        objects = ObjectDataArray{1};
        objects[0] = Containers::Pointer<Trade::ObjectData3D>(new Trade::MeshObjectData3D{{}, {}, 0, 0});
    }

    UnsignedInt objectCount = objects.size();

    for(UnsignedInt i = 0; i < objectCount; ++i)
    {
        auto& obj = objects[i];

        if(!obj || obj->instanceType() != Magnum::Trade::ObjectInstanceType3D::Mesh)
            continue;

        auto& mesh = meshData[obj->instance()];

        if(!mesh)
            continue;

        indexCount += mesh->isIndexed() ? mesh->indexCount() : mesh->vertexCount();
        vertexCount += mesh->vertexCount();
    }

    Containers::Array<char> indexData(indexCount * sizeof(UnsignedInt));
    Containers::Array<char> vertexData(vertexCount * sizeof(Vertex));

    auto indices = Containers::arrayCast<UnsignedInt>(indexData);
    auto vertices = Containers::arrayCast<Vertex>(vertexData);

    ConsolidatedMesh out{
        Trade::MeshData{
            primitive,
            std::move(indexData), Trade::MeshIndexData{indices},
            std::move(vertexData), {
                Trade::MeshAttributeData{Trade::MeshAttribute::Position,
                    Containers::StridedArrayView1D<const Vector3>{vertices,
                        &vertices[0].position, vertexCount, sizeof(Vertex)}
                },
                Trade::MeshAttributeData{Trade::MeshAttribute::TextureCoordinates,
                    Containers::StridedArrayView1D<const Vector2>{vertices,
                        &vertices[0].textureCoords, vertexCount, sizeof(Vertex)}
                },
                Trade::MeshAttributeData{Trade::MeshAttribute::Color,
                    Containers::StridedArrayView1D<const Color4>{vertices,
                        &vertices[0].color, vertexCount, sizeof(Vertex)}
                },
                Trade::MeshAttributeData{Trade::MeshAttribute::Tangent,
                    Containers::StridedArrayView1D<const Vector4>{vertices,
                        &vertices[0].tangent, vertexCount, sizeof(Vertex)}
                },
                Trade::MeshAttributeData{Trade::MeshAttribute::ObjectId,
                    Containers::StridedArrayView1D<const UnsignedInt>{vertices,
                        &vertices[0].vertexIndex, vertexCount, sizeof(Vertex)}
                },
                Trade::MeshAttributeData{Trade::MeshAttribute::Normal,
                    Containers::StridedArrayView1D<const Vector3>{vertices,
                        &vertices[0].normal, vertexCount, sizeof(Vertex)}
                },
            }
        },
        Containers::Array<Containers::Optional<Trade::MeshData>>{objectCount},
        Containers::Array<UnsignedInt>{objectCount+1},
        Containers::Array<UnsignedInt>{objectCount+1},
        sizeof(Vertex)
    };

    UnsignedInt vertexOffset = 0;
    UnsignedInt indexOffset = 0;

    // Amazing recursive lambda construction...
    auto recurse = [&](UnsignedInt objectID, Matrix4 parentTransform, auto& recurse_ref){
        out.indexOffsets[objectID] = indexOffset;
        out.vertexOffsets[objectID] = vertexOffset;

        auto& obj = objects[objectID];
        if(!obj)
            return;

        Matrix4 transform = parentTransform * obj->transformation();

        auto addMesh = [&](){
            if(obj->instanceType() != Magnum::Trade::ObjectInstanceType3D::Mesh)
                return;

            auto& inputMesh = meshData[obj->instance()];

            if(!inputMesh)
                return;

            if(inputMesh->primitive() != primitive)
            {
                Error{} << "consolidateMesh: sub-mesh has primitive" << inputMesh->primitive() << ", but expected" << primitive;
                return;
            }

            UnsignedInt indexLength;

            // If the mesh is indexed, copy the indices over, expanded to 32bit
            if(inputMesh->isIndexed())
            {
                Containers::ArrayView<UnsignedInt> dst = indices.slice(indexOffset, indexOffset + inputMesh->indexCount());
                inputMesh->indicesInto(dst);
                indexLength = inputMesh->indexCount();

                for(auto& idx : dst)
                    idx += vertexOffset;
            }
            // Otherwise, generate a trivial indexing
            else
            {
                std::iota(indices + indexOffset, indices + indexOffset + inputMesh->vertexCount(), UnsignedInt{vertexOffset});
                indexLength = inputMesh->vertexCount();
            }

            // Now, copy mesh attributes. Most of these need to be transformed...
            Containers::Array<Trade::MeshAttributeData> attributes;

            // Position
            if(auto attr = transferAttribute<Vector3>(*inputMesh, out.data,
                vertexOffset, Trade::MeshAttribute::Position,
                [&](const Vector3& a){ return transform.transformPoint(a); }))
            {
                Containers::arrayAppend(attributes, std::move(*attr));
            }

            // Texture coordinates
            if(auto attr = transferAttribute<Vector2>(*inputMesh, out.data,
                vertexOffset, Trade::MeshAttribute::TextureCoordinates,
                [&](const Vector2& a){ return a; }))
            {
                Containers::arrayAppend(attributes, std::move(*attr));
            }

            // Vertex color
            if(auto attr = transferAttribute<Color4>(*inputMesh, out.data,
                vertexOffset, Trade::MeshAttribute::Color,
                [&](const Color4& a){ return a; }))
            {
                Containers::arrayAppend(attributes, std::move(*attr));
            }

            // Tangent
            if(auto attr = transferAttribute<Vector4>(*inputMesh, out.data,
                vertexOffset, Trade::MeshAttribute::Tangent,
                [&](const Vector4& tangent){ return Vector4{transform.transformVector(tangent.xyz()), 1.0f}; }))
            {
                Containers::arrayAppend(attributes, std::move(*attr));
            }

            // Vertex index (will be filled later)
            Containers::arrayAppend(attributes, Trade::MeshAttributeData{
                Trade::MeshAttribute::ObjectId,
                Containers::StridedArrayView1D<const UnsignedInt>{vertices,
                        &vertices[0].vertexIndex, vertexCount, sizeof(Vertex)}
            });

            // Normal
            if(auto attr = transferAttribute<Vector3>(*inputMesh, out.data,
                vertexOffset, Trade::MeshAttribute::Normal,
                [&](const Vector3& normal){ return transform.transformVector(normal); }))
            {
                Containers::arrayAppend(attributes, std::move(*attr));
            }

            auto subIndices = indices.slice(indexOffset, indexOffset + indexLength);
            out.meshes[objectID] = Trade::MeshData{
                primitive,
                Trade::DataFlag::Mutable, subIndices, Trade::MeshIndexData{subIndices},
                Trade::DataFlag::Mutable, vertices, std::move(attributes)
            };

            /* Update vertex offset for the next mesh */
            vertexOffset += inputMesh->vertexCount();
            indexOffset += indexLength;
        };

        addMesh();

        for(auto idx : obj->children())
            recurse_ref(idx, transform, recurse_ref);
    };

    if(importer.defaultScene() != -1)
    {
        auto scene = importer.scene(importer.defaultScene());
        for(auto idx : scene->children3D())
        {
            recurse(idx, {}, recurse);
        }
    }
    else
        recurse(0, {}, recurse);

    out.vertexOffsets.back() = vertexOffset;
    out.indexOffsets.back() = indexOffset;

    CORRADE_INTERNAL_ASSERT(vertexOffset == vertexCount);
    CORRADE_INTERNAL_ASSERT(indexOffset == indexCount);

    // Fill vertex indices (one-based)
    auto vertexIndices = out.data.mutableAttribute<UnsignedInt>(Trade::MeshAttribute::ObjectId);
    std::iota(vertexIndices.begin(), vertexIndices.end(), 1);

    return out;
}

}
