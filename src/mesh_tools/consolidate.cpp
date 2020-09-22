// Consolidate multiple MeshData instances into one buffer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh_tools/consolidate.h>

#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Utility/Algorithms.h>

#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Math/Color.h>

#include <numeric>
#include <unordered_map>

namespace sl
{

using namespace Magnum;

Containers::Optional<ConsolidatedMesh> consolidateMesh(const Containers::ArrayView<Containers::Optional<Trade::MeshData>>& input)
{
    struct Vertex
    {
        Vector3 position;
        Vector2 textureCoords;
        Color4 color;
        Vector4 tangent;
        UnsignedInt vertexIndex;
        Vector3 normal;
    };

    MeshPrimitive primitive;
    bool valid = false;
    for(auto& mesh : input)
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
    for(const auto& mesh : input)
    {
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
        Containers::Array<Containers::Optional<Trade::MeshData>>{input.size()},
        Containers::Array<UnsignedInt>{input.size()+1},
        Containers::Array<UnsignedInt>{input.size()+1},
        sizeof(Vertex)
    };

    std::unordered_multimap<Trade::MeshAttribute, std::pair<UnsignedInt, bool>> attributeMap;
    attributeMap.reserve(out.data.attributeCount());
    for(UnsignedInt i = 0; i != out.data.attributeCount(); ++i)
        attributeMap.emplace(out.data.attributeName(i), std::make_pair(i, false));

    UnsignedInt vertexOffset = 0;
    UnsignedInt indexOffset = 0;

    for(std::size_t i = 0; i < input.size(); ++i)
    {
        out.indexOffsets[i] = indexOffset;
        out.vertexOffsets[i] = vertexOffset;

        if(!input[i])
            continue;

        const auto& inputMesh = *input[i];

        if(inputMesh.primitive() != primitive)
        {
            Error{} << "consolidateMesh: sub-mesh has primitive" << inputMesh.primitive() << ", but expected" << primitive;
            return {};
        }

        UnsignedInt indexLength;

        // If the mesh is indexed, copy the indices over, expanded to 32bit
        if(inputMesh.isIndexed())
        {
            Containers::ArrayView<UnsignedInt> dst = indices.slice(indexOffset, indexOffset + inputMesh.indexCount());
            inputMesh.indicesInto(dst);
            indexLength = inputMesh.indexCount();

            for(auto& idx : dst)
                idx += vertexOffset;
        }
        // Otherwise, generate a trivial indexing
        else
        {
            std::iota(indices + indexOffset, indices + indexOffset + inputMesh.vertexCount(), UnsignedInt{vertexOffset});
            indexLength = inputMesh.vertexCount();
        }

        Containers::Array<Trade::MeshAttributeData> attributes;

        /* Copy attributes to their destination, skipping ones that don't have
           any equivalent in the destination mesh */
        for(UnsignedInt src = 0; src != inputMesh.attributeCount(); ++src)
        {
            /* Go through destination attributes of the same name and find the
               earliest one that hasn't been copied yet */
            auto range = attributeMap.equal_range(inputMesh.attributeName(src));
            UnsignedInt dst = ~UnsignedInt{};
            auto found = attributeMap.end();
            for(auto it = range.first; it != range.second; ++it)
            {
                if(it->second.second) continue;

                /* The range is unordered so we need to go through everything
                   and pick one with smallest ID */
                if(it->second.first < dst) {
                    dst = it->second.first;
                    found = it;
                }
            }

            /* No corresponding attribute found, continue */
            if(dst == ~UnsignedInt{}) continue;

            CORRADE_ASSERT(out.data.attributeFormat(dst) == inputMesh.attributeFormat(src),
                "consolidateMesh expected" << out.data.attributeFormat(dst) << "for attribute" << dst << "(" << Debug::nospace << out.data.attributeName(dst) << Debug::nospace << ") but got" << inputMesh.attributeFormat(src) << "in mesh attribute" << src,
                {});

            /* Copy the data to a slice of the output, mark the attribute as
               copied */
            Utility::copy(inputMesh.attribute(src), out.data.mutableAttribute(dst)
                .slice(vertexOffset, vertexOffset + inputMesh.vertexCount()));
            found->second.second = true;

            // Create output attribute
            Containers::arrayAppend(attributes, Trade::MeshAttributeData{
                out.data.attributeName(dst),
                out.data.attributeFormat(dst),
                out.data.mutableAttribute(dst)
            });
        }

        auto subIndices = indices.slice(indexOffset, indexOffset + indexLength);
        auto subVertices = vertices.slice(vertexOffset, vertexOffset + inputMesh.vertexCount());
        out.meshes[i] = Trade::MeshData{
            primitive,
            Trade::DataFlag::Mutable, subIndices, Trade::MeshIndexData{subIndices},
            Trade::DataFlag::Mutable, subVertices, std::move(attributes)
        };

        /* Update vertex offset for the next mesh */
        vertexOffset += inputMesh.vertexCount();
        indexOffset += indexLength;
    }

    out.vertexOffsets.back() = vertexOffset;
    out.indexOffsets.back() = indexOffset;

    // Fill vertex indices (one-based)
    auto vertexIndices = out.data.mutableAttribute<UnsignedInt>(Trade::MeshAttribute::ObjectId);
    std::iota(vertexIndices.begin(), vertexIndices.end(), 1);

    return out;
}

}
