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
#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Trade/SceneData.h>

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
        Color4 color{{1.0f, 1.0f, 1.0f}};
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

        /* Reset markers saying which attribute has already been copied */
        for(auto it = attributeMap.begin(); it != attributeMap.end(); ++it)
            it->second.second = false;

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
        out.meshes[i] = Trade::MeshData{
            primitive,
            Trade::DataFlag::Mutable, subIndices, Trade::MeshIndexData{subIndices},
            Trade::DataFlag::Mutable, vertices, std::move(attributes)
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
            output.mutableAttribute<T>(Trade::MeshAttribute::Position)
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
    UnsignedInt objectCount = importer.object3DCount();
    for(UnsignedInt i = 0; i < objectCount; ++i)
    {
        auto obj = importer.object3D(i);

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

    std::unordered_multimap<Trade::MeshAttribute, std::pair<UnsignedInt, bool>> attributeMap;
    attributeMap.reserve(out.data.attributeCount());
    for(UnsignedInt i = 0; i != out.data.attributeCount(); ++i)
        attributeMap.emplace(out.data.attributeName(i), std::make_pair(i, false));

    UnsignedInt vertexOffset = 0;
    UnsignedInt indexOffset = 0;

    auto recurse = [&](UnsignedInt objectID, Matrix4 parentTransform){
        out.indexOffsets[objectID] = indexOffset;
        out.vertexOffsets[objectID] = vertexOffset;

        auto obj = importer.object3D(objectID);

        if(!obj || obj->instanceType() != Magnum::Trade::ObjectInstanceType3D::Mesh)
            return;

        auto& inputMesh = meshData[obj->instance()];

        if(!inputMesh)
            return;

        if(inputMesh->primitive() != primitive)
        {
            Error{} << "consolidateMesh: sub-mesh has primitive" << inputMesh->primitive() << ", but expected" << primitive;
            return;
        }

        Matrix4 transform = parentTransform * obj->transformation();

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

    if(importer.defaultScene() != -1)
    {
        auto scene = importer.scene(importer.defaultScene());
        for(auto idx : scene->children3D())
            recurse(idx, {});
    }
    else
        recurse(0, {});

    out.vertexOffsets.back() = vertexOffset;
    out.indexOffsets.back() = indexOffset;

    // Fill vertex indices (one-based)
    auto vertexIndices = out.data.mutableAttribute<UnsignedInt>(Trade::MeshAttribute::ObjectId);
    std::iota(vertexIndices.begin(), vertexIndices.end(), 1);

    return out;
}

}
