// Compute tangents for a mesh with normals
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>
// Inspired from easy_pbr by Radu Alexandru Rosu <rosu@ais.uni-bonn.de>, MIT License

#include <stillleben/mesh_tools/compute_tangents.h>

#include <Corrade/Containers/StridedArrayView.h>

#include <Magnum/MeshTools/Interleave.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Trade/MeshData.h>

#include <stdexcept>

using namespace Magnum;

namespace sl
{
namespace mesh_tools
{

Trade::MeshData computeTangents(Trade::MeshData&& mesh)
{
    if(!mesh.hasAttribute(Trade::MeshAttribute::Normal))
        throw std::invalid_argument{"Called mesh_tools::computeTangents() on a mesh without normals."};

    if(mesh.primitive() != MeshPrimitive::Triangles)
        throw std::invalid_argument{"mesh_tools::computeTangents() supports only triangle meshes"};

    if(!mesh.isIndexed())
        throw std::invalid_argument{"mesh_tools::computeTangents() only supports indexed meshes"};

    std::size_t vertexCount = mesh.vertexCount();

    auto positions = mesh.positions3DAsArray();

    auto faces = mesh.indicesAsArray();
    UnsignedInt faceCount = faces.size() / 3;

    auto normals = mesh.attribute<Vector3>(Trade::MeshAttribute::Normal);

    Containers::Array<UnsignedInt> degree{ValueInit, vertexCount}; // initialized to 0

    Containers::Array<Vector3> tangents{vertexCount};
    Containers::Array<Vector3> bitangents{vertexCount};

    // If we have UV per vertex then we can calculate a tangent that is aligned with the U direction.
    // Code from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/.
    // More explanation in https://learnopengl.com/Advanced-Lighting/Normal-Mapping.
    if(mesh.hasAttribute(Trade::MeshAttribute::TextureCoordinates))
    {
        if(mesh.attributeFormat(Trade::MeshAttribute::TextureCoordinates) != VertexFormat::Vector2)
            throw std::invalid_argument{"mesh_tools::computeTangents(): Need Vector2 texture coordinates"};

        auto texCoords = mesh.attribute<Vector2>(Trade::MeshAttribute::TextureCoordinates);

        // Compute the tangent for each triangle and then average for each vertex
        for(UnsignedInt f = 0; f < faceCount; ++f)
        {
            UnsignedInt i0 = faces[f*3+0];
            UnsignedInt i1 = faces[f*3+1];
            UnsignedInt i2 = faces[f*3+2];

            degree[i0]++;
            degree[i1]++;
            degree[i2]++;

            Vector3 v0 = positions[i0];
            Vector3 v1 = positions[i1];
            Vector3 v2 = positions[i2];

            Vector2 uv0 = texCoords[i0];
            Vector2 uv1 = texCoords[i1];
            Vector2 uv2 = texCoords[i2];

            // Edges of the triangle: position delta
            Vector3 deltaPos1 = v1 - v0;
            Vector3 deltaPos2 = v2 - v0;

            // UV delta
            Vector2 deltaUV1 = uv1 - uv0;
            Vector2 deltaUV2 = uv2 - uv0;

            Float r = 1.0f / Math::cross(deltaUV1, deltaUV2);
            Vector3 tangent = (deltaPos1 * deltaUV2.y() - deltaPos2 * deltaUV1.y()) * r;
            Vector3 bitangent = (deltaPos2 * deltaUV1.x() - deltaPos1 * deltaUV2.x()) * r;

            tangents[i0] += tangent;
            tangents[i1] += tangent;
            tangents[i2] += tangent;

            bitangents[i0] += bitangent;
            bitangents[i1] += bitangent;
            bitangents[i2] += bitangent;
        }

        for(UnsignedInt v = 0; v < vertexCount; ++v)
        {
            tangents[v] = (tangents[v] / degree[v]).normalized();
            bitangents[v] = (bitangents[v] / degree[v]).normalized();
        }
    }
    else
    {
        // We do not have UV coordinates, so we get a random tangent vector.
        Error{} << "WARNING: computeTangents(): outputting empty tangents";
    }

    // Compute compressed representation (tangent + bitangent sign)
    Containers::Array<char> tangentBuffer{vertexCount*sizeof(Vector4)};
    auto tangentsWithBitangentSign = Containers::arrayCast<Vector4>(tangentBuffer);

    for(std::size_t i = 0; i < vertexCount; ++i)
    {
        Vector3 computedBitangent = Math::cross(normals[i], tangents[i]);
        Float sign = Math::sign(Math::dot(computedBitangent, bitangents[i]));

        tangentsWithBitangentSign[i] = Vector4{tangents[i], sign};
    }

    // Add the tangent data to the mesh
    return MeshTools::interleave(std::move(mesh), {
        Trade::MeshAttributeData{Trade::MeshAttribute::Tangent,
            Containers::StridedArrayView1D<const Vector4>{
                tangentsWithBitangentSign,
                &tangentsWithBitangentSign[0],
                vertexCount,
                sizeof(Vector4)
            }
        }
    });
}

}
}
