// Consolidate multiple MeshData instances into one buffer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_MESH_TOOLS_CONSOLIDATE_H
#define SL_MESH_TOOLS_CONSOLIDATE_H

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Optional.h>

#include <Magnum/Trade/MeshData.h>

namespace sl
{

class ConsolidatedMesh
{
public:
    //! Owns the data buffers
    Magnum::Trade::MeshData data;

    //! Individual sub-meshes
    Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::MeshData>> meshes;

    //! Start of each sub-mesh in the index buffer
    Corrade::Containers::Array<Magnum::UnsignedInt> indexOffsets;

    //! Start of each sub-mesh in the vertex buffer
    Corrade::Containers::Array<Magnum::UnsignedInt> vertexOffsets;

    //! Size of each vertex
    std::size_t vertexStride;
};

/**
 * @brief Consolidate multiple MeshData instances into one buffer
 **/
Corrade::Containers::Optional<ConsolidatedMesh> consolidateMesh(const Corrade::Containers::ArrayView<Corrade::Containers::Optional<Magnum::Trade::MeshData>>& input);

}

#endif
