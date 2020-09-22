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
    Magnum::Trade::MeshData data;
    Corrade::Containers::Array<Corrade::Containers::Optional<Magnum::Trade::MeshData>> meshes;
    Corrade::Containers::Array<Magnum::UnsignedInt> indexOffsets;
    Corrade::Containers::Array<Magnum::UnsignedInt> vertexOffsets;
    std::size_t vertexStride;
};

Corrade::Containers::Optional<ConsolidatedMesh> consolidateMesh(const Corrade::Containers::ArrayView<Corrade::Containers::Optional<Magnum::Trade::MeshData>>& input);

}

#endif
