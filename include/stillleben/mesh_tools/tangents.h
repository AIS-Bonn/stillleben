// Obtain tangent attribute from GLTF
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_TANGENT_H
#define STILLLEBEN_TANGENT_H

#include <Corrade/Containers/Optional.h>

#include <Magnum/Magnum.h>
#include <Magnum/Trade/Trade.h>

#include <vector>

namespace sl
{

Corrade::Containers::Optional<std::vector<Magnum::Vector3>> extractTangents(
    const Magnum::Trade::AbstractImporter& importer,
    const Magnum::Trade::MeshData3D& mesh
);

}

#endif
