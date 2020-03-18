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

/**
 * @brief Extract tangents from TinyGltfImporter
 *
 * If you use Magnum::Trade::TinyGltfImporter, you can use this function to
 * extract a tangent buffer for a given mesh.
 *
 * @warning Calling this when not using TinyGltfImporter will crash.
 **/
Corrade::Containers::Optional<std::vector<Magnum::Vector3>> extractTangents(
    const Magnum::Trade::AbstractImporter& importer,
    const Magnum::Trade::MeshData3D& mesh
);

}

#endif
