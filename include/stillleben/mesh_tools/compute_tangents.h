// Compute tangents for a mesh with normals
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_MESH_TOOLS_COMPUTE_TANGENTS_H
#define SL_MESH_TOOLS_COMPUTE_TANGENTS_H

#include <Magnum/Trade/Trade.h>

namespace sl
{
namespace mesh_tools
{

Magnum::Trade::MeshData computeTangents(Magnum::Trade::MeshData&& mesh);

}
}

#endif
