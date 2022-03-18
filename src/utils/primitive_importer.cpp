// Imports primitive meshes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "primitive_importer.h"

#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/String.h>

#include <Magnum/Trade/MeshData.h>
#include <Magnum/Trade/MaterialData.h>
#include <Magnum/Trade/ObjectData3D.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Reference.h>
#include <Magnum/Primitives/Cube.h>

#include <stdexcept>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

class PrimitiveImporter::Private
{
public:
    bool isOpen = false;
    Containers::Optional<Trade::MeshData> mesh;
};

PrimitiveImporter::PrimitiveImporter()
 : m_d{InPlaceInit}
{
}

PrimitiveImporter::~PrimitiveImporter()
{
}

Trade::ImporterFeatures PrimitiveImporter::doFeatures() const
{
    return {};
}

bool PrimitiveImporter::doIsOpened() const
{
    return m_d->isOpen;
}

void PrimitiveImporter::doOpenFile(const std::string& filename)
{
    if(!Utility::String::beginsWith(filename, "primitive://"))
        throw std::invalid_argument{"File name should begin with primitive://"};

    std::string primitive = Utility::String::stripPrefix(filename, "primitive://");

    if(primitive == "cube")
    {
        m_d->mesh = Primitives::cubeSolid();
        m_d->isOpen = true;
    }
    else
        throw std::invalid_argument{Utility::formatString("Unknown primitive {}", primitive)};
}

UnsignedInt PrimitiveImporter::doMaterialCount() const
{
    return 1;
}

Containers::Optional<Trade::MaterialData> PrimitiveImporter::doMaterial(UnsignedInt id)
{
    if(id == 0)
    {
        return Trade::MaterialData{Trade::MaterialType::PbrMetallicRoughness, {
            {Trade::MaterialAttribute::BaseColor, 0x3bd267ff_srgbaf}
        }};
    }
    else
        return {};
}

UnsignedInt PrimitiveImporter::doMeshCount() const
{
    return 1;
}

Containers::Optional<Trade::MeshData> PrimitiveImporter::doMesh(UnsignedInt id, UnsignedInt level)
{
    if(id != 0 || !m_d->isOpen)
        return {};

    return MeshTools::reference(*m_d->mesh);
}

void PrimitiveImporter::doClose()
{
    m_d->isOpen = false;
}


}
