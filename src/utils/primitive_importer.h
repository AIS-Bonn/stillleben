// Imports primitive meshes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_UTILS_PRIMITIVE_IMPORTER_H
#define SL_UTILS_PRIMITIVE_IMPORTER_H

#include <Corrade/Containers/Pointer.h>

#include <Magnum/Trade/AbstractImporter.h>

namespace sl
{

class PrimitiveImporter : public Magnum::Trade::AbstractImporter
{
public:
    PrimitiveImporter();
    ~PrimitiveImporter();

private:
    Magnum::Trade::ImporterFeatures doFeatures() const override;
    bool doIsOpened() const override;
    void doOpenFile(const std::string& filename) override;

    Magnum::UnsignedInt doMaterialCount() const override;
    Corrade::Containers::Optional<Magnum::Trade::MaterialData> doMaterial(Magnum::UnsignedInt id) override;

    Magnum::UnsignedInt doMeshCount() const override;
    Corrade::Containers::Optional<Magnum::Trade::MeshData> doMesh(Magnum::UnsignedInt id, Magnum::UnsignedInt level) override;

    void doClose() override;

    class Private;
    Corrade::Containers::Pointer<Private> m_d;
};

}

#endif
