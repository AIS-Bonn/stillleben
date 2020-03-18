// PBR Material for Magnum::Trade
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_PBR_MATERIAL_DATA_H
#define STILLLEBEN_PBR_MATERIAL_DATA_H

#include <Magnum/Trade/AbstractMaterialData.h>
#include <Magnum/Trade/PhongMaterialData.h>

#include <Corrade/Containers/Pointer.h>

namespace sl
{

class PBRMaterialData : public Magnum::Trade::AbstractMaterialData
{
public:
    static constexpr Magnum::UnsignedByte MaterialType = 100;

    enum class Flag : Magnum::UnsignedShort
    {
        DoubleSided = (1 << 0),
        BaseColorTexture = (1 << 1),
        NormalMap = (1 << 2)
    };
    using Flags = Corrade::Containers::EnumSet<Flag>;

    PBRMaterialData(
        AbstractMaterialData::Flags flags, Magnum::Trade::MaterialAlphaMode alphaMode,
        Magnum::Float alphaMask, const void* importerState = nullptr
    ) : AbstractMaterialData(static_cast<Magnum::Trade::MaterialType>(MaterialType), flags, alphaMode, alphaMask, importerState)
    {}

    PBRMaterialData(const PBRMaterialData&) = delete;
    PBRMaterialData(PBRMaterialData&&) = default;

    PBRMaterialData& operator=(const PBRMaterialData&) = delete;
    PBRMaterialData& operator=(PBRMaterialData&&) = default;

    Magnum::Color4 baseColor() const
    { return m_baseColor; }
    Magnum::UnsignedInt baseColorTexture() const
    { return m_baseColorTexture; }

    Magnum::UnsignedInt normalMapTexture() const
    { return m_normalMapTexture; }

    Flags flags() const
    { return m_flags; }

    Magnum::Float metallic() const
    { return m_metallic; }

    Magnum::Float roughness() const
    { return m_roughness; }

    static PBRMaterialData parse(const Magnum::Trade::PhongMaterialData& materialData, bool haveTinyGltf = false);

private:
    Flags m_flags;

    Magnum::Color4 m_baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    Magnum::UnsignedInt m_baseColorTexture = 0;

    Magnum::UnsignedInt m_normalMapTexture = 0;

    Magnum::Float m_metallic = 0.04f;
    Magnum::Float m_roughness = 0.5f;
};

}

#endif
