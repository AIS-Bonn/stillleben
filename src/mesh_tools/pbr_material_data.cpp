// PBR Material for Magnum::Trade
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh_tools/pbr_material_data.h>

#include <Corrade/Utility/DebugStl.h>

#include <MagnumExternal/TinyGltf/tiny_gltf.h>

using namespace Corrade::Containers;
using namespace Corrade::Utility;
using namespace Magnum;

namespace
{
    constexpr bool DUMP = false;
}

namespace sl
{

PBRMaterialData PBRMaterialData::parse(const Trade::PhongMaterialData& materialData, bool haveTinyGltf)
{
    auto ret = PBRMaterialData{
        AbstractMaterialData::Flags{}, materialData.alphaMode(), materialData.alphaMask(), materialData.importerState()
    };

    ret.m_baseColor = materialData.diffuseColor();

    if(materialData.flags() & Trade::PhongMaterialData::Flag::DiffuseTexture)
    {
        ret.m_baseColorTexture = materialData.diffuseTexture();
        ret.m_flags |= PBRMaterialData::Flag::BaseColorTexture;
    }

    if(haveTinyGltf)
    {
        auto material = reinterpret_cast<const tinygltf::Material*>(materialData.importerState());

        if constexpr(DUMP)
        {
            for(auto& pair : material->values)
                Debug{Debug::Flag::NoSpace} << "Value: '" << pair.first << "'";
            for(auto& pair : material->additionalValues)
                Debug{Debug::Flag::NoSpace} << "Additional value: '" << pair.first << "'";
        }

        auto metallic = material->values.find("metallicFactor");
        if(metallic != material->values.end())
            ret.m_metallic = metallic->second.Factor();

        auto roughness = material->values.find("roughnessFactor");
        if(roughness != material->values.end())
            ret.m_roughness = roughness->second.Factor();

        auto normalMap = material->additionalValues.find("normalTexture");
        if(normalMap != material->additionalValues.end())
        {
            ret.m_flags |= PBRMaterialData::Flag::NormalMap;
            ret.m_normalMapTexture = normalMap->second.TextureIndex();
        }
    }

    return ret;
}

}
