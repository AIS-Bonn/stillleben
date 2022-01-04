// Shader which outputs all needed information
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>
// with parts taken from the Magnum engine

#include "render_shader.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/Resource.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Extensions.h>

#include <Magnum/Trade/MaterialData.h>
#include <Magnum/Trade/PbrMetallicRoughnessMaterialData.h>

#include <stillleben/light_map.h>

using namespace Magnum;

namespace sl
{

namespace
{
    enum class TextureInput : Int
    {
        BaseColor = 0,
        Normal,
        MetallicRoughness,
        Emissive,
        Occlusion,
        LightMapIrradiance,
        LightMapPrefilter,
        LightMapBRDFLUT,
        Sticker,
        MinDepth
    };

    enum class Uniform : Int
    {
        MeshToObject,
        ObjectToWorld,
        Projection,
        WorldToCam,
        NormalToWorld,
        NormalToCam,
        Material, // size 3
        AvailableTextures = Material + 3,
        ClassIndex,
        InstanceIndex,
        StickerProjection,
        StickerRange,
        CamPosition
    };

    template<class T>
    constexpr Int eVal(T val)
    { return static_cast<Int>(val); }
}

RenderShader::RenderShader()
 : _flags{}
{
    Utility::Resource rs("stillleben-data");

    const auto version = GL::Version::GL450;

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
        throw std::runtime_error{"The shading system needs the EXPLICIT_ATTRIB_LOCATION GL extension"};

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
        throw std::runtime_error{"The shading system needs the EXPLICIT_UNIFORM_LOCATION GL extension"};

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::shading_language_420pack>(version))
        throw std::runtime_error{"Need SHADING_LANGUAGE_420PACK"};

    GL::Shader vert{version, GL::Shader::Type::Vertex}, geom{version, GL::Shader::Type::Geometry},
        frag{version, GL::Shader::Type::Fragment};

    std::string header = Corrade::Utility::formatString(R"EOS(
// Mesh attributes
#define POSITION_ATTRIBUTE_LOCATION {}
#define TEXTURECOORDINATES_ATTRIBUTE_LOCATION {}
#define COLOR_ATTRIBUTE_LOCATION {}
#define TANGENT_ATTRIBUTE_LOCATION {}
#define VERTEX_INDEX_ATTRIBUTE_LOCATION {}
#define NORMAL_ATTRIBUTE_LOCATION {}
)EOS",
        Shaders::GenericGL3D::Position::Location,
        Shaders::GenericGL3D::TextureCoordinates::Location,
        Shaders::GenericGL3D::Color4::Location,
        Shaders::GenericGL3D::Tangent4::Location,
        Shaders::GenericGL3D::ObjectId::Location,
        Shaders::GenericGL3D::Normal::Location
    );

    header += Corrade::Utility::formatString(R"EOS(
// Texture samplers
#define BASE_COLOR_TEXTURE {}
#define NORMAL_TEXTURE {}
#define METALLIC_ROUGHNESS_TEXTURE {}
#define EMISSIVE_TEXTURE {}
#define OCCLUSION_TEXTURE {}
#define LIGHTMAP_IRRADIANCE_TEXTURE {}
#define LIGHTMAP_PREFILTER_TEXTURE {}
#define LIGHTMAP_BRDF_LUT_TEXTURE {}
#define STICKER_TEXTURE {}
#define DEPTH_TEXTURE {}
)EOS",
        eVal(TextureInput::BaseColor),
        eVal(TextureInput::Normal),
        eVal(TextureInput::MetallicRoughness),
        eVal(TextureInput::Emissive),
        eVal(TextureInput::Occlusion),
        eVal(TextureInput::LightMapIrradiance),
        eVal(TextureInput::LightMapPrefilter),
        eVal(TextureInput::LightMapBRDFLUT),
        eVal(TextureInput::Sticker),
        eVal(TextureInput::MinDepth)
    );

    header += Corrade::Utility::formatString(R"EOS(
// Uniforms
#define UNIFORM_MESH_TO_OBJECT {}
#define UNIFORM_OBJECT_TO_WORLD {}
#define UNIFORM_PROJECTION {}
#define UNIFORM_WORLD_TO_CAM {}
#define UNIFORM_NORMAL_TO_WORLD {}
#define UNIFORM_NORMAL_TO_CAM {}

#define UNIFORM_MATERIAL {}
#define UNIFORM_AVAILABLE_TEXTURES {}

#define UNIFORM_CLASS_INDEX {}
#define UNIFORM_INSTANCE_INDEX {}

#define UNIFORM_STICKER_PROJECTION {}
#define UNIFORM_STICKER_RANGE {}

#define UNIFORM_CAM_POSITION {}
)EOS",
        eVal(Uniform::MeshToObject),
        eVal(Uniform::ObjectToWorld),
        eVal(Uniform::Projection),
        eVal(Uniform::WorldToCam),
        eVal(Uniform::NormalToWorld),
        eVal(Uniform::NormalToCam),
        eVal(Uniform::Material),
        eVal(Uniform::AvailableTextures),
        eVal(Uniform::ClassIndex),
        eVal(Uniform::InstanceIndex),
        eVal(Uniform::StickerProjection),
        eVal(Uniform::StickerRange),
        eVal(Uniform::CamPosition)
    );

    header += Corrade::Utility::formatString(R"EOS(
// Outputs
#define COLOR_OUTPUT_ATTRIBUTE_LOCATION {}
#define OBJECT_COORDINATES_OUTPUT_ATTRIBUTE_LOCATION {}
#define CLASS_INDEX_OUTPUT_ATTRIBUTE_LOCATION {}
#define INSTANCE_INDEX_OUTPUT_ATTRIBUTE_LOCATION {}
#define NORMAL_OUTPUT_ATTRIBUTE_LOCATION {}
#define VERTEX_INDEX_OUTPUT_ATTRIBUTE_LOCATION {}
#define BARYCENTRIC_COEFFS_OUTPUT_ATTRIBUTE_LOCATION {}
#define CAM_COORDINATES_OUTPUT_ATTRIBUTE_LOCATION {}
)EOS",
        ColorOutput,
        ObjectCoordinatesOutput,
        ClassIndexOutput,
        InstanceIndexOutput,
        NormalOutput,
        VertexIndexOutput,
        BarycentricCoeffsOutput,
        CamCoordinatesOutput
    );

    header += R"EOS(
// other stuff
#define M_PI 3.141592653589793
)EOS";

    vert.addSource(header)
        .addSource(rs.get("render_shader.glsl"))
        .addSource(rs.get("render_shader.vert"));

    geom.addSource(header)
        .addSource(rs.get("render_shader.glsl"))
        .addSource(rs.get("render_shader.geom"));

    frag.addSource(header)
        .addSource(rs.get("render_shader.glsl"))
        .addSource(rs.get("render_shader.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, geom, frag}));

    attachShaders({vert, geom, frag});

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

RenderShader& RenderShader::setTransformations(const Magnum::Matrix4& meshToObject, const Magnum::Matrix4& objectToWorld, const Magnum::Matrix4& worldToCam)
{
    // 3D transformation chain mesh -> cam
    setUniform(eVal(Uniform::MeshToObject), meshToObject);
    setUniform(eVal(Uniform::ObjectToWorld), objectToWorld);
    setUniform(eVal(Uniform::WorldToCam), worldToCam);

    // Normal matrices
    Matrix4 meshToWorld = objectToWorld * meshToObject;
    setUniform(eVal(Uniform::NormalToCam), (worldToCam * meshToWorld).normalMatrix());
    setUniform(eVal(Uniform::NormalToWorld), meshToWorld.normalMatrix());

    // Camera position in world frame
    setUniform(eVal(Uniform::CamPosition), worldToCam.invertedRigid().translation());

    return *this;
}

RenderShader& RenderShader::setProjectionMatrix(const Magnum::Matrix4& projection)
{
    setUniform(eVal(Uniform::Projection), projection);

    return *this;
}

RenderShader& RenderShader::setClassIndex(unsigned int classIndex)
{
    setUniform(eVal(Uniform::ClassIndex), classIndex);
    return *this;
}

RenderShader& RenderShader::setInstanceIndex(unsigned int instanceIndex)
{
    setUniform(eVal(Uniform::InstanceIndex), instanceIndex);
    return *this;
}

RenderShader& RenderShader::setLightMap(LightMap& lightMap)
{
    lightMap.irradianceMap().bind(eVal(TextureInput::LightMapIrradiance));
    lightMap.prefilterMap().bind(eVal(TextureInput::LightMapPrefilter));
    lightMap.brdfLUT().bind(eVal(TextureInput::LightMapBRDFLUT));
    return *this;
}

RenderShader& RenderShader::setMaterial(
    const Trade::MaterialData& data,
    const Containers::ArrayView<Containers::Optional<Magnum::GL::Texture2D>>& textures,
    const MaterialOverride& materialOverride)
{
    auto& material = data.as<Trade::PbrMetallicRoughnessMaterialData>();

    if(!material.hasCommonTextureCoordinates())
    {
        Error{} << "We only support common texture coordinates";
        std::exit(1);
    }

    Matrix3 textureMatrix = material.commonTextureMatrix();
    if((textureMatrix - Matrix3{}).toVector().length() > 1e-5)
    {
        Error{} << "We only support identity texture transformation";
        Error{} << "But I got:\n";
        Error{} << textureMatrix;
        std::exit(1);
    }


    Float metallic = material.metalness();
    Float roughness = material.roughness();

    Color4 baseColor = material.baseColor();
    Color4 emissiveFactor = material.emissiveColor();

    if(materialOverride.metallic() >= 0.0f)
        metallic = materialOverride.metallic();

    if(materialOverride.roughness() >= 0.0f)
        roughness = materialOverride.roughness();

    auto materialParameters = Containers::array<Vector4>({
        baseColor,
        emissiveFactor,
        {0.5f, metallic, roughness, 0.0f}
    });

    setUniform(eVal(Uniform::Material), materialParameters);

    // Bind textures
    UnsignedInt availableTextures = 0;

    auto bindTex = [&](UnsignedInt num, TextureInput input){
        if(auto& texture = textures[num])
        {
            texture->bind(eVal(input));
            availableTextures |= (1 << eVal(input));
        }
    };

    if(auto tex = data.tryAttribute<UnsignedInt>(Trade::MaterialAttribute::BaseColorTexture))
        bindTex(*tex, TextureInput::BaseColor);
    else if(auto tex = data.tryAttribute<UnsignedInt>(Trade::MaterialAttribute::DiffuseTexture))
        bindTex(*tex, TextureInput::BaseColor);

    if(auto tex = data.tryAttribute<UnsignedInt>(Trade::MaterialAttribute::EmissiveTexture))
        bindTex(*tex, TextureInput::Emissive);

    if(auto tex = data.tryAttribute<UnsignedInt>(Trade::MaterialAttribute::NormalTexture))
        bindTex(*tex, TextureInput::Normal);

    if(material.hasNoneRoughnessMetallicTexture())
        bindTex(material.roughnessTexture(), TextureInput::MetallicRoughness);

    if(auto tex = data.tryAttribute<UnsignedInt>(Trade::MaterialAttribute::OcclusionTexture))
        bindTex(material.occlusionTexture(), TextureInput::Occlusion);

    setUniform(eVal(Uniform::AvailableTextures), availableTextures);

    return *this;
}

RenderShader& RenderShader::setStickerProjection(const Magnum::Matrix4 proj)
{
    setUniform(eVal(Uniform::StickerProjection), proj);
    return *this;
}

RenderShader& RenderShader::setStickerRange(const Magnum::Range2D& range)
{
    Magnum::Vector4 v{
        range.min().x(),
        range.min().y(),
        std::max(1e-6f, range.size().x()),
        std::max(1e-6f, range.size().y())
    };

    setUniform(eVal(Uniform::StickerRange), v);
    return *this;
}


RenderShader& RenderShader::bindDepthTexture(GL::RectangleTexture& texture)
{
    texture.bind(eVal(TextureInput::MinDepth));
    return *this;
}

RenderShader& RenderShader::bindStickerTexture(Magnum::GL::RectangleTexture& texture)
{
    texture.bind(eVal(TextureInput::Sticker));
    return *this;
}

}
