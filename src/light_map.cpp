// IBL light map
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/light_map.h>
#include <stillleben/context.h>

#include "shaders/cubemap_shader.h"
#include "shaders/brdf_shader.h"

#include <Corrade/Containers/Optional.h>

#include <Corrade/PluginManager/PluginManager.h>
#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Directory.h>

#include <Magnum/GL/CubeMapTexture.h>
#include <Magnum/GL/DebugOutput.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Image.h>


using namespace Corrade;
using namespace Magnum;

namespace sl
{

namespace
{
    constexpr bool DEBUG_OUTPUT = false;

    struct IBLSpec
    {
        std::string file;
        float gamma = 1.0f;

        static Containers::Optional<IBLSpec> load(const Utility::ConfigurationGroup& group, const std::string& prefix)
        {
            using namespace Utility;

            std::string fileTag = prefix + "file";
            if(!group.hasValue(fileTag))
            {
                Error{} << "IBL file does not contain" << fileTag;
                return {};
            }

            std::string mapTag = prefix + "map";
            if(!group.hasValue(mapTag))
            {
                Error{} << "IBL file does not contain" << mapTag;
                return {};
            }

            unsigned int mapMode = group.value<unsigned int>(mapTag);
            if(mapMode != 1)
            {
                Error{} << "IBL file uses unsupported mapping mode" << mapMode;
                return {};
            }

            std::string gammaTag = prefix + "gamma";

            IBLSpec ret;

            ret.file = group.value(fileTag);

            if(group.hasValue(gammaTag))
                ret.gamma = group.value<float>(gammaTag);

            return ret;
        }
    };

    Containers::Optional<Magnum::GL::Texture2D> loadTexture(const IBLSpec& spec, Trade::AbstractImporter& importer, const std::string& baseDir)
    {
        auto fullPath = Corrade::Utility::Directory::join({baseDir, spec.file});
        if(!importer.openFile(fullPath))
            return {};

        auto image = importer.image2D(0);
        if(!image)
        {
            Error{} << "Could not load image from" << fullPath;
            return {};
        }

        Magnum::GL::Texture2D texture;
        texture.setMagnificationFilter(GL::SamplerFilter::Linear)
            .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Linear)
            .setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy())
            .setStorage(Math::log2(image->size().x())+1, GL::TextureFormat::RGB32F, image->size())
            .setSubImage(0, {}, *image)
            .generateMipmap();

        return texture;
    }

    struct CubeMapSide
    {
        GL::CubeMapCoordinate coordinate;
        Matrix4 view;
    };

    const std::array<CubeMapSide, 6> CUBE_MAP_SIDES{{
        {GL::CubeMapCoordinate::PositiveX, Matrix4::lookAt({}, { 1.0f,  0.0f,  0.0f}, {0.0f, -1.0f,  0.0f}).invertedRigid()},
        {GL::CubeMapCoordinate::NegativeX, Matrix4::lookAt({}, {-1.0f,  0.0f,  0.0f}, {0.0f, -1.0f,  0.0f}).invertedRigid()},
        {GL::CubeMapCoordinate::PositiveY, Matrix4::lookAt({}, { 0.0f,  1.0f,  0.0f}, {0.0f,  0.0f,  1.0f}).invertedRigid()},
        {GL::CubeMapCoordinate::NegativeY, Matrix4::lookAt({}, { 0.0f, -1.0f,  0.0f}, {0.0f,  0.0f, -1.0f}).invertedRigid()},
        {GL::CubeMapCoordinate::PositiveZ, Matrix4::lookAt({}, { 0.0f,  0.0f,  1.0f}, {0.0f, -1.0f,  0.0f}).invertedRigid()},
        {GL::CubeMapCoordinate::NegativeZ, Matrix4::lookAt({}, { 0.0f,  0.0f, -1.0f}, {0.0f, -1.0f,  0.0f}).invertedRigid()},
    }};

    Trade::MeshData3D cubeFromInside()
    {
        return Trade::MeshData3D{MeshPrimitive::Triangles, {
             0,  2,  1,  0,  3,  2, /* +Z */
             4,  6,  5,  4,  7,  6, /* +X */
             8, 10,  9,  8, 11, 10, /* +Y */
            12, 14, 13, 12, 15, 14, /* -Z */
            16, 18, 17, 16, 19, 18, /* -Y */
            20, 22, 21, 20, 23, 22  /* -X */
        }, {{
            {-1.0f, -1.0f,  1.0f},
            { 1.0f, -1.0f,  1.0f},
            { 1.0f,  1.0f,  1.0f}, /* +Z */
            {-1.0f,  1.0f,  1.0f},

            { 1.0f, -1.0f,  1.0f},
            { 1.0f, -1.0f, -1.0f},
            { 1.0f,  1.0f, -1.0f}, /* +X */
            { 1.0f,  1.0f,  1.0f},

            {-1.0f,  1.0f,  1.0f},
            { 1.0f,  1.0f,  1.0f},
            { 1.0f,  1.0f, -1.0f}, /* +Y */
            {-1.0f,  1.0f, -1.0f},

            { 1.0f, -1.0f, -1.0f},
            {-1.0f, -1.0f, -1.0f},
            {-1.0f,  1.0f, -1.0f}, /* -Z */
            { 1.0f,  1.0f, -1.0f},

            {-1.0f, -1.0f, -1.0f},
            { 1.0f, -1.0f, -1.0f},
            { 1.0f, -1.0f,  1.0f}, /* -Y */
            {-1.0f, -1.0f,  1.0f},

            {-1.0f, -1.0f, -1.0f},
            {-1.0f, -1.0f,  1.0f},
            {-1.0f,  1.0f,  1.0f}, /* -X */
            {-1.0f,  1.0f, -1.0f}
        }}, {{
            { 0.0f,  0.0f, -1.0f},
            { 0.0f,  0.0f, -1.0f},
            { 0.0f,  0.0f, -1.0f}, /* +Z */
            { 0.0f,  0.0f, -1.0f},

            {-1.0f,  0.0f,  0.0f},
            {-1.0f,  0.0f,  0.0f},
            {-1.0f,  0.0f,  0.0f}, /* +X */
            {-1.0f,  0.0f,  0.0f},

            { 0.0f, -1.0f,  0.0f},
            { 0.0f, -1.0f,  0.0f},
            { 0.0f, -1.0f,  0.0f}, /* +Y */
            { 0.0f, -1.0f,  0.0f},

            { 0.0f,  0.0f,  1.0f},
            { 0.0f,  0.0f,  1.0f},
            { 0.0f,  0.0f,  1.0f}, /* -Z */
            { 0.0f,  0.0f,  1.0f},

            { 0.0f,  1.0f,  0.0f},
            { 0.0f,  1.0f,  0.0f},
            { 0.0f,  1.0f,  0.0f}, /* -Y */
            { 0.0f,  1.0f,  0.0f},

            { 1.0f,  0.0f,  0.0f},
            { 1.0f,  0.0f,  0.0f},
            { 1.0f,  0.0f,  0.0f}, /* -X */
            { 1.0f,  0.0f,  0.0f}
        }}, {}, {}, nullptr};
    }
}

LightMap::LightMap()
{
}

LightMap::LightMap(const std::string& path, const std::shared_ptr<Context>& ctx)
{
    if(!load(path, ctx))
        throw std::runtime_error("Could not load light map " + path);
}


bool LightMap::load(const std::string& path, const std::shared_ptr<Context>& ctx)
{
    using namespace Utility;

    Configuration config{path, Configuration::Flag::ReadOnly};

    if(config.isEmpty())
    {
        Error{} << "Could not open .ibl file:" << path;
        return false;
    }

    std::string baseDir = Directory::path(path);

    auto reflectionGroup = config.group("Reflection");
    if(!reflectionGroup)
    {
        Error{} << path << "does not contain a Reflection group";
        return false;
    }

    auto refSpec = IBLSpec::load(*reflectionGroup, "REF");

    if(!refSpec)
        return false;

    // Load the texture!
    Magnum::GL::Texture2D hdrEquirectangular{NoCreate};
    {
        Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter> manager(ctx->importerPluginPath());
        auto importer = manager.loadAndInstantiate("StbImageImporter");
        if(!importer) Fatal{} << "Cannot load the StbImageImporter plugin";

        auto refTex = loadTexture(*refSpec, *importer, baseDir);
        if(!refTex)
        {
            Error{} << "Could not load ref texture";
            return false;
        }

        hdrEquirectangular = std::move(*refTex);
    }

    // Setup a framebuffer for the remapping / convolution operations
    GL::Renderer::enable(GL::Renderer::Feature::SeamlessCubeMapTexture);

    auto viewport = Magnum::Vector2i{{512, 512}};

    GL::Framebuffer framebuffer{Range2Di::fromSize({}, viewport)};
    GL::Renderbuffer depthBuffer;

    depthBuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, viewport);
    framebuffer.attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depthBuffer);

    // Setup the mesh primitive
    // NOTE: Primitives::cubeSolid() has normals facing outwards
    auto cube = MeshTools::compile(cubeFromInside());
    Matrix4 perspective = Matrix4::perspectiveProjection(Deg{90.0}, 1.0f, 0.1f, 10.0f);

    // Equirectangular -> Cubemap
    GL::CubeMapTexture hdrCubeMap;
    {
        hdrCubeMap
            .setStorage(Math::log2(viewport.x())+1, GL::TextureFormat::RGBA32F, viewport)
            .setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Linear)
            .setMagnificationFilter(GL::SamplerFilter::Linear);

        CubeMapShader shader{CubeMapShader::Phase::EquirectangularConversion};
        shader.setProjection(perspective);
        shader.bindInputTexture(hdrEquirectangular);

        for(const auto& side : CUBE_MAP_SIDES)
        {
            framebuffer.attachCubeMapTexture(
                GL::Framebuffer::ColorAttachment{0}, hdrCubeMap,
                side.coordinate, 0
            );
            framebuffer.mapForDraw({{0, GL::Framebuffer::ColorAttachment{0}}});
            framebuffer.bind();

            if(framebuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
            {
                Error{} << "Invalid framebuffer status:" << framebuffer.checkStatus(GL::FramebufferTarget::Draw);
                std::abort();
            }

            framebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);

            shader.setView(side.view);

            cube.draw(shader);
        }

        hdrCubeMap.generateMipmap();

        if constexpr(DEBUG_OUTPUT)
        {
            Corrade::PluginManager::Manager<Magnum::Trade::AbstractImageConverter> manager(ctx->imageConverterPluginPath());

            Image2D image = hdrCubeMap.image(GL::CubeMapCoordinate::PositiveX, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/cubemap_xp.png");

            image = hdrCubeMap.image(GL::CubeMapCoordinate::NegativeX, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/cubemap_xn.png");

            image = hdrCubeMap.image(GL::CubeMapCoordinate::PositiveY, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/cubemap_yp.png");

            image = hdrCubeMap.image(GL::CubeMapCoordinate::NegativeY, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/cubemap_yn.png");

            image = hdrCubeMap.image(GL::CubeMapCoordinate::PositiveZ, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/cubemap_zp.png");

            image = hdrCubeMap.image(GL::CubeMapCoordinate::NegativeZ, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/cubemap_zn.png");
        }
    }

    // Create irradiance cubemap
    GL::CubeMapTexture hdrIrradiance;
    {
        const Vector2i irradianceSize(32, 32);

        hdrIrradiance
            .setStorage(Math::log2(irradianceSize.x())+1, GL::TextureFormat::RGBA32F, irradianceSize)
            .setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Linear)
            .setMagnificationFilter(GL::SamplerFilter::Linear);

        framebuffer.setViewport(Range2Di::fromSize({}, irradianceSize));
        depthBuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, irradianceSize);
        framebuffer.attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depthBuffer);

        CubeMapShader shader{CubeMapShader::Phase::IrradianceConvolution};
        shader.setProjection(perspective);
        shader.bindInputTexture(hdrCubeMap);

        for(const auto& side : CUBE_MAP_SIDES)
        {
            framebuffer.attachCubeMapTexture(
                GL::Framebuffer::ColorAttachment{0}, hdrIrradiance,
                side.coordinate, 0
            );
            framebuffer.mapForDraw({{0, GL::Framebuffer::ColorAttachment{0}}});
            framebuffer.bind();

            framebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);

            shader.setView(side.view);

            cube.draw(shader);
        }

        if constexpr(DEBUG_OUTPUT)
        {
            Corrade::PluginManager::Manager<Magnum::Trade::AbstractImageConverter> manager(ctx->imageConverterPluginPath());

            Image2D image = hdrIrradiance.image(GL::CubeMapCoordinate::PositiveX, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/irradiance_xp.png");

            image = hdrIrradiance.image(GL::CubeMapCoordinate::NegativeX, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/irradiance_xn.png");

            image = hdrIrradiance.image(GL::CubeMapCoordinate::PositiveY, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/irradiance_yp.png");

            image = hdrIrradiance.image(GL::CubeMapCoordinate::NegativeY, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/irradiance_yn.png");

            image = hdrIrradiance.image(GL::CubeMapCoordinate::PositiveZ, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/irradiance_zp.png");

            image = hdrIrradiance.image(GL::CubeMapCoordinate::NegativeZ, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/irradiance_zn.png");
        }

        hdrIrradiance.generateMipmap();
    }

    // Create pre-filter cubemap
    GL::CubeMapTexture hdrPrefilter;
    {
        const Vector2i prefilterSize(128, 128);

        const unsigned int MAX_MIP_LEVELS = 5;

        hdrPrefilter
            .setStorage(MAX_MIP_LEVELS, GL::TextureFormat::RGBA32F, prefilterSize)
            .setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Linear)
            .setMagnificationFilter(GL::SamplerFilter::Linear)
            .generateMipmap();

        CubeMapShader shader{CubeMapShader::Phase::Prefilter};
        shader.setProjection(perspective);
        shader.bindInputTexture(hdrCubeMap);

        for(unsigned int mip = 0; mip < MAX_MIP_LEVELS; ++mip)
        {
            const Vector2i mipSize(
                prefilterSize.x() * Math::pow(0.5f, static_cast<float>(mip)),
                prefilterSize.y() * Math::pow(0.5f, static_cast<float>(mip))
            );

            framebuffer.setViewport(Range2Di::fromSize({}, mipSize));
            depthBuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, mipSize);
            framebuffer.attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depthBuffer);

            const float roughness = static_cast<float>(mip) / (MAX_MIP_LEVELS - 1);
            shader.setRoughness(roughness);

            for(const auto& side : CUBE_MAP_SIDES)
            {
                framebuffer.attachCubeMapTexture(
                    GL::Framebuffer::ColorAttachment{0}, hdrPrefilter,
                    side.coordinate, mip
                );
                framebuffer.mapForDraw({{0, GL::Framebuffer::ColorAttachment{0}}});
                framebuffer.bind();

                framebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);

                shader.setView(side.view);

                cube.draw(shader);
            }
        }

        if constexpr(DEBUG_OUTPUT)
        {
            Corrade::PluginManager::Manager<Magnum::Trade::AbstractImageConverter> manager(ctx->imageConverterPluginPath());

            Image2D image = hdrPrefilter.image(GL::CubeMapCoordinate::PositiveX, 0, {PixelFormat::RGBA8Unorm});
            manager.loadAndInstantiate("PngImageConverter")->exportToFile(image, "/tmp/prefilter.png");
        }
    }

    // Generate 2D LUT for the BRDF equations
    GL::Texture2D brdfLUT;
    {
        const Vector2i lutSize(512, 512);

        brdfLUT
            .setStorage(Math::log2(lutSize.x())+1, GL::TextureFormat::RGBA32F, lutSize)
            .setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Linear)
            .setMagnificationFilter(GL::SamplerFilter::Linear);

        framebuffer.setViewport(Range2Di::fromSize({}, lutSize));
        depthBuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, lutSize);
        framebuffer.attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depthBuffer);

        framebuffer.attachTexture(GL::Framebuffer::ColorAttachment{0}, brdfLUT, 0);

        framebuffer.mapForDraw({{0, GL::Framebuffer::ColorAttachment{0}}});
        framebuffer.bind();

        framebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);

        BRDFShader shader;

        auto quad = MeshTools::compile(Primitives::planeSolid(Primitives::PlaneTextureCoords::Generate));
        quad.draw(shader);

        brdfLUT.generateMipmap();
    }

    m_cubeMap = std::move(hdrCubeMap);
    m_irradiance = std::move(hdrIrradiance);
    m_prefilter = std::move(hdrPrefilter);
    m_brdfLUT = std::move(brdfLUT);

    m_path = path;
    return true;
}

}
