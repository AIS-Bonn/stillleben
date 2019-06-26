// IBL light map
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/light_map.h>
#include <stillleben/context.h>

#include <Corrade/Containers/Optional.h>

#include <Corrade/PluginManager/PluginManager.h>
#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/Directory.h>

#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Image.h>


using namespace Corrade;
using namespace Magnum;

namespace sl
{

namespace
{
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
            .setStorage(Math::log2(image->size().x())+1, GL::TextureFormat::RGBA32F, image->size())
            .setSubImage(0, {}, *image)
            .generateMipmap();

        return texture;
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

    auto environmentGroup = config.group("Enviroment"); // sic!
    if(!environmentGroup)
    {
        Error{} << path << "does not contain an Enviroment (sic!) group";
        return false;
    }

    auto reflectionGroup = config.group("Reflection");
    if(!reflectionGroup)
    {
        Error{} << path << "does not contain a Reflection group";
        return false;
    }

    auto envSpec = IBLSpec::load(*environmentGroup, "EV");
    auto refSpec = IBLSpec::load(*reflectionGroup, "REF");

    if(!envSpec || !refSpec)
        return false;

    {
        auto importer = ctx->instantiateImporter("StbImageImporter");
        if(!importer) Fatal{} << "Cannot load the StbImageImporter plugin";

        auto envTex = loadTexture(*envSpec, *importer, baseDir);
        if(!envTex)
        {
            Error{} << "Could not load env texture";
            return false;
        }

        auto refTex = loadTexture(*refSpec, *importer, baseDir);
        if(!refTex)
        {
            Error{} << "Could not load ref texture";
            return false;
        }

        m_diffuseTexture = std::move(*envTex);
        m_specularTexture = std::move(*refTex);
    }

    m_path = path;
    return true;
}

}
