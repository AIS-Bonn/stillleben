// Multi-threaded image loader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/image_loader.h>
#include <stillleben/context.h>

#include <functional>
#include <random>
#include <chrono>
#include <sstream>
#include <thread>

#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Color.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>

#include <Corrade/Containers/Optional.h>
#include <Corrade/Containers/Array.h>
#include <Corrade/Utility/String.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/FormatStl.h>

using namespace Magnum;

namespace sl
{

ImageLoader::ImageLoader(
    const std::string& path,
    const std::shared_ptr<sl::Context>& context,
    std::uint32_t seed)
 : m_path{path}
 , m_context{context}
 , m_generator{seed}
{
    namespace Dir = Corrade::Utility::Directory;

    m_pluginPath = context->importerPluginPath();

    // List files
    auto files = Dir::list(m_path, Dir::Flag::SkipDirectories | Dir::Flag::SkipDotAndDotDot);
    for(const auto& name : files)
        m_paths.push_back(Dir::join(m_path, name));

    if(m_paths.empty())
        throw std::runtime_error{Utility::formatString("Could not find any images in '{}'", path)};

    for(unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i)
    {
        m_threads.emplace_back(std::bind(&ImageLoader::thread, this));
        enqueue();
    }
}

ImageLoader::~ImageLoader()
{
    m_shouldExit = true;
    m_inputCond.notify_all();

    for(auto& t : m_threads)
        t.join();
}

void ImageLoader::enqueue()
{
    using namespace Corrade::Utility;

    while(1)
    {
        std::uniform_int_distribution<std::size_t> distribution{0, m_paths.size()-1};
        auto path = m_paths[distribution(m_generator)];

        std::string normalized = String::lowercase(path);

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_inputQueue.push(path);
            m_inputCond.notify_one();
        }

        break;
    }
}

void ImageLoader::thread()
{
    Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter> manager{m_pluginPath};

    unsigned int errorCounter = 0;
    auto sendEmptyResult = [&](){
        std::unique_lock<std::mutex> lock(m_mutex);
        m_outputQueue.emplace();
        m_outputCond.notify_all();
        errorCounter++;
    };

    // Swallow any warnings / errors
    // NOTE: This is thread-local as of corrade
    // 2244c61d2c1dbb9e5fceb28daca00b24f4219f3f

    std::stringstream null;
    Corrade::Utility::Error errorRedirect{&null};
    Corrade::Utility::Warning warningRedirect{&null};


    while(1)
    {
        if(errorCounter > 10)
            fprintf(stderr, "Image error: '%s'\n", null.str().c_str());

        // Prevent errors / warnings from piling up
        null.clear();

        Request request;
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            while(m_inputQueue.empty() && !m_shouldExit)
                m_inputCond.wait(lock);

            if(m_shouldExit)
                return;

            request = std::move(m_inputQueue.front());
            m_inputQueue.pop();
        }

        ImporterPtr importer = manager.loadAndInstantiate("AnyImageImporter");
        if(!importer)
        {
            Corrade::Utility::Error{} << "Could not instantiate AnyImageImporter";
            sendEmptyResult();
            continue;
        }

        if(!importer->openFile(request))
        {
            Corrade::Utility::Warning{} << "Could not open file" << request;
            sendEmptyResult();
            continue;
        }

        auto imageData = importer->image2D(0);
        if(!imageData)
        {
            sendEmptyResult();
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_outputQueue.emplace(std::move(imageData));
            m_outputCond.notify_all();
        }

        errorCounter = 0;
    }
}

ImageLoader::Result ImageLoader::nextResult()
{
    unsigned int errorCounter = 0;

    while(1)
    {
        if(errorCounter >= 10)
        {
            using namespace std::chrono_literals;

            Magnum::Warning{} << "ImageLoader: 10 errors in a row, probably something is wrong with your images";
            std::this_thread::sleep_for(500ms); // crude rate limiting
            errorCounter = 0;
        }

        enqueue();

        Result result;
        {
            std::unique_lock lock(m_mutex);
            while(m_outputQueue.empty())
                m_outputCond.wait(lock);

            result = std::move(m_outputQueue.front());
            m_outputQueue.pop();
        }

        // If some error occured in the worker thread, enqueue a new image
        // and try again.
        if(!result)
        {
            errorCounter++;
            continue;
        }

        return result;
    }
}

Magnum::GL::RectangleTexture ImageLoader::nextRectangleTexture()
{
    using namespace Magnum;

    while(1)
    {
        Result result = nextResult();

        GL::TextureFormat format;
        if(result->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(result->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
            continue; // just try the next one

        GL::RectangleTexture texture;
        texture.setStorage(format, result->size());
        texture.setSubImage({}, *result);

        // Needed for sticker textures - this is ugly.
        texture.setWrapping(Magnum::SamplerWrapping::ClampToBorder);
        texture.setBorderColor(Magnum::Color4{0.0, 0.0, 0.0, 0.0});

        return texture;
    }
}

Magnum::GL::Texture2D ImageLoader::nextTexture2D()
{
    using namespace Magnum;

    while(1)
    {
        Result result = nextResult();

        GL::TextureFormat format;
        if(result->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(result->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
            continue; // just try the next one

        GL::Texture2D texture;

        // Needed for sticker textures - this is ugly.
        texture.setWrapping(Magnum::SamplerWrapping::ClampToBorder);
        texture.setBorderColor(Magnum::Color4{0.0, 0.0, 0.0, 0.0});

        texture.setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy());

        texture.setStorage(Math::log2(result->size().max()), format, result->size());
        texture.setSubImage(0, {}, *result);
        texture.generateMipmap();

        return texture;
    }
}

}
