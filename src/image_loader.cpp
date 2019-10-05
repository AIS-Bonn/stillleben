// Multi-threaded image loader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/image_loader.h>
#include <stillleben/context.h>

#include <functional>
#include <random>
#include <chrono>
#include <sstream>

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

namespace sl
{

ImageLoader::Result::Result()
{
}

ImageLoader::Result::~Result()
{
}

ImageLoader::Result::Result(ImageLoader::ImporterPtr&& imp)
 : importer{std::move(imp)}
{}

ImageLoader::Result::Result(ImageLoader::ImporterPtr&& imp, Corrade::Containers::Optional<Magnum::Trade::ImageData2D>&& img)
 : importer{std::move(imp)}, image{std::move(img)}
{}

ImageLoader::ImageLoader(
    const std::string& path,
    const std::shared_ptr<sl::Context>& context,
    std::uint32_t seed)
 : m_path{path}
 , m_context{context}
 , m_generator{seed}
{
    namespace Dir = Corrade::Utility::Directory;

    // List files
    auto files = Dir::list(m_path, Dir::Flag::SkipDirectories | Dir::Flag::SkipDotAndDotDot);
    for(const auto& name : files)
        m_paths.push_back(Dir::join(m_path, name));

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

        ImporterPtr importer;
        bool openHere = false;
        if(String::endsWith(normalized, ".png"))
            importer = m_context->instantiateImporter("PngImporter");
        else if(String::endsWith(normalized, ".jpeg") || String::endsWith(normalized, ".jpg"))
            importer = m_context->instantiateImporter("JpegImporter");
        else
        {
            importer = m_context->instantiateImporter("AnyImageImporter");
            openHere = true;
        }

        if(!importer)
            throw std::logic_error("Could not load AnyImageImporter plugin");

        if(openHere)
        {
            if(!importer->openFile(path))
            {
                Corrade::Utility::Warning{} << "Could not open file" << path;
                continue;
            }
            path = {};
        }

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_inputQueue.emplace(std::move(importer), path);
            m_inputCond.notify_one();
        }

        break;
    }
}

void ImageLoader::thread()
{
    // We need to be careful here not to destroy the importer instance, as
    // the Corrade plugin system is not thread-safe. Instead, destruction
    // should be done in the main thread.

    auto sendEmptyResult = [this](ImporterPtr&& importer){
        std::unique_lock<std::mutex> lock(m_mutex);
        m_outputQueue.emplace(std::move(importer));
        m_outputCond.notify_all();
    };

    // Swallow any warnings / errors
    // NOTE: This is thread-local as of corrade
    // 2244c61d2c1dbb9e5fceb28daca00b24f4219f3f

    std::stringstream null;
    Corrade::Utility::Error errorRedirect{&null};
    Corrade::Utility::Warning warningRedirect{&null};

    unsigned int errorCounter = 0;

    while(1)
    {
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

        ImporterPtr& importer = request.first;

        if(!request.second.empty())
        {
            if(!importer->openFile(request.second))
            {
                Corrade::Utility::Warning{} << "Could not open file" << request.second;
                sendEmptyResult(std::move(importer));
                continue;
            }
        }

        auto imageData = importer->image2D(0);
        if(!imageData)
        {
            errorCounter++;
            if(errorCounter > 10)
                fprintf(stderr, "Image error: '%s'\n", null.str().c_str());

            sendEmptyResult(std::move(importer));
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_outputQueue.emplace(std::move(importer), std::move(imageData));
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
            Magnum::Warning{} << "ImageLoader: 10 errors in a row, probably something is wrong with your images";
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
        if(!result.image)
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
        if(result.image->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(result.image->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
            continue; // just try the next one

        GL::RectangleTexture texture;
        texture.setStorage(format, result.image->size());
        texture.setSubImage({}, *result.image);

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
        if(result.image->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(result.image->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
            continue; // just try the next one

        GL::Texture2D texture;

        // Needed for sticker textures - this is ugly.
        texture.setWrapping(Magnum::SamplerWrapping::ClampToBorder);
        texture.setBorderColor(Magnum::Color4{0.0, 0.0, 0.0, 0.0});

        texture.setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy());

        texture.setStorage(Math::log2(result.image->size().max()), format, result.image->size());
        texture.setSubImage(0, {}, *result.image);
        texture.generateMipmap();

        return texture;
    }
}

}
