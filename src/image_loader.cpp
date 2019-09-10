// Multi-threaded image loader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/image_loader.h>
#include <stillleben/context.h>

#include <functional>
#include <random>
#include <chrono>
#include <sstream>

#include <experimental/filesystem>

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
    // List files
    namespace fs = std::experimental::filesystem;
    fs::directory_iterator it{m_path};
    fs::directory_iterator end;
    for(; it != end; ++it)
    {
        m_paths.push_back(fs::absolute(it->path()).string());
    }

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
    auto sendEmptyResult = [this](){
        std::unique_lock<std::mutex> lock(m_mutex);
        m_outputQueue.emplace();
        m_outputCond.notify_all();
    };

    // Swallow any warnings / errors
    // NOTE: This is thread-local as of corrade
    // 2244c61d2c1dbb9e5fceb28daca00b24f4219f3f

    std::stringstream null;
    Corrade::Utility::Error errorRedirect{&null};
    Corrade::Utility::Warning warningRedirect{&null};

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
                sendEmptyResult();
                continue;
            }
        }

        auto imageData = importer->image2D(0);
        if(!imageData)
        {
            sendEmptyResult();
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_outputQueue.emplace(Result{std::move(importer), std::move(*imageData)});
            m_outputCond.notify_all();
        }
    }
}

Magnum::GL::RectangleTexture ImageLoader::next()
{
    using namespace Magnum;

    unsigned int errorCounter = 0;

    while(1)
    {
        if(errorCounter >= 10)
        {
            Warning{} << "ImageLoader: 10 errors in a row, probably something is wrong with your images";
            errorCounter = 0;
        }

        enqueue();

        Corrade::Containers::Optional<Result> result;
        {
            std::unique_lock lock(m_mutex);
            while(m_outputQueue.empty())
                m_outputCond.wait(lock);

            result = std::move(m_outputQueue.front());
            m_outputQueue.pop();
        }

        // If some error occured in the worker thread, enqueue a new image
        // and try again.
        if(!result || !result->first)
        {
            errorCounter++;
            continue;
        }

        GL::TextureFormat format;
        if(result->second.format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(result->second.format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
        {
//             Warning{} << "Unsupported texture format:" << result->second.format();
            errorCounter++;
            continue; // just try the next one
        }

        errorCounter = 0;

        GL::RectangleTexture texture;
        texture.setStorage(format, result->second.size());
        texture.setSubImage({}, result->second);

        // Needed for sticker textures - this is ugly.
        texture.setWrapping(Magnum::SamplerWrapping::ClampToBorder);
        texture.setBorderColor(Magnum::Color4{0.0, 0.0, 0.0, 0.0});

        return texture;
    }
}

}
