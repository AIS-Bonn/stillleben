// Multi-threaded image loader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/image_loader.h>
#include <stillleben/context.h>

#include <functional>
#include <random>

#include <experimental/filesystem>

#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>

#include <Corrade/Containers/Optional.h>
#include <Corrade/Containers/Array.h>

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
    while(1)
    {
        auto importer = m_context->instantiateImporter("AnyImageImporter");
        if(!importer)
            throw std::logic_error("Could not load AnyImageImporter plugin");

        std::uniform_int_distribution<std::size_t> distribution{0, m_paths.size()-1};
        auto path = m_paths[distribution(m_generator)];

        if(!importer->openFile(path))
        {
            Corrade::Utility::Warning{} << "Could not open file" << path;
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_inputQueue.push(std::move(importer));
            m_inputCond.notify_one();
        }

        break;
    }
}

void ImageLoader::thread()
{
    while(1)
    {
        ImporterPtr importer;
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            while(m_inputQueue.empty() && !m_shouldExit)
                m_inputCond.wait(lock);

            if(m_shouldExit)
                return;

            importer = std::move(m_inputQueue.front());
            m_inputQueue.pop();
        }

        auto imageData = importer->image2D(0);
        if(!imageData)
            continue;

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_outputQueue.emplace(std::move(importer), std::move(*imageData));
            m_outputCond.notify_all();
        }
    }
}

Magnum::GL::RectangleTexture ImageLoader::next()
{
    using namespace Magnum;

    while(1)
    {
        enqueue();

        Corrade::Containers::Optional<Result> result;
        {
            std::unique_lock lock(m_mutex);
            while(m_outputQueue.empty())
                m_outputCond.wait(lock);

            result = std::move(m_outputQueue.front());
            m_outputQueue.pop();
        }

        GL::TextureFormat format;
        if(result->second.format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(result->second.format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
        {
            Warning{} << "Unsupported texture format:" << result->second.format();
            continue; // just try the next one
        }

        GL::RectangleTexture texture;
        texture.setStorage(format, result->second.size());
        texture.setSubImage({}, result->second);

        return texture;
    }
}

}
