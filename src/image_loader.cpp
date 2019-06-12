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
 , m_queueLength{std::max(10u, std::thread::hardware_concurrency())}
 , m_seed{seed}
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
        auto importer = m_context->instantiateImporter("AnyImageImporter");
        if(!importer)
            throw std::logic_error("Could not load AnyImageImporter plugin");

        m_threads.emplace_back(std::bind(&ImageLoader::thread, this, ImporterRef(*importer), i));
        m_importers.push_back(std::move(importer));
    }
}

ImageLoader::~ImageLoader()
{
    m_shouldExit = true;
    m_cond.notify_all();

    for(auto& t : m_threads)
        t.join();
}

void ImageLoader::thread(ImporterRef& importer, unsigned int id)
{
    std::mt19937 rd{m_seed + id};
    std::uniform_int_distribution<std::size_t> distribution(0, m_paths.size()-1);

    while(1)
    {
        auto path = m_paths[distribution(rd)];

        if(!importer->openFile(path))
            continue;

        auto imageData = importer->image2D(0);
        if(!imageData)
            continue;

        Corrade::Containers::Array<char> data{Corrade::Containers::NoInit, imageData->data().size()};
        std::uninitialized_copy(imageData->data().begin(), imageData->data().end(), data.begin());
        Magnum::Image2D img{
            imageData->storage(),
            imageData->format(),
            imageData->size(),
            std::move(data)
        };

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            while(!m_shouldExit && m_outputQueue.size() >= m_queueLength)
                m_cond.wait(lock);

            if(m_shouldExit)
                return;

            m_outputQueue.push(std::move(img));
            m_outputCond.notify_all();
        }
    }
}

Magnum::GL::RectangleTexture ImageLoader::next()
{
    using namespace Magnum;

    while(1)
    {
        Corrade::Containers::Optional<Image2D> img;
        {
            std::unique_lock lock(m_mutex);
            while(m_outputQueue.empty())
                m_outputCond.wait(lock);

            img = std::move(m_outputQueue.front());
            m_outputQueue.pop();
            m_cond.notify_one();
        }

        GL::TextureFormat format;
        if(img->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(img->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else
        {
            Warning{} << "Unsupported texture format:" << img->format();
            continue; // just try the next one
        }

        GL::RectangleTexture texture;
        texture.setStorage(format, img->size());
        texture.setSubImage({}, *img);

        return texture;
    }
}

}
