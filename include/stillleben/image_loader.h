// Multi-threaded image loader
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_IMAGE_LOADER_H
#define STILLLEBEN_IMAGE_LOADER_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <ctime>
#include <random>
#include <atomic>

#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/PluginManager.h>

#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Image.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/AbstractImporter.h>

namespace Magnum
{
namespace Trade
{
    class AbstractImporter;
}
}

namespace sl
{
class Context;

class ImageLoader
{
public:
    explicit ImageLoader(
        const std::string& path,
        const std::shared_ptr<sl::Context>& context,
        std::uint32_t seed = std::time(nullptr)
    );
    ~ImageLoader();

    ImageLoader(const ImageLoader&) = delete;
    ImageLoader& operator=(const ImageLoader&) = delete;

    Magnum::GL::Texture2D nextTexture2D();
    Magnum::GL::RectangleTexture nextRectangleTexture();

    [[deprecated("Use nextRectangleTexture() instead")]]
    Magnum::GL::RectangleTexture next()
    { return nextRectangleTexture(); }
private:
    using Importer = Magnum::Trade::AbstractImporter;
    using ImporterPtr = Corrade::Containers::Pointer<Importer>;
    using Request = std::pair<ImporterPtr, std::string>;

    struct Result
    {
        Result();
        Result(const Result&) = delete;
        Result(Result&&) = default;
        Result(ImporterPtr&& imp);
        Result(ImporterPtr&& imp, Corrade::Containers::Optional<Magnum::Trade::ImageData2D>&& img);

        ~Result();

        Result& operator=(const Result&) = delete;
        Result& operator=(Result&&) = default;

        ImporterPtr importer;
        Corrade::Containers::Optional<Magnum::Trade::ImageData2D> image;
    };

    void thread();
    void enqueue();

    Result nextResult();

    std::string m_path;
    std::vector<std::string> m_paths;

    std::shared_ptr<sl::Context> m_context;

    using ImporterManager = Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter>;
    Corrade::Containers::Pointer<ImporterManager> m_converterManager;
    std::vector<std::thread> m_threads;

    std::mt19937 m_generator;

    std::mutex m_mutex;
    std::condition_variable m_inputCond;
    std::condition_variable m_outputCond;
    std::atomic<bool> m_shouldExit{false};

    std::queue<Request> m_inputQueue;
    std::queue<Result> m_outputQueue;
};

}

#endif
