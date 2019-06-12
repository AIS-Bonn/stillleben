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
#include <Corrade/Containers/Reference.h>

#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Image.h>

#include <Magnum/Trade/ImageData.h>

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

    Magnum::GL::RectangleTexture next();
private:
    using Importer = Magnum::Trade::AbstractImporter;
    using ImporterPtr = Corrade::Containers::Pointer<Importer>;
    using Result = std::pair<ImporterPtr, Magnum::Trade::ImageData2D>;

    void thread();
    void enqueue();

    std::string m_path;
    std::vector<std::string> m_paths;

    std::shared_ptr<sl::Context> m_context;

    std::vector<std::thread> m_threads;

    std::mt19937 m_generator;

    std::mutex m_mutex;
    std::condition_variable m_inputCond;
    std::condition_variable m_outputCond;
    std::atomic<bool> m_shouldExit{false};

    std::queue<ImporterPtr> m_inputQueue;
    std::queue<Result> m_outputQueue;
};

}

#endif
