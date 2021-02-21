// Multi-threaded image saver
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_IMAGE_SAVER_H
#define STILLLEBEN_IMAGE_SAVER_H

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

class ImageRef
{
    Magnum::ImageView2D image;
    std::function<void()> deleter;
};

class ImageSaver
{
public:
    explicit ImageSaver();
    ~ImageSaver();

    ImageSaver(const ImageSaver&) = delete;
    ImageSaver& operator=(const ImageSaver&) = delete;

    void save(
private:
    using Importer = Magnum::Trade::AbstractImporter;
    using ImporterPtr = Corrade::Containers::Pointer<Importer>;
    using Request = std::string;

    using Result = Corrade::Containers::Optional<Magnum::Trade::ImageData2D>;

    void thread();
    void enqueue();

    std::string m_path;
    std::vector<std::string> m_paths;

    std::shared_ptr<sl::Context> m_context;

    std::vector<std::thread> m_threads;

    std::mutex m_mutex;
    std::condition_variable m_inputCond;
    std::condition_variable m_outputCond;
    std::atomic<bool> m_shouldExit{false};

    std::queue<Request> m_inputQueue;
    std::size_t m_jobsInFlight = 0;

    std::string m_pluginPath;
};

}

#endif
