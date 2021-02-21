// Multi-threaded image saver
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_IMAGE_SAVER_H
#define STILLLEBEN_IMAGE_SAVER_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>

#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>

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

class ImageSaver
{
public:
    class Job
    {
    public:
        Job() = default;

        Job(const Job&) = delete;
        Job& operator=(const Job&) = delete;

        Job(Job&&) = default;
        Job& operator=(Job&&) = default;

        Magnum::ImageView2D image{Magnum::PixelFormat::RGB8Unorm, {}};
        std::string path;
        std::function<void()> deleter;
    };

    explicit ImageSaver(const std::shared_ptr<sl::Context>& context);
    ~ImageSaver();

    ImageSaver(const ImageSaver&) = delete;
    ImageSaver& operator=(const ImageSaver&) = delete;

    void save(Job&& image);

private:
    void thread();
    void drain();

    std::vector<std::thread> m_threads;

    std::mutex m_mutex;
    std::condition_variable m_inputCond;
    std::condition_variable m_inputFreeCond;
    std::atomic<bool> m_shouldExit{false};

    std::queue<Job> m_inputQueue;
    std::queue<Job> m_outputQueue;
    std::size_t m_inputQueueSize = 0;
    std::size_t m_outputQueueSize = 0;

    std::string m_pluginPath;
};

}

#endif
