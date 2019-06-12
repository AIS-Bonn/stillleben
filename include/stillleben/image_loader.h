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

#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/Reference.h>

#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Image.h>

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
    using ImporterRef = Corrade::Containers::Reference<Magnum::Trade::AbstractImporter>;

    void thread(ImporterRef& importer, unsigned int id);

    std::string m_path;
    std::vector<std::string> m_paths;

    std::shared_ptr<sl::Context> m_context;

    std::vector<Corrade::Containers::Pointer<Magnum::Trade::AbstractImporter>> m_importers;
    std::vector<std::thread> m_threads;

    unsigned int m_queueLength;
    std::uint32_t m_seed;

    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::condition_variable m_outputCond;
    bool m_shouldExit = false;
    std::queue<Magnum::Image2D> m_outputQueue;
};

}

#endif
