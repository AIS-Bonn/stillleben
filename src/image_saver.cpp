// Multi-threaded image saver
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/image_saver.h>

#include <stillleben/context.h>

#include <Corrade/Containers/StringView.h>
#include <Corrade/Containers/StringStl.h>

#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Utility/FormatStl.h>

#include <Magnum/Trade/AbstractImageConverter.h>

using namespace Magnum;


namespace sl
{

ImageSaver::ImageSaver(const std::shared_ptr<sl::Context>& context)
{
    m_pluginPath = context->imageConverterPluginPath();

    for(unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i)
    {
        m_threads.emplace_back(std::bind(&ImageSaver::thread, this));
    }
}

ImageSaver::~ImageSaver()
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_inputFreeCond.wait(lock, [&](){
            return m_inputQueueSize == 0;
        });
    }

    m_shouldExit = true;
    m_inputCond.notify_all();

    for(auto& t : m_threads)
        t.join();

    drain();
}

void ImageSaver::drain()
{
    while(!m_outputQueue.empty())
    {
        m_outputQueue.front().deleter();
        m_outputQueue.pop();
    }
}

void ImageSaver::thread()
{
    PluginManager::Manager<Trade::AbstractImageConverter> manager{m_pluginPath};

    Containers::Pointer<Trade::AbstractImageConverter> converter = manager.loadAndInstantiate("AnyImageConverter");

    if(!converter)
        throw std::runtime_error{"Could not load AnyImageConverter"};

    while(true)
    {
        Job image;

        {
            std::unique_lock<std::mutex> lock(m_mutex);

            m_inputCond.wait(lock, [&](){
                return !m_inputQueue.empty() || m_shouldExit;
            });

            if(m_shouldExit)
                return;

            image = std::move(m_inputQueue.front());
            m_inputQueue.pop();
            m_inputQueueSize--;
            m_inputFreeCond.notify_all();
        }

        if(!converter->convertToFile(image.image, image.path))
            throw std::runtime_error{Utility::formatString("Could not write image {}", image.path)};

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_outputQueue.push(std::move(image));
        }
    }
}

void ImageSaver::save(sl::ImageSaver::Job && image)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    drain();

    m_inputFreeCond.wait(lock, [&](){
        return m_inputQueueSize < 2*m_threads.size();
    });

    m_inputQueue.push(std::move(image));
    m_inputQueueSize++;
    m_inputCond.notify_one();
}

}
