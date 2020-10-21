// Threaded physics simulator
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/physics_sim.h>

#include <stillleben/scene.h>

#include <Corrade/Containers/Array.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

using namespace Corrade;

namespace sl
{

class PhysicsSim::Private
{
public:
    Private(int numThreads)
    {
        if(numThreads == -1)
        {
            numThreads = std::thread::hardware_concurrency() / 4;
            if(numThreads < 1)
                numThreads = 1;
        }

        CORRADE_INTERNAL_ASSERT(numThreads >= 1);

        m_threads = Containers::Array<std::thread>{
            Containers::DirectInit, static_cast<std::size_t>(numThreads), [&](){
                worker();
            }
        };
    }

    ~Private()
    {
        stop();
    }

    void addScene(const std::shared_ptr<sl::Scene>& scene)
    {
        {
            std::unique_lock lock{m_workQueueMutex};
            m_workQueue.push(scene);
        }
        m_workQueueCond.notify_one();
    }

    std::shared_ptr<sl::Scene> retrieveScene()
    {
        std::unique_lock lock{m_outputQueueMutex};
        while(m_outputQueue.empty())
            m_outputQueueCond.wait(lock);

        auto scene = std::move(m_outputQueue.front());
        m_outputQueue.pop();

        return scene;
    }

    void stop()
    {
        m_shouldQuit = true;
        m_workQueueCond.notify_all();

        for(auto& thread : m_threads)
            thread.join();

        m_threads = {};
    }

private:
    void worker()
    {
        while(!m_shouldQuit)
        {
            std::shared_ptr<sl::Scene> scene;
            {
                std::unique_lock lock{m_workQueueMutex};

                while(m_workQueue.empty() && !m_shouldQuit)
                {
                    m_workQueueCond.wait(lock);
                }

                if(m_shouldQuit)
                    break;

                scene = std::move(m_workQueue.front());
                m_workQueue.pop();
            }

            scene->simulateTableTopScene();

            {
                std::unique_lock lock{m_outputQueueMutex};

                m_outputQueue.push(std::move(scene));
            }
            m_outputQueueCond.notify_all();
        }
    }

    Containers::Array<std::thread> m_threads;
    bool m_shouldQuit = false;

    std::queue<std::shared_ptr<sl::Scene>> m_workQueue;
    std::mutex m_workQueueMutex;
    std::condition_variable m_workQueueCond;

    std::queue<std::shared_ptr<sl::Scene>> m_outputQueue;
    std::mutex m_outputQueueMutex;
    std::condition_variable m_outputQueueCond;
};

PhysicsSim::PhysicsSim(int numThreads)
 : m_d{Containers::InPlaceInit, numThreads}
{
}

PhysicsSim::~PhysicsSim()
{
}

void PhysicsSim::addScene(const std::shared_ptr<sl::Scene>& scene)
{
    m_d->addScene(scene);
}

std::shared_ptr<sl::Scene> PhysicsSim::retrieveScene()
{
    return m_d->retrieveScene();
}

void PhysicsSim::stop()
{
    m_d->stop();
}

}
