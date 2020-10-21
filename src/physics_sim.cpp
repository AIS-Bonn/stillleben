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

namespace
{
    struct Job
    {
        std::shared_ptr<sl::Scene> scene;
        bool done = false;
        bool taken = false;
    };
}

class PhysicsSim::Private
{
public:
    Private(int numThreads)
    {
        if(numThreads == -1)
        {
            numThreads = std::thread::hardware_concurrency() / 2;
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
            m_workQueue.push_back({scene});
            m_freeJobs++;
        }
        m_workQueueCond.notify_one();
    }

    std::shared_ptr<sl::Scene> retrieveScene()
    {
        std::unique_lock lock{m_workQueueMutex};

        if(m_workQueue.empty())
            throw std::logic_error{"PhysicsSim::retrieveScene(): No scenes in work queue. You need to add scenes first!"};

        while(!m_workQueue.front().done)
            m_workQueueCond.wait(lock);

        auto job = std::move(m_workQueue.front());
        m_workQueue.pop_front();

        return job.scene;
    }

    void stop()
    {
        m_shouldQuit = true;
        m_workQueueCond.notify_all();

        for(auto& thread : m_threads)
            thread.join();

        m_threads = {};
    }

    std::size_t numThreads() const
    { return m_threads.size(); }

private:
    void worker()
    {
        while(!m_shouldQuit)
        {
            Job* job{};
            {
                std::unique_lock lock{m_workQueueMutex};

                while(m_freeJobs == 0 && !m_shouldQuit)
                {
                    m_workQueueCond.wait(lock);
                }

                if(m_shouldQuit)
                    break;

                auto it = std::find_if(m_workQueue.begin(), m_workQueue.end(), [](auto& job){
                    return !job.done && !job.taken;
                });
                CORRADE_INTERNAL_ASSERT(it != m_workQueue.end());

                job = &*it;
                job->taken = true;
                m_freeJobs--;
            }

            job->scene->simulateTableTopScene();
            job->done = true;

            {
                std::unique_lock lock{m_workQueueMutex};

                if(job == &m_workQueue.front())
                    m_workQueueCond.notify_all();
            }
        }
    }

    Containers::Array<std::thread> m_threads;
    bool m_shouldQuit = false;

    std::deque<Job> m_workQueue;
    std::mutex m_workQueueMutex;
    std::condition_variable m_workQueueCond;
    unsigned int m_freeJobs = 0;
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

std::size_t PhysicsSim::numThreads() const
{
    return m_d->numThreads();
}


}
