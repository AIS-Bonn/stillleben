// Threaded physics simulator
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_JOB_QUEUE_H
#define SL_JOB_QUEUE_H

#include <Corrade/Containers/Pointer.h>

#include <memory>

namespace sl
{
class Scene;

class JobQueue
{
public:
    explicit JobQueue(int numThreads = -1);
    ~JobQueue();

    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    void addScene(const std::shared_ptr<sl::Scene>& scene);
    std::shared_ptr<sl::Scene> retrieveScene();

    void stop();

    std::size_t numThreads() const;

private:
    class Private;

    Corrade::Containers::Pointer<Private> m_d;
};

}

#endif
