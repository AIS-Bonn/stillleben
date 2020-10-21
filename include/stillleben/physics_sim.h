// Threaded physics simulator
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PHYSICS_SIM_H
#define SL_PHYSICS_SIM_H

#include <Corrade/Containers/Pointer.h>

#include <memory>

namespace sl
{
class Scene;

class PhysicsSim
{
public:
    explicit PhysicsSim(int numThreads = -1);
    ~PhysicsSim();

    PhysicsSim(const PhysicsSim&) = delete;
    PhysicsSim& operator=(const PhysicsSim&) = delete;

    void addScene(const std::shared_ptr<sl::Scene>& scene);
    std::shared_ptr<sl::Scene> retrieveScene();

    void stop();

private:
    class Private;

    Corrade::Containers::Pointer<Private> m_d;
};

}

#endif
