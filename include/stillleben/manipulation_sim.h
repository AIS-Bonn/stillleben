// Simulate a manipulator interacting with the scene
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_MANIPULATION_SIM_H
#define SL_MANIPULATION_SIM_H

#include <memory>

#include <Corrade/Containers/Pointer.h>
#include <Magnum/Magnum.h>

namespace sl
{

class Scene;
class Object;

class ManipulationSim
{
public:
    explicit ManipulationSim(const std::shared_ptr<sl::Scene>& scene, const std::shared_ptr<sl::Object>& manipulator, const Magnum::Matrix4& initialPose);
    ~ManipulationSim();
    
    ManipulationSim(const ManipulationSim&) = delete;
    ManipulationSim& operator=(const ManipulationSim&) = delete;
    
    void setSpringParameters(float stiffness, float damping, float forceLimit);

    void lockRotationAxes(bool x, bool y, bool z);

    void step(const Magnum::Matrix4& goalPose, float dt);
private:
    class Private;
    Corrade::Containers::Pointer<Private> m_d;
};

}

#endif
