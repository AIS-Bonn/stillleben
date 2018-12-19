// Represents a scene composed of multiple objects
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/scene.h>

#include <stillleben/object.h>

#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

Scene::Scene(const std::shared_ptr<Context>& ctx, const ViewportSize& viewportSize)
 : m_ctx{ctx}
{
    // Every scene needs a camera
    const Rad FOV_X = Deg(58.0);
    auto projection = Matrix4::perspectiveProjection(FOV_X, 1.0f, 0.01f, 1000.0f);
    m_cameraObject.setParent(&m_scene);
    (*(m_camera = new SceneGraph::Camera3D{m_cameraObject}))
        .setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(projection)
        .setViewport(viewportSize);
}

Scene::~Scene()
{
    // The SceneObject destructor of this instance will delete child objects,
    // but they are reference counted using shared_ptr => first release them
    for(auto& obj : m_objects)
    {
        obj->setParentSceneObject(nullptr);
    }
}

void Scene::setCameraPose(const PoseMatrix& pose)
{
    m_cameraObject.setTransformation(pose);
}

PoseMatrix Scene::cameraPose() const
{
    return m_cameraObject.absoluteTransformationMatrix();
}

void Scene::setCameraIntrinsics(float fx, float fy, float cx, float cy)
{
    // Source: https://blog.noctua-software.com/opencv-opengl-projection-matrix.html

    // far and near
    constexpr float f = 10.0f;
    constexpr float n = 0.01;

    const float H = m_camera->viewport().y();
    const float W = m_camera->viewport().x();

    // Caution, this is column-major
    Matrix4 P{
        {2.0f*fx/cx, 0.0f, 0.0f, 0.0f},
        {0.0f, -2.0f*fy/cy, 0.0f, 0.0f},
        {1.0f - 2.0f*cx/W, 2.0f*cy/H - 1.0f, (f+n)/(n-f), -1.0f},
        {0.0f, 0.0f, 2.0f*f*n/(n-f), 0.0f}
    };

    m_camera->setProjectionMatrix(P);
}

void Scene::addObject(const std::shared_ptr<Object>& obj)
{
    m_objects.push_back(obj);

    obj->setParentSceneObject(&m_scene);

    // Automatically set the instance index if not set by the user already
    if(obj->instanceIndex() == 0)
        obj->setInstanceIndex(m_objects.size());
}

float Scene::minimumDistanceForObjectDiameter(float diameter) const
{
    auto P = m_camera->projectionMatrix();

    // for perspective projection:
    // P[0][0] = 1.0 / std::tan(alpha)
    // NOTE: alpha is the half horizontal view angle.

    return std::max(
        P[0][0] * diameter / 2.0,
        P[1][1] * diameter / 2.0
    );
}

}
