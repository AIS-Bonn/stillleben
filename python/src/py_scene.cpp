// sl::Scene binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_scene.h"
#include "py_context.h"
#include "py_magnum.h"

#include <Corrade/Utility/Configuration.h>

#include <stillleben/light_map.h>
#include <stillleben/mesh.h>
#include <stillleben/mesh_cache.h>
#include <stillleben/scene.h>

#include <pybind11/functional.h>

using namespace sl::python::magnum;

namespace
{
    std::shared_ptr<sl::Scene> Scene_factory(const std::tuple<int, int>& viewportSize)
    {
        return std::make_shared<sl::Scene>(sl::python::Context::instance(), sl::ViewportSize{
            std::get<0>(viewportSize),
            std::get<1>(viewportSize)
        });
    }

    std::tuple<int, int> Scene_viewport(const std::shared_ptr<sl::Scene>& scene)
    {
        auto vp = scene->viewport();
        return std::make_tuple(vp.x(), vp.y());
    }
}

namespace sl
{
namespace python
{
namespace Scene
{

void init(py::module& m)
{
    py::class_<sl::Scene, std::shared_ptr<sl::Scene>>(m, "Scene", R"EOS(
        Represents a scene with multiple objects.

        Typical usage
        -------------

        .. code:: python

            import stillleben as sl
            sl.init()

            # Load & instantiate meshes
            mesh = sl.Mesh('my_mesh.gltf')
            objectA = sl.Object(mesh)
            objectB = sl.Object(mesh)

            # Setup the scene
            scene = sl.Scene((1920,1080))
            scene.add_object(objectA)
            scene.add_object(objectB)
    )EOS")

        .def(py::init(&Scene_factory), R"EOS(
            Constructor

            :param viewport_size: Size of the rendered image (W,H)
        )EOS", py::arg("viewport_size"))

        .def("camera_pose", wrapShared(&sl::Scene::cameraPose), R"EOS(
            Retrieve current camera pose (see :ref:`set_camera_pose`).
        )EOS")
        .def("set_camera_pose", wrapShared(&sl::Scene::setCameraPose), R"EOS(
            Set the camera pose within the scene.

            :param pose: 4x4 matrix transforming camera coordinates to
                global coordinates.
        )EOS", py::arg("pose"))
        .def("set_camera_look_at", wrapShared(&sl::Scene::setCameraLookAt), R"EOS(
            Sets the camera pose within the scene using lookAt parameters.

            :param position: 3D position vector
            :param look_at: 3D lookAt vector
            :param up: 3D up vector (defaults to Z axis)
        )EOS", py::arg("position"), py::arg("look_at"), py::arg("up")=torch::tensor({0.0, 0.0, 1.0}))

        .def("set_camera_intrinsics", &sl::Scene::setCameraIntrinsics, R"EOS(
            Set camera intrinsics directly.

            :param fx: :math:`f_x`
            :param fy: :math:`f_y`
            :param cx: :math:`c_x`
            :param cy: :math:`c_y`

            Set the camera intrinsics assuming a pinhole camera with focal
            lengths :math:`f_x`, :math:`f_y`, and projection center :math:`p_x`, :math:`p_y`.
        )EOS", py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))

        .def("set_camera_hfov", [](const std::shared_ptr<sl::Scene>& scene, float hfov){
            scene->setCameraFromFOV(Magnum::Rad{hfov});
        }, R"EOS(
            Set camera intrinsics from horizontal FOV.

            :param hfov: Horizontal FOV in radians

            This assumes a pinhole camera with centered principal point
            and horizontal FOV :p:`hfov`. The vertical FOV is set such that
            the pixel size is 1:1 (in other words, :math:`f_x=f_y`).
        )EOS", py::arg("hfov"))

        .def("set_camera_projection", wrapShared(&sl::Scene::setCameraProjection), R"EOS(
            Set the camera intrinsics from a 4x4 matrix.

            :param P: The projection matrix.
        )EOS", py::arg("P"))

        .def("projection_matrix", wrapShared(&sl::Scene::projectionMatrix), R"EOS(
                Return the currently used OpenGL projection matrix.
            )EOS")

        .def_property_readonly("viewport", &Scene_viewport, R"EOS(
            The current viewport size (W,H) as set in the constructor.
        )EOS")

        .def_property("background_image",
            &sl::Scene::backgroundImage, &sl::Scene::setBackgroundImage, R"EOS(
            The background image. If None (default), the background color
            (see :ref:`background_color`) is used.
        )EOS")

        .def_property("background_color",
            wrapShared(&sl::Scene::backgroundColor), wrapShared(&sl::Scene::setBackgroundColor), R"EOS(
            The background color (RGBA, float, range 0-1). The default is white.
        )EOS")

        .def("min_dist_for_object_diameter", &sl::Scene::minimumDistanceForObjectDiameter, R"EOS(
            Calculates the minimum Z distance from the camera to have an object
            of diameter :p:`diameter` fully visible in the camera frustrum.

            :param diameter: Diameter of the object.
        )EOS", py::arg("diameter"))

        .def("place_object_randomly", wrapShared(&sl::Scene::placeObjectRandomly), R"EOS(
                Generates a random pose for an object of given diameter.

                :param diameter: Object diameter
                :param min_size_factor: The object will occupy at least this
                    much of the screen space.
                :return: Object pose (4x4 tensor)

                The pose obeys the following constraints (relative to the camera
                coordinate system):

                * :math:`z` is between :code:`1.2*min_dist_for_object_diameter()`
                  and :code:`(1.0/min_size_factor) * min_dist_for_object_diameter()`, and
                * :math:`x` and :math:`y` are choosen such that the object center is
                  inside 80% of the camera frustrum in each axis.
            )EOS",
            py::arg("diameter"),
            py::arg("min_size_factor")=sl::pose::DEFAULT_MIN_SIZE_FACTOR
        )

        .def("camera_to_world", wrapShared(&sl::Scene::cameraToWorld), R"EOS(
            Transform a pose from camera coordinates to world coordinates.

            :param poseInCamera: 4x4 float pose
        )EOS", py::arg("poseInCamera"))

        .def("add_object", &sl::Scene::addObject, R"EOS(
            Adds an object to the scene.

            :param object: Object to be added
        )EOS", py::arg("object"))

        .def_property_readonly("objects", &sl::Scene::objects, R"EOS(
            All objects added to the scene.

            This is a list of :ref:`Object`. See :ref:`add_object`.
        )EOS")

        .def("find_noncolliding_pose", [](
            const std::shared_ptr<sl::Scene>& scene,
            const std::shared_ptr<sl::Object>& object,
            const std::string& sampler, int max_iterations,
            py::kwargs kwargs){
                if(sampler == "random")
                {
                    sl::pose::RandomPositionSampler posSampler{
                        scene->projectionMatrix(),
                        object->mesh()->bbox().size().length()
                    };
                    sl::pose::RandomPoseSampler sampler{posSampler};
                    return scene->findNonCollidingPose(*object, sampler, max_iterations);
                }
                else if(sampler == "viewpoint")
                {
                    if(!kwargs.contains("viewpoint"))
                        throw std::invalid_argument{"sampler='viewpoint' needs viewpoint argument"};

                    auto viewPoint = fromTorch<Magnum::Vector3>::convert(
                        kwargs["viewpoint"].cast<at::Tensor>()
                    );

                    sl::pose::RandomPositionSampler posSampler{
                        scene->projectionMatrix(),
                        object->mesh()->bbox().size().length()
                    };
                    sl::pose::ViewPointPoseSampler sampler{posSampler};
                    sampler.setViewPoint(viewPoint);

                    return scene->findNonCollidingPose(*object, sampler, max_iterations);
                }
                else if(sampler == "view_corrected")
                {
                    if(!kwargs.contains("orientation"))
                        throw std::invalid_argument{"sampler='view_corrected' needs orientation argument"};

                    auto orientation = fromTorch<Magnum::Matrix3>::convert(
                        kwargs["orientation"].cast<at::Tensor>()
                    );

                    sl::pose::RandomPositionSampler posSampler{
                        scene->projectionMatrix(),
                        object->mesh()->bbox().size().length()
                    };
                    sl::pose::ViewCorrectedPoseSampler sampler{posSampler, orientation};

                    return scene->findNonCollidingPose(*object, sampler, max_iterations);
                }
                else
                    throw std::invalid_argument{"Unknown sampler"};
            }, R"EOS(
            Finds a non-colliding random pose for an object. The object should
            already have been added using add_object().

            :param object: The object to place. It should already be added to
                the scene.
            :param sampler: "random" for fully random pose, "viewpoint"
                for a pose that ensures we look from a certain viewpoint
                onto the object, or "view_corrected" for a perspective-
                corrected constant orientation.
            :param max_iterations: Maximum number of attempts
            :param viewpoint: 3D view point for "viewpoint" sampler
            :param orientation: 3x3 orientation matrix for "view_corrected"
                sampler
            :return: True if a non-colliding pose was found.
        )EOS", py::arg("object"), py::arg("sampler") = "random", py::arg("max_iterations")=10)

        .def_property("light_position",
            wrapShared(&sl::Scene::lightPosition),
            wrapShared(&sl::Scene::setLightPosition),
        R"EOS(
            The light position in world coordinates. This is a float tensor
            of size 3.
        )EOS")

        .def_property("ambient_light",
            wrapShared(&sl::Scene::ambientLight),
            wrapShared(&sl::Scene::setAmbientLight),
        R"EOS(
            The color & intensity of the ambient light. This is a float
            tensor of size 3 (RGB, range 0-1). This color is multiplied
            with the object color / texture during rendering.
        )EOS")

        .def("simulate_tabletop_scene", &sl::Scene::simulateTableTopScene, R"EOS(
            Arrange the objects as if they were standing on a supporting surface.
            This also calls :ref:`choose_random_camera_pose`.

            :param vis_cb: This callback is called after each physics simulation
                timestep.
        )EOS", py::arg("vis_cb")=std::function<void()>{})

        .def("choose_random_camera_pose", &sl::Scene::chooseRandomCameraPose, R"EOS(
            Choose a random camera pose in the upper hemisphere (Z up) under the constraint
            that all objects in the scene should be within the camera frustum.
        )EOS")

        .def("choose_random_light_position", &sl::Scene::chooseRandomLightPosition, R"EOS(
            Choose a random light position.

            The position obeys the following constraints:

            * The light comes from above (negative Y direction)
            * The light never comes from behind the objects.
        )EOS")


        .def("serialize",
            [](const std::shared_ptr<sl::Scene>& scene){
                std::ostringstream ss;
                Corrade::Utility::Configuration config;

                scene->serialize(config);
                config.save(ss);

                return ss.str();
            }, R"EOS(
                Serialize the scene to a string
            )EOS")

        .def("deserialize",
            [](const std::shared_ptr<sl::Scene>& scene, const std::string& str, sl::MeshCache* cache){
                std::istringstream ss{str};
                Corrade::Utility::Configuration config{ss};

                scene->deserialize(config, cache);
            }, R"EOS(
                Deserialize the scene from a string
            )EOS", py::arg("str"), py::arg("cache")=nullptr)

        .def("load_visual", &sl::Scene::loadVisual, R"EOS(
                Load visual meshes
            )EOS")
        .def("load_physics", &sl::Scene::loadPhysics, R"EOS(
                Load physics meshes
            )EOS")


        .def_property("light_map", &sl::Scene::lightMap, &sl::Scene::setLightMap, R"EOS(
                Light map used for image-based lighting.
            )EOS")

        .def_property("background_plane_pose", wrapShared(&sl::Scene::backgroundPlanePose), wrapShared(&sl::Scene::setBackgroundPlanePose),
            "Pose of the background plane (plane normal is in +Z direction)")
        .def_property("background_plane_size", wrapShared(&sl::Scene::backgroundPlaneSize), wrapShared(&sl::Scene::setBackgroundPlaneSize),
            "Size of the background plane in local X/Y directions")
        .def_property("background_plane_texture", &sl::Scene::backgroundPlaneTexture, &sl::Scene::setBackgroundPlaneTexture,
            "Texture of the background plane")
    ;
}

}
}
}
