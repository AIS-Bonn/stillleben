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
        )EOS")

        .def(py::init(&Scene_factory), R"EOS(
            Constructor

            Args:
                viewport_size (int,int): Size of the rendered image (W,H)
        )EOS", py::arg("viewport_size"))

        .def("camera_pose", wrapShared(&sl::Scene::cameraPose), R"EOS(
            Retrieve current camera pose (see :func:`setCameraPose`).
        )EOS")
        .def("set_camera_pose", wrapShared(&sl::Scene::setCameraPose), R"EOS(
            Set the camera pose within the scene. For most applications, leaving
            this at identity is a good idea - that way your object poses are
            expressed in camera coordinates.

            Args:
                pose (tensor): 4x4 matrix transforming camera coordinates to
                    global coordinates.
        )EOS", py::arg("pose"))
        .def("set_camera_look_at", wrapShared(&sl::Scene::setCameraLookAt), R"EOS(
            Sets the camera pose within the scene using lookAt parameters.

            Args:
                position (tensor): 3D position vector
                look_at (tensor): 3D lookAt vector
                up (tensor): 3D up vector (defaults to Z axis)
        )EOS", py::arg("position"), py::arg("look_at"), py::arg("up")=torch::tensor({0.0, 0.0, 1.0}))

        .def("set_camera_intrinsics", &sl::Scene::setCameraIntrinsics, R"EOS(
            Set the camera intrinsics assuming a pinhole camera with focal
            lengths :math:`f_x`, :math:`f_y`, and projection center :math:`p_x`, :math:`p_y`.

            Note: Magnum may slightly modify the resulting matrix, I have not
            checked the accuracy of this method.

            Args:
                fx (float): :math:`f_x`
                fy (float): :math:`f_y`
                cx (float): :math:`c_x`
                cy (float): :math:`c_y`
        )EOS", py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))

        .def("set_camera_projection", wrapShared(&sl::Scene::setCameraProjection), R"EOS(
            Set the camera intrinsics from a 4x4 matrix.

            Args:
                P (tensor): The matrix.
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
            (see `background_color`) is used.
        )EOS")

        .def_property("background_color",
            wrapShared(&sl::Scene::backgroundColor), wrapShared(&sl::Scene::setBackgroundColor), R"EOS(
            The background color (RGBA, float, range 0-1). The default is white.
        )EOS")

        .def("min_dist_for_object_diameter", &sl::Scene::minimumDistanceForObjectDiameter, R"EOS(
            Calculates the minimum Z distance from the camera to have an object
            of diameter :attr:`diameter` fully visible in the camera frustrum.

            Args:
                diameter (float): Diameter of the object.
        )EOS", py::arg("diameter"))

        .def("place_object_randomly", wrapShared(&sl::Scene::placeObjectRandomly), R"EOS(
                Generates a random pose for an object of given diameter.

                The pose obeys the following constraints (relative to the camera
                coordinate system):
                * :math:`z` is between `0.8*min_dist_for_object_diameter()` and
                `2.0*min_dist_for_object_diameter()`, and
                * :math:`x` and :math:`y` are choosen such that the object center is
                inside 80% of the camera frustrum in each axis.
            )EOS",
            py::arg("diameter"),
            py::arg("min_size_factor")=sl::pose::DEFAULT_MIN_SIZE_FACTOR
        )

        .def("camera_to_world", wrapShared(&sl::Scene::cameraToWorld), R"EOS(
                Transform a pose from camera coordinates to world coordinates.

                Args:
                    poseInCamera (tensor): 4x4 float pose
            )EOS", py::arg("poseInCamera")
        )

        .def("add_object", &sl::Scene::addObject, R"EOS(
            Adds an object to the scene.

            Args:
                object (Object): Object to be added.
        )EOS", py::arg("object"))

        .def_property_readonly("objects", &sl::Scene::objects, R"EOS(
            Contains all objects added to the scene. See add_object()

            Returns:
                list: List of sl::Object
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

            Args:
                object (stillleben.object): The object to place
                sampler (str): "random" for fully random pose, "viewpoint"
                    for a pose that ensures we look from a certain viewpoint
                    onto the object, or "view_corrected" for a perspective-
                    corrected constant orientation.
                max_iterations (int): Maximum number of attempts
                viewpoint (tensor): 3D view point for "viewpoint" sampler
                orientation (tensor): 3x3 orientation matrix for
                    "view_corrected" sampler

            Returns:
                bool: True if a non-colliding pose was found.
        )EOS", py::arg("object"), py::arg("sampler") = "random", py::arg("max_iterations")=10)

        .def("resolve_collisions", &sl::Scene::resolveCollisions, R"EOS(
            Resolve collisions by forward-simulation using the physics engine.
        )EOS")

        .def_property("light_position",
            wrapShared(&sl::Scene::lightPosition),
            wrapShared(&sl::Scene::setLightPosition),
            R"EOS(
                The light position in world coordinates. This is a float tensor
                of size 3.
            )EOS"
        )

        .def_property("ambient_light",
            wrapShared(&sl::Scene::ambientLight),
            wrapShared(&sl::Scene::setAmbientLight),
            R"EOS(
                The color & intensity of the ambient light. This is a float
                tensor of size 3 (RGB, range 0-1). This color is multiplied
                with the object color / texture during rendering.
            )EOS"
        )

        .def("simulate_tabletop_scene", &sl::Scene::simulateTableTopScene, R"EOS(
            Arrange the objects as if they were standing on a supporting surface.
        )EOS", py::arg("vis_cb")=std::function<void()>{})

        .def("choose_random_light_position", &sl::Scene::chooseRandomLightPosition, R"EOS(
            Choose a random light position under the following constraints:

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
