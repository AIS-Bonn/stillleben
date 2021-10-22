// sl::Object binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_object.h"
#include "py_context.h"
#include "py_magnum.h"

#include <stillleben/mesh.h>
#include <stillleben/object.h>

using namespace sl::python::magnum;


namespace sl
{
namespace python
{
namespace Object
{

void init(py::module& m)
{
    py::class_<sl::Object, std::shared_ptr<sl::Object>>(m, "Object", R"EOS(
        An instantiated mesh with associated pose and other instance
        properties.
    )EOS")

        .def(py::init([](const std::shared_ptr<sl::Mesh>& mesh, const py::dict& options){
                sl::InstantiationOptions opts;
                for(const auto& entry : options)
                {
                    auto key = entry.first.cast<std::string>();
                    if(key == "color")
                        opts.color = fromTorch<Magnum::Color4>::convert(entry.second.cast<torch::Tensor>());
                    else if(key == "force_color")
                        opts.forceColor = entry.second.cast<bool>();
                    else
                        throw std::invalid_argument("Invalid key in options");
                }

                auto obj = std::make_shared<sl::Object>();
                obj->setMesh(mesh);
                obj->setInstantiationOptions(opts);

                return obj;
            }), R"EOS(
            Constructor

            :param mesh: Mesh to instantiate
            :param options: Dictionary of instantiation options (see below)

            .. block-info:: Instantiation options

                :color: RGBA color used if no color information is
                  present in the mesh. Defaults to white.
                :force_color: If true, the color specified in
                  `color` is used even if the mesh is colored.
        )EOS", py::arg("mesh"), py::arg("options")=py::dict())

        .def("pose", wrapShared(&sl::Object::pose), R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.
        )EOS")
        .def("set_pose", wrapShared(&sl::Object::setPose), R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.

            .. code:: python

                obj = Object(Mesh("mesh.gltf"))
                p = obj.pose()
                p[:3,3] = torch.tensor([0, 1, 0])
                obj.set_pose(p)

            .. block-info:: Supported poses

                The stillleben renderer supports arbitrary 4x4 pose matrices.
                However, if you use the physics engine (e.g. via
                :ref:`Scene.simulate`), you are restricted to proper rigid transforms,
                i.e. pure rotations and translations.
        )EOS", py::arg("pose"))

        .def_property("instance_index", &sl::Object::instanceIndex, &sl::Object::setInstanceIndex, R"EOS(
            Instance index for training semantic segmentation. This is
            automatically set by :ref:`Scene.add_object` but can also be
            set manually. A manual assignment always takes precedence.
        )EOS")

        .def_property_readonly("mesh", &sl::Object::mesh, R"EOS(
            The associated :ref:`Mesh` instance.
        )EOS")

        .def_property("specular_color", wrapShared(&sl::Object::specularColor), wrapShared(&sl::Object::setSpecularColor), R"EOS(
            Specular color for Phong shading. This is multiplied with the
            calculated specular intensity. Set to white for a fully-specular
            object, set to black for a fully diffuse object.
        )EOS")

        .def_property("shininess", &sl::Object::shininess, &sl::Object::setShininess, R"EOS(
            Shininess parameter for Phong shading. Low values result in very
            spread out specular highlights, high values in very small, sharp
            highlights. Default value is 80 (pretty sharp).
        )EOS")

        .def_property("metallic", &sl::Object::metallic, &sl::Object::setMetallic, R"EOS(
            Metallic parameter for PBR shading (0-1).

            A negative value (default) means that the metallic parameter of the
            mesh is used.
        )EOS")

        .def_property("roughness", &sl::Object::roughness, &sl::Object::setRoughness, R"EOS(
            Roughness parameter for PBR shading (0-1).

            A negative value (default) means that the roughness parameter of the
            mesh is used.
        )EOS")


        .def_property("sticker_rotation", wrapShared(&sl::Object::stickerRotation), wrapShared(&sl::Object::setStickerRotation), R"EOS(
            Sticker rotation quaternion.
        )EOS")

        .def_property("sticker_range", wrapShared(&sl::Object::stickerRange), wrapShared(&sl::Object::setStickerRange), R"EOS(
            Sticker range.
        )EOS")

        .def_property("sticker_texture", &sl::Object::stickerTexture, &sl::Object::setStickerTexture, R"EOS(
            Sticker texture.
        )EOS")


        .def_property("static", &sl::Object::isStatic, &sl::Object::setStatic, R"EOS(
            If set to true, the object is marked as static for physics simulation. This is a good choice
            for containers or supporting planes.
        )EOS")

        .def_property_readonly("separation", &sl::Object::separation, R"EOS(
            After physics simulation, this contains the minimum separation of
            this object to other objects in the scene. If the separation is
            negative, the two objects intersect.
        )EOS")

        .def_property("mass", &sl::Object::mass, &sl::Object::setMass, R"EOS(
            Mass of the object in kg.
        )EOS")
        .def_property_readonly("volume", &sl::Object::volume, R"EOS(
            Volume of the object in m^3.
        )EOS")
        .def_property("density", &sl::Object::density, &sl::Object::setDensity, R"EOS(
            Density of the object in kg / m^3.
        )EOS")

        .def_property("linear_velocity_limit", &sl::Object::linearVelocityLimit, &sl::Object::setLinearVelocityLimit, R"EOS(
            Linear velocity limit in m/s.
        )EOS")

        .def_property("linear_velocity", wrapShared(&sl::Object::linearVelocity), wrapShared(&sl::Object::setLinearVelocity), R"EOS(
            Linear velocity in m/s (in global frame).
        )EOS")

        .def_property("angular_velocity", wrapShared(&sl::Object::angularVelocity), wrapShared(&sl::Object::setAngularVelocity), R"EOS(
            Angular velocity in rad/s (in global frame).
        )EOS")

        .def_property("static_friction", &sl::Object::staticFriction, &sl::Object::setStaticFriction, R"EOS(
            Static friction coefficient (should be in range 0-1).
        )EOS")
        .def_property("dynamic_friction", &sl::Object::dynamicFriction, &sl::Object::setDynamicFriction, R"EOS(
            Dynamic friction coefficient (should be in range 0-1).
        )EOS")
        .def_property("restitution", &sl::Object::restitution, &sl::Object::setRestitution, R"EOS(
            Coefficient of restitution.
            A coefficient of 0 makes the object bounce as little as possible, higher values up to 1.0 result in more bounce.
        )EOS")
    ;
}

}
}
}
