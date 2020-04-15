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

            Args:
                mesh (Mesh): Mesh to instantiate
                options (dict): Dictionary of options. Supported keys:
                    * color (tensor): RGBA color used if no color information is
                      present in the mesh. Defaults to white.
                    * force_color (bool): If true, the color specified in
                      `color` is used even if the mesh is colored.
        )EOS", py::arg("mesh"), py::arg("options")=py::dict())

        .def("pose", wrapShared(&sl::Object::pose), R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.

            Examples:
                >>> obj = Object(Mesh("mesh.gltf"))
                >>> obj.pose()
        )EOS")
        .def("set_pose", wrapShared(&sl::Object::setPose), R"EOS(
            Pose matrix. This 4x4 matrix transforms object points to global
            points.

            Note: This is implemented as separate getter/setter methods since
            in-place operations on the returned pose do not work.

            Examples:
                >>> obj = Object(Mesh("mesh.gltf"))
                >>> p = obj.pose()
                >>> p[:3,3] = torch.tensor([0, 1, 0])
        )EOS", py::arg("pose"))

        .def_property("instance_index", &sl::Object::instanceIndex, &sl::Object::setInstanceIndex, R"EOS(
            Instance index for training semantic segmentation. This is
            automatically set by :func:`Scene.addObject` but can also be
            set manually. A manual assignment always takes precedence.
        )EOS")

        .def_property_readonly("mesh", &sl::Object::mesh, R"EOS(
            The associated :class:`Mesh` instance.
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

        .def_property("metalness", &sl::Object::metalness, &sl::Object::setMetalness, R"EOS(
            Metalness parameter for PBR shading (0-1).
        )EOS")

        .def_property("roughness", &sl::Object::roughness, &sl::Object::setRoughness, R"EOS(
            Roughness parameter for PBR shading (0-1).
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
    ;
}

}
}
}
