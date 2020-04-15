// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <stillleben/context.h>
#include <stillleben/mesh.h>
#include <stillleben/object.h>
#include <stillleben/scene.h>
#include <stillleben/render_pass.h>
#include <stillleben/debug.h>
#include <stillleben/image_loader.h>
#include <stillleben/cuda_interop.h>
#include <stillleben/animator.h>
#include <stillleben/light_map.h>
#include <stillleben/mesh_cache.h>
#include <stillleben/contrib/ctpl_stl.h>

#include <Corrade/Utility/Configuration.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Containers/ArrayView.h>

#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Magnum.h>

#include <future>
#include <memory>
#include <functional>

#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "py_context.h"
#include "py_magnum.h"
#include "py_mesh.h"
#include "py_object.h"
#include "py_scene.h"
#include "py_renderpass.h"

using namespace sl::python;
using namespace sl::python::magnum;

PYBIND11_MODULE(libstillleben_python, m)
{
    sl::python::Context::init(m);
    sl::python::magnum::init(m);
    sl::python::Mesh::init(m);
    sl::python::Object::init(m);
    sl::python::Scene::init(m);
    sl::python::RenderPass::init(m);

    py::class_<sl::LightMap, std::shared_ptr<sl::LightMap>>(m, "LightMap", R"EOS(
            An .ibl light map for image-based lighting.
        )EOS")

        .def(py::init(), "Constructor")

        .def(py::init([](const std::string& path){
                return std::make_shared<sl::LightMap>(path, sl::python::Context::instance());
            }),
            R"EOS(
                Constructs and calls load().
            )EOS"
        )

        .def("load", &sl::LightMap::load, R"EOS(
            Opens an .ibl file.

            Args:
                path (str): Path to .ibl file
            Returns:
                bool: True if successful
        )EOS")
    ;

    py::class_<sl::Animator>(m, "Animator", R"EOS(
            Generates interpolated object poses.
        )EOS")

        .def(py::init([](const std::vector<at::Tensor>& poses, unsigned int ticks){
            std::vector<Magnum::Matrix4> mPoses;
            for(auto& p : poses)
                mPoses.push_back(fromTorch<Magnum::Matrix4>::convert(p));
            return std::make_unique<sl::Animator>(mPoses, ticks);
        }), "Constructor", py::arg("poses"), py::arg("ticks"))

        .def("__iter__", [](py::object s) { return s; })

        .def("__next__", [](sl::Animator& s){
            if(s.currentTick() >= s.totalTicks())
                throw py::stop_iteration{};

            return toTorch<Magnum::Matrix4>::convert(s());
        })

        .def("__len__", [](sl::Animator& s){ return s.totalTicks(); })
    ;

    py::class_<sl::MeshCache>(m, "MeshCache", R"EOS(
            Caches Mesh instances.
        )EOS")

        .def(py::init([](){ return new sl::MeshCache(sl::python::Context::instance()); }))

        .def("add", &sl::MeshCache::add, R"EOS(
            Add a list of meshes to the cache.

            Args:
                meshes (list): list of :class:`Mesh` instances
        )EOS", py::arg("meshes"))
    ;

    py::class_<sl::ImageLoader>(m, "ImageLoader", R"EOS(
            Multi-threaded image loader.
        )EOS")

        .def(py::init([](const std::string& path){
                return new sl::ImageLoader(path, sl::python::Context::instance());
            }), R"EOS(
            Constructor.

            Args:
                path: Path to the image directory
            )EOS", py::arg("path")
        )

        .def("next", &sl::ImageLoader::nextRectangleTexture, R"EOS(
            Return next image (randomly sampled). This is the same as nextRectangleTexture().
        )EOS")

        .def("next_texture2d", &sl::ImageLoader::nextTexture2D, R"EOS(
            Return next image (randomly sampled) as 2D texture
        )EOS")

        .def("next_rectangle_texture", &sl::ImageLoader::nextRectangleTexture, R"EOS(
            Return next image (randomly sampled) as rectangle texture
        )EOS")
    ;
}
