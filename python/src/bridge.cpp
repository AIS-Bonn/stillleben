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
#include "py_render_pass.h"
#include "py_image_loader.h"
#include "py_light_map.h"
#include "py_animator.h"

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
    sl::python::ImageLoader::init(m);
    sl::python::LightMap::init(m);
    sl::python::Animator::init(m);
}
