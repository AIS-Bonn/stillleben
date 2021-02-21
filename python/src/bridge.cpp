// Python/C++ bridge
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <pybind11/pybind11.h>

#include "py_context.h"
#include "py_magnum.h"
#include "py_mesh.h"
#include "py_object.h"
#include "py_scene.h"
#include "py_render_pass.h"
#include "py_image_loader.h"
#include "py_image_saver.h"
#include "py_light_map.h"
#include "py_animator.h"
#include "py_viewer.h"
#include "py_job_queue.h"
#include "py_manipulation_sim.h"

using namespace sl::python;
using namespace sl::python::magnum;

PYBIND11_MODULE(libstillleben_python, m)
{
    m.doc() = R"EOS(
        The main stillleben module.
    )EOS";

    sl::python::Context::init(m);
    sl::python::magnum::init(m);
    sl::python::Mesh::init(m);
    sl::python::Object::init(m);
    sl::python::LightMap::init(m);
    sl::python::Scene::init(m);
    sl::python::RenderPass::init(m);
    sl::python::ImageLoader::init(m);
    sl::python::ImageSaver::init(m);
    sl::python::Animator::init(m);
    sl::python::Viewer::init(m);
    sl::python::JobQueue::init(m);
    sl::python::ManipulationSim::init(m);
}
