// sl::RenderPass binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "py_render_pass.h"
#include "py_context.h"
#include "py_magnum.h"

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/debug.h>

#include <Magnum/Image.h>

using namespace sl::python::magnum;

namespace
{
    at::Tensor readRGBATensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::RGBA8Unorm, 4, at::kByte);
    }

    at::Tensor readXYZWTensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat);
    }

    at::Tensor readCoordTensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).slice(2, 0, 3);
    }

    at::Tensor readDepthTensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).select(2, 3);
    }

    at::Tensor readShortTensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::R16UI, 1, at::kShort);
    }

    at::Tensor readVertexIndicesTensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::RGBA32UI, 4, at::kInt).slice(2, 0, 3);
    }

    at::Tensor readBaryCentricCoeffsTensor(sl::CUDATexture& texture)
    {
        return extract(texture, Magnum::PixelFormat::RGBA32F, 4, at::kFloat).slice(2, 0, 3);
    }

    at::Tensor renderDebugImage(const std::shared_ptr<sl::Scene>& scene)
    {
        auto texture = sl::renderDebugImage(*scene);
        Magnum::Image2D* img = new Magnum::Image2D{Magnum::PixelFormat::RGBA8Unorm};

        texture.image(*img);

        at::Tensor tensor = torch::from_blob(img->data(),
            {img->size().y(), img->size().x(), 4},
            [=](void*){ delete img; },
            at::kByte
        );

        return tensor;
    }
}

namespace sl
{
namespace python
{
namespace RenderPass
{

void init(py::module& m)
{
    py::class_<sl::RenderPass::Result, ContextSharedPtr<sl::RenderPass::Result>>(m, "RenderPassResult", R"EOS(
            Result of a :ref:`RenderPass` run.

            CUDA support
            ------------

            All accessor methods defined in this class will return a PyTorch
            Tensor residing on the GPU which was used during rendering - as long
            as CUDA support is enabled (see :ref:`init_cuda`).
        )EOS")

        .def(py::init([](){
                if(!sl::python::Context::instance())
                    throw std::logic_error("Call sl::init() first");

                return std::make_shared<sl::RenderPass::Result>(
                    sl::python::Context::cudaEnabled()
                );
            }), R"EOS(
            Constructor.
         )EOS")

        .def("rgb", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readRGBATensor(result->rgb);
            }, R"EOS(
                Read RGBA tensor.

                :return: (H x W x 4) byte tensor with R,G,B,A values.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.
            )EOS")

        .def("class_index", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readShortTensor(result->classIndex);
            }, R"EOS(
                Read class index map (see :ref:`Mesh.class_index`).

                :return: (H x W) short tensor with class values.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. The background index is 0.
            )EOS")

        .def("instance_index", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readShortTensor(result->instanceIndex);
            }, R"EOS(
                Read instance index map (see :ref:`Object.instance_index`).

                :return: (H x W) short tensor with instance values.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering. The background index is 0.
            )EOS")

        .def("coordinates", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readCoordTensor(result->objectCoordinates);
            }, R"EOS(
                Read object coordinates map.

                :return: (H x W x 3) float tensor with coordinates.

                Each pixel specifies the XYZ
                coordinate of the point in the respective object coordinate
                system.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.
            )EOS")

        .def("depth", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readDepthTensor(result->objectCoordinates);
            }, R"EOS(
                Read depth map.

                :return: (H x W) float tensor with depth values.

                Each pixel specifies Z depth in camera frame.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.
            )EOS")

        .def("coordDepth", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readXYZWTensor(result->objectCoordinates);
            }, R"EOS(
                Read combined coordinate + depth map.

                :return: (H x W x 4) float tensor with coordinate and depth
                    values.

                This is the concatenation of the `coordinates` and `depth`
                fields. Using this avoids a copy, since this is the packed
                format used in the shaders.
            )EOS")

        .def("normals", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readXYZWTensor(result->normals);
            }, R"EOS(
                Read normal map.

                :return: (H x W x 4) float tensor with normals.

                Each pixel (XYZW) specifies the normal direction in the camera
                frame (XYZ) and, in the W component, the dot product with the
                camera direction.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.
            )EOS")

        .def("vertex_indices", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readVertexIndicesTensor(result->vertexIndex);
            }, R"EOS(
                Read vertex indices map.

                :return: (H x W x 4) float tensor with vertex indices.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.
            )EOS")

        .def("barycentric_coeffs", [](const std::shared_ptr<sl::RenderPass::Result>& result){
                return readBaryCentricCoeffsTensor(result->barycentricCoeffs);
            }, R"EOS(
                Read barycentric coefficients map.

                :return: (H x W x 3) float tensor with barycentric coefficients.
            )EOS")

        .def("cam_coordinates", [](const ContextSharedPtr<sl::RenderPass::Result>& result){
                return readXYZWTensor(result->camCoordinates);
            }, R"EOS(
                Read dense coordinate map.

                :return: (H x W x 4) float tensor (x, y, z, 1)

                Each pixel contains the coordinates
                of the 3D point in camera space as 4D homogenous coordinates.

                If CUDA support is active, the tensor will reside on the GPU
                which was used during rendering.
            )EOS")
    ;

    py::class_<sl::RenderPass, ContextSharedPtr<sl::RenderPass>>(m, "RenderPass", R"EOS(
            Renders a :ref:`Scene`.
        )EOS")

        .def(py::init([](const std::string& shading){
                if(shading == "pbr")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::PBR, sl::python::Context::cudaEnabled())
                    );
                else if(shading == "phong")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::Phong, sl::python::Context::cudaEnabled())
                    );
                else if(shading == "flat")
                    return ContextSharedPtr<sl::RenderPass>(
                        new sl::RenderPass(sl::RenderPass::Type::Flat, sl::python::Context::cudaEnabled())
                    );
                else
                    throw std::invalid_argument("unknown shading type specified");
            }), R"EOS(
            Constructor.

            :param shading: "pbr", "phong", or "flat". PBR shading falls back to
                Phong shading if no light map is specified in the scene.
         )EOS", py::arg("shading")="pbr")

        .def("render",
            [](const ContextSharedPtr<sl::RenderPass>& pass,
                const std::shared_ptr<sl::Scene>& scene,
                std::shared_ptr<sl::RenderPass::Result> result,
                std::shared_ptr<sl::RenderPass::Result> depthBufferResult){
                return ContextSharedPtr<sl::RenderPass::Result>{pass->render(*scene, result, depthBufferResult.get())};
            }, R"EOS(
            Render a scene.

            :param scene: The scene to render.
            :param result: The caller can pass in a result
                instance to be filled. If this is None, the internal result
                instance of the RenderPass object will be used.
            :param depth_peel: If you want to retrieve the layer behind the last
                rendered one, pass in the result of the previous render here
                (depth peeling).
            :return: A RenderPassResult instance with the resulting data.

            .. block-info :: Internal result instance

                A second render with `result=None` will overwrite
                the results of the first render.
        )EOS", py::arg("scene"), py::arg("result")=nullptr, py::arg("depth_peel")=nullptr)

        .def_property("ssao_enabled", &sl::RenderPass::ssaoEnabled, &sl::RenderPass::setSSAOEnabled, "SSAO enable")
    ;

    m.def("render_debug_image", &::renderDebugImage, R"EOS(
        Render a debug image with object coordinate systems
    )EOS");
}

}
}
}
