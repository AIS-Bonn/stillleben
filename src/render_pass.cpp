// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/mesh.h>
#include <stillleben/cuda_interop.h>
#include <stillleben/context.h>
#include <stillleben/light_map.h>

#include <Corrade/Utility/DebugStl.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/DebugOutput.h>
#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>

#include <Magnum/MeshTools/Compile.h>

#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/Primitives/UVSphere.h>

#include <Magnum/PixelFormat.h>

#include "shaders/render_shader.h"
#include "shaders/background_shader.h"
#include "shaders/background_cube_shader.h"
#include "shaders/ssao_shader.h"
#include "shaders/ssao_apply_shader.h"
#include "shaders/tone_map_shader.h"

using namespace Magnum;
using namespace Math::Literals;

namespace sl
{

namespace
{
constexpr RenderShader::Flags flagsForType(RenderPass::Type type)
{
    switch(type)
    {
        case RenderPass::Type::Flat:
            return RenderShader::Flag::Flat;
        case RenderPass::Type::Phong:
        case RenderPass::Type::PBR:
            return {};
    }
    return {};
}

}

RenderPass::Result::Result(bool cuda)
 : m_mapper{cuda}
 , rgb{m_mapper}
 , objectCoordinates{m_mapper}
 , classIndex{m_mapper}
 , instanceIndex{m_mapper}
 , normals{m_mapper}
 , vertexIndex{m_mapper}
 , barycentricCoeffs{m_mapper}
 , camCoordinates{m_mapper}
{
}

RenderPass::Result::~Result() = default;


void RenderPass::Result::mapCUDA()
{
#if HAVE_CUDA
    m_mapper.mapAll();
#endif

    m_mapped = true;
}

void RenderPass::Result::unmapCUDA()
{
#if HAVE_CUDA
    m_mapper.unmapAll();
#endif

    m_mapped = false;
}


RenderPass::RenderPass(Type type, bool cuda)
 : m_cuda{cuda}
 , m_type{type}
 , m_framebuffer{Magnum::NoCreate}
 , m_ssaoFramebuffer{Magnum::NoCreate}
 , m_ssaoApplyFramebuffer{Magnum::NoCreate}
 , m_toneMapFramebuffer{Magnum::NoCreate}
 , m_renderShader{std::make_unique<RenderShader>()}
 , m_backgroundShader{std::make_unique<BackgroundShader>()}
 , m_backgroundCubeShader{std::make_unique<BackgroundCubeShader>()}
 , m_ssaoShader{std::make_unique<SSAOShader>()}
 , m_ssaoApplyShader{std::make_unique<SSAOApplyShader>()}
 , m_toneMapShader{std::make_unique<ToneMapShader>()}
 , m_meshShader{std::make_unique<Magnum::Shaders::MeshVisualizerGL3D>(Shaders::MeshVisualizerGL3D::Flag::Wireframe)}
{
    m_quadMesh = MeshTools::compile(Primitives::squareSolid());
    m_cubeMesh = MeshTools::compile(Primitives::cubeSolid());
    m_backgroundPlaneMesh = MeshTools::compile(Primitives::planeSolid(Primitives::PlaneFlag::TextureCoordinates));
    m_sphereMesh = MeshTools::compile(Primitives::uvSphereSolid(80, 80));

    m_result = std::make_shared<Result>(cuda);
}

RenderPass::~RenderPass()
{
}

std::shared_ptr<RenderPass::Result> RenderPass::render(Scene& scene, const std::shared_ptr<Result>& preAllocatedResult, RenderPass::Result* depthBufferResult, const DrawPredicate& predicate)
{
    scene.loadVisual();

    if(m_drawPhysics)
    {
        for(auto& obj : scene.objects())
            obj->loadPhysicsVisualization();
    }

    // At the moment, SSAO + physics are not compatible, needs some work below
    bool ssaoEnabled = m_ssaoEnabled && !m_drawPhysics;

    constexpr Color4 invalid{3000.0, 3000.0, 3000.0, 3000.0};

    if constexpr(false)
    {
        GL::Renderer::enable(GL::Renderer::Feature::DebugOutput);
        GL::Renderer::enable(GL::Renderer::Feature::DebugOutputSynchronous);
        GL::DebugOutput::setDefaultCallback();
    }

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);

    // Not really, but we skip the usual right-handed -> left-handed
    // coordinate system change (see scene.cpp), which messes everything up.
    GL::Renderer::setFrontFace(GL::Renderer::FrontFace::ClockWise);

    // Setup the framebuffer
    auto viewport = scene.viewport();

    std::shared_ptr<Result> result = preAllocatedResult;
    if (!result)
    {
        result = m_result;
    }

    // Make sure the result textures are not mapped before writing/changing anything
    if(result->isMapped())
        result->unmapCUDA();

    if (result->rgb.imageSize() != scene.viewport())
    {
        result->rgb.setStorage(GL::TextureFormat::RGBA8, 4, viewport);
        result->objectCoordinates.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
        result->classIndex
            .setStorage(GL::TextureFormat::R16UI, 2, viewport)
            .setMagnificationFilter(GL::SamplerFilter::Nearest)
            .setMinificationFilter(GL::SamplerFilter::Nearest);
        result->instanceIndex
            .setStorage(GL::TextureFormat::R16UI, 2, viewport)
            .setMagnificationFilter(GL::SamplerFilter::Nearest)
            .setMinificationFilter(GL::SamplerFilter::Nearest);

        result->normals.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);

        result->vertexIndex.setStorage(GL::TextureFormat::RGBA32UI, 4 * sizeof(std::uint32_t), viewport);
        result->barycentricCoeffs.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
        result->camCoordinates.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
    }

    if(!m_initialized || m_framebuffer.viewport().size() != scene.viewport())
    {
        UnsignedInt levels = Math::log2(viewport.max())+1;

        m_framebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

        m_depthbuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, viewport);

        // SSAO
        m_ssaoFramebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};
        m_ssaoRGBInputTexture
            .setStorage(levels, GL::TextureFormat::RGBA32F, viewport)
            .setMinificationFilter(SamplerFilter::Linear, SamplerMipmap::Linear)
            .setMagnificationFilter(SamplerFilter::Nearest);
        m_ssaoTexture
            .setStorage(1, GL::TextureFormat::R32F, viewport)
            .setMinificationFilter(SamplerFilter::Nearest)
            .setMagnificationFilter(SamplerFilter::Nearest);

        m_ssaoApplyFramebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

        m_toneMapInputTexture
            .setStorage(levels, GL::TextureFormat::RGBA32F, viewport)
            .setMinificationFilter(SamplerFilter::Linear, SamplerMipmap::Linear)
            .setMaxLevel(levels-1)
        ;
        m_toneMapFramebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

        Corrade::Containers::Array<Magnum::Float> data(Corrade::ValueInit, 4 * viewport.x()*viewport.y());
        ImageView2D zeroImage{Magnum::PixelFormat::RGBA32F, {viewport.x(), viewport.y()}, data};
        m_zeroMinDepth.setMagnificationFilter(GL::SamplerFilter::Linear)
            .setMagnificationFilter(GL::SamplerFilter::Linear)
            .setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy())
            .setStorage(GL::TextureFormat::RGBA32F, {viewport.x(), viewport.y()})
            .setSubImage({}, zeroImage);

        m_initialized = true;
    }

    Magnum::GL::RectangleTexture* minDepth;
    if (depthBufferResult)
    {
        minDepth = &depthBufferResult->objectCoordinates;
    }
    else
    {
        minDepth = &m_zeroMinDepth;
    }

    m_framebuffer
        .attachTexture(
            GL::Framebuffer::ColorAttachment{0},
            ssaoEnabled ? m_ssaoRGBInputTexture : m_toneMapInputTexture,
            0
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{1},
            result->objectCoordinates
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{2},
            result->classIndex
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{3},
            result->instanceIndex
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{4},
            result->normals
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{5},
            result->vertexIndex
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{6},
            result->barycentricCoeffs
        )
        .attachTexture(
            GL::Framebuffer::ColorAttachment{7},
            result->camCoordinates
        )
        .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, m_depthbuffer)
        .mapForDraw({
            {RenderShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}},
            {RenderShader::ObjectCoordinatesOutput, GL::Framebuffer::ColorAttachment{1}},
            {RenderShader::ClassIndexOutput, GL::Framebuffer::ColorAttachment{2}},
            {RenderShader::InstanceIndexOutput, GL::Framebuffer::ColorAttachment{3}},
            {RenderShader::NormalOutput, GL::Framebuffer::ColorAttachment{4}},
            {RenderShader::VertexIndexOutput, GL::Framebuffer::ColorAttachment{5}},
            {RenderShader::BarycentricCoeffsOutput, GL::Framebuffer::ColorAttachment{6}},
            {RenderShader::CamCoordinatesOutput, GL::Framebuffer::ColorAttachment{7}}
        })
    ;

    if(m_framebuffer.checkStatus(GL::FramebufferTarget::Draw) != GL::Framebuffer::Status::Complete)
    {
        Error{} << "Invalid framebuffer status:" << m_framebuffer.checkStatus(GL::FramebufferTarget::Draw);
        std::abort();
    }

    m_framebuffer.bind();

    m_framebuffer.clear(GL::FramebufferClear::Depth);

    m_framebuffer.clearColor(0, 0x00000000_rgbaf);
    m_framebuffer.clearColor(1, invalid);
    m_framebuffer.clearColor(2, Vector4ui{0});
    m_framebuffer.clearColor(3, Vector4ui{0});
    m_framebuffer.clearColor(4, 0x00000000_rgbaf);
    m_framebuffer.clearColor(5, Vector4ui{0});
    m_framebuffer.clearColor(6, 0x00000000_rgbf);
    m_framebuffer.clearColor(7, invalid);

    if(scene.lightMap())
        m_renderShader->setLightMap(*scene.lightMap());
    else
        Error{} << "Scenes without light maps are currently not supported";

    m_renderShader->bindDepthTexture(*minDepth);

    // Do we have a background plane?
    if(scene.backgroundPlaneSize().dot() > 0)
    {
        // Scale the unit plane s.t. it has the desired dimensions
        Matrix4 scaledPoseInWorld = scene.backgroundPlanePose() * Matrix4::scaling({
            scene.backgroundPlaneSize().x() / 2.0f,
            scene.backgroundPlaneSize().y() / 2.0f,
            1.0f
        });

        Error{} << "Background plane is not supported";
    }

    // Let the fun begin!
    for(auto& object : scene.objects())
    {
        if(predicate && !predicate(object))
            continue;

        Matrix4 objectToWorld = object->pose();

        Matrix4 objectToCam = scene.camera().cameraMatrix() * object->pose();
        Matrix4 objectToCamInv = objectToCam.invertedRigid();

        Matrix4 worldToCam = scene.camera().object().absoluteTransformationMatrix().invertedRigid();

        (*m_renderShader)
            .setProjectionMatrix(scene.camera().projectionMatrix())

            .setClassIndex(object->mesh()->classIndex())
            .setInstanceIndex(object->instanceIndex())

            .setStickerProjection(object->stickerViewProjection())
            .setStickerRange(object->stickerRange())
        ;

        if(object->stickerTexture())
            m_renderShader->bindStickerTexture(*object->stickerTexture());

        auto materialOverride = MaterialOverride{}
            .metallic(object->metallic())
            .roughness(object->roughness())
        ;

        object->draw(scene.camera(), [&](const Matrix4& meshToCam, SceneGraph::Camera3D& cam, Drawable* drawable) {
            Matrix4 meshToObject = objectToCamInv * meshToCam;

            (*m_renderShader)
                .setMaterial(drawable->material(), object->mesh()->textures(), materialOverride)
                .setTransformations(meshToObject, objectToWorld, worldToCam)
                .draw(drawable->mesh())
            ;
        });
    }

    std::mt19937 seqGen{0};
    std::uniform_real_distribution<float> scalarGen(0.0, 1.0);
    auto randomColor = [&](float alpha = 1.0f) -> Color4 {
        return {scalarGen(seqGen), scalarGen(seqGen), scalarGen(seqGen), alpha};
    };

    if(m_drawBounding != DrawBounding::Disabled)
    {
        m_framebuffer.mapForDraw({
            {RenderShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
        });

        GL::Renderer::enable(GL::Renderer::Feature::Blending);
        GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
            GL::Renderer::BlendEquation::Max);
        GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
            GL::Renderer::BlendFunction::OneMinusSourceAlpha);

        switch(m_drawBounding)
        {
            case DrawBounding::Disabled:
                break;

            case DrawBounding::Spheres:
                for(auto& object : scene.objects())
                {
                    Matrix4 scaling = Matrix4::scaling(Vector3{0.5f * object->mesh()->bbox().size().length()});
                    Matrix4 pos = Matrix4::translation(object->mesh()->bbox().center());

                    (*m_meshShader)
                        .setColor(randomColor(0.8f))
                        .setWireframeColor(0xdcdcdc_rgbf)
                        .setViewportSize(Vector2{scene.viewport()})
                        .setTransformationMatrix(scene.camera().cameraMatrix() * object->pose() * pos * scaling)
                        .setProjectionMatrix(scene.camera().projectionMatrix())
                        .draw(m_sphereMesh);
                }
                break;

            case DrawBounding::Boxes:
                for(auto& object : scene.objects())
                {
                    Matrix4 scaling = Matrix4::scaling(0.5f * object->mesh()->bbox().size());
                    Matrix4 pos = Matrix4::translation(object->mesh()->bbox().center());

                    (*m_meshShader)
                        .setColor(randomColor(0.8f))
                        .setWireframeColor(0xdcdcdc_rgbf)
                        .setViewportSize(Vector2{scene.viewport()})
                        .setTransformationMatrix(scene.camera().cameraMatrix() * object->pose() * pos * scaling)
                        .setProjectionMatrix(scene.camera().projectionMatrix())
                        .draw(m_cubeMesh);
                }
                break;
        }
    }

    if(m_drawPhysics)
    {
        m_framebuffer.mapForDraw({
            {RenderShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
        });

        // Just draw over everything
        m_framebuffer.clear(GL::FramebufferClear::Depth);

        GL::Renderer::enable(GL::Renderer::Feature::Blending);
        GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
            GL::Renderer::BlendEquation::Max);
        GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
            GL::Renderer::BlendFunction::OneMinusSourceAlpha);

        for(auto& object : scene.objects())
        {
            object->drawPhysics(scene.camera(), [&](const Matrix4& meshToCam, SceneGraph::Camera3D& cam, Drawable* drawable) {
                (*m_meshShader)
                    .setColor(randomColor())
                    .setWireframeColor(0xdcdcdc_rgbf)
                    .setViewportSize(Vector2{scene.viewport()})
                    .setTransformationMatrix(meshToCam)
                    .setProjectionMatrix(cam.projectionMatrix())
                    .draw(drawable->mesh());
            });
        }
    }

    GL::Renderer::setFrontFace(GL::Renderer::FrontFace::CounterClockWise);

    if(ssaoEnabled)
        m_ssaoRGBInputTexture.generateMipmap();
    else
        m_toneMapInputTexture.generateMipmap();

    // Do we have a background texture?
    if(scene.backgroundImage())
    {
        m_framebuffer.mapForDraw({
            {BackgroundShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
        });
        m_backgroundShader->
            bindRGB(*scene.backgroundImage())
            .draw(m_quadMesh);
    }
    else if(scene.lightMap())
    {
        m_framebuffer.mapForDraw({
            {BackgroundCubeShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
        });

        GL::Renderer::setDepthFunction(GL::Renderer::DepthFunction::LessOrEqual);
        m_backgroundCubeShader->bindRGB(scene.lightMap()->cubeMap());
        m_backgroundCubeShader->setViewMatrix(scene.camera().cameraMatrix());
        m_backgroundCubeShader->setProjectionMatrix(scene.camera().projectionMatrix());
        m_backgroundCubeShader->draw(m_cubeMesh);

        GL::Renderer::setDepthFunction(GL::Renderer::DepthFunction::Less);
    }

    if(ssaoEnabled)
    {
        m_ssaoFramebuffer
            .attachTexture(GL::Framebuffer::ColorAttachment{0}, m_ssaoTexture, 0)
            .mapForDraw({
                {SSAOShader::AOOutput, GL::Framebuffer::ColorAttachment{0}}
            });

        m_ssaoFramebuffer.bind();
        m_ssaoFramebuffer.clearColor(0, Color4{1.0f});

        (*m_ssaoShader)
            .setProjection(scene.camera().projectionMatrix())
            .bindCoordinates(result->camCoordinates)
            .bindNormals(result->normals)
            .bindNoise()
            .draw(m_quadMesh);

        m_ssaoApplyFramebuffer
            .attachTexture(GL::Framebuffer::ColorAttachment{0}, m_toneMapInputTexture, 0)
            .mapForDraw({
                {SSAOApplyShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
            });

        m_ssaoApplyFramebuffer.bind();
        m_ssaoApplyFramebuffer.clearColor(0, Color4{0.0f});

        (*m_ssaoApplyShader)
            .bindAO(m_ssaoTexture)
            .bindColor(m_ssaoRGBInputTexture)
            .bindCoordinates(result->camCoordinates)
            .draw(m_quadMesh);
    }

    // HDR Tonemapping
    {
        m_toneMapFramebuffer
            .attachTexture(GL::Framebuffer::ColorAttachment{0}, result->rgb)
            .mapForDraw({
                {ToneMapShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
            });

        m_toneMapFramebuffer.bind();
        (*m_toneMapShader)
            .bindColor(m_toneMapInputTexture)
            .bindObjectLuminance(ssaoEnabled ? m_ssaoRGBInputTexture : m_toneMapInputTexture)
            .draw(m_quadMesh);
    }

    // Map for CUDA access
    result->mapCUDA();

    return result;
}

void RenderPass::setSSAOEnabled(bool enabled)
{
    m_ssaoEnabled = enabled;
}

void RenderPass::setDrawPhysicsEnabled(bool enabled)
{
    m_drawPhysics = enabled;
}

void RenderPass::setDrawBounding(DrawBounding drawBounding)
{
    m_drawBounding = drawBounding;
}

}
