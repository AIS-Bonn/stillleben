// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/mesh.h>
#include <stillleben/cuda_interop.h>
#include <stillleben/context.h>

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

#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/Square.h>

#include <Magnum/PixelFormat.h>

#include <Magnum/Trade/MeshData2D.h>

#include "shaders/render_shader.h"
#include "shaders/background_shader.h"
#include "shaders/ssao_shader.h"
#include "shaders/ssao_apply_shader.h"

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
            return {};
        default:
            return {};
    }
}

}

RenderPass::Result::Result(bool cuda)
 : m_mapper{cuda}
 , rgb{m_mapper}
 , objectCoordinates{m_mapper}
 , classIndex{m_mapper}
 , instanceIndex{m_mapper}
 , normals{m_mapper}
 , validMask{m_mapper}
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
 , m_framebuffer{Magnum::NoCreate}
 , m_ssaoFramebuffer{Magnum::NoCreate}
 , m_ssaoApplyFramebuffer{Magnum::NoCreate}
 , m_shaderTextured{std::make_unique<RenderShader>(flagsForType(type) | RenderShader::Flag::DiffuseTexture)}
 , m_shaderVertexColors{std::make_unique<RenderShader>(flagsForType(type) | RenderShader::Flag::VertexColors)}
 , m_shaderUniform{std::make_unique<RenderShader>(flagsForType(type))}
 , m_backgroundShader{std::make_unique<BackgroundShader>()}
 , m_ssaoShader{std::make_unique<SSAOShader>()}
 , m_ssaoApplyShader{std::make_unique<SSAOApplyShader>()}
{
    m_quadMesh = MeshTools::compile(Primitives::squareSolid(Primitives::SquareTextureCoords::DontGenerate));
    m_backgroundPlaneMesh = MeshTools::compile(Primitives::planeSolid(Primitives::PlaneTextureCoords::Generate));

    m_result = std::make_shared<Result>(cuda);
}

RenderPass::~RenderPass()
{
}

std::shared_ptr<RenderPass::Result> RenderPass::render(Scene& scene, const std::shared_ptr<Result>& preAllocatedResult, RenderPass::Result* depthBufferResult)
{
    scene.loadVisual();

    constexpr Color4 invalid{3000.0, 3000.0, 3000.0, 3000.0};

    if constexpr(false)
    {
        GL::Renderer::enable(GL::Renderer::Feature::DebugOutput);
        GL::Renderer::enable(GL::Renderer::Feature::DebugOutputSynchronous);
        GL::DebugOutput::setDefaultCallback();
    }

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

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
        result->classIndex.setStorage(GL::TextureFormat::R16UI, 2, viewport);
        result->instanceIndex.setStorage(GL::TextureFormat::R16UI, 2, viewport);
        result->normals.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
        result->validMask.setStorage(GL::TextureFormat::R8UI, 1, viewport);

        result->vertexIndex.setStorage(GL::TextureFormat::RGBA32UI, 4 * sizeof(std::uint32_t), viewport);
        result->barycentricCoeffs.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
        result->camCoordinates.setStorage(GL::TextureFormat::RGBA32F, 4 * sizeof(float), viewport);
    }

    if(!m_initialized || m_framebuffer.viewport().size() != scene.viewport())
    {
        m_framebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

        m_depthbuffer.setStorage(GL::RenderbufferFormat::DepthComponent24, viewport);

        // SSAO
        m_ssaoFramebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};
        m_ssaoRGBInputTexture
            .setStorage(GL::TextureFormat::RGBA32F, viewport)
            .setMinificationFilter(SamplerFilter::Nearest)
            .setMagnificationFilter(SamplerFilter::Nearest);
        m_ssaoTexture
            .setStorage(GL::TextureFormat::R32F, viewport)
            .setMinificationFilter(SamplerFilter::Nearest)
            .setMagnificationFilter(SamplerFilter::Nearest);

        m_ssaoApplyFramebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

        Corrade::Containers::Array<Magnum::Float> data(Corrade::Containers::ValueInit, 4 * viewport.x()*viewport.y());
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
            m_ssaoEnabled ? m_ssaoRGBInputTexture : result->rgb
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

    // Do we have a background texture?
    if(scene.backgroundImage())
    {
        GL::Renderer::setFrontFace(GL::Renderer::FrontFace::CounterClockWise);
        m_backgroundShader->bindRGB(*scene.backgroundImage());
        m_quadMesh.draw(*m_backgroundShader);

        // Draw on top
        m_framebuffer.clear(GL::FramebufferClear::Depth);
        GL::Renderer::setFrontFace(GL::Renderer::FrontFace::ClockWise);
    }
    else
    {
        m_framebuffer.clearColor(0, scene.backgroundColor());
    }

    m_framebuffer.clearColor(1, invalid);
    m_framebuffer.clearColor(2, Vector4ui{0});
    m_framebuffer.clearColor(3, Vector4ui{0});
    m_framebuffer.clearColor(4, 0x00000000_rgbaf);
    m_framebuffer.clearColor(5, Vector4ui{0});
    m_framebuffer.clearColor(6, 0x00000000_rgbf);
    m_framebuffer.clearColor(7, invalid);

    for(auto& shader : {std::ref(m_shaderTextured), std::ref(m_shaderUniform), std::ref(m_shaderVertexColors)})
    {
        shader.get()->bindDepthTexture(*minDepth);

        // Setup image-based lighting if required
        if(scene.lightMap())
            shader.get()->bindLightMap(*scene.lightMap());
        else
            shader.get()->disableLightMap();
    }

    Vector3 lightPositionInCam = scene.camera().cameraMatrix().transformPoint(scene.lightPosition());

    // Do we have a background plane?
    if(scene.backgroundPlaneSize().dot() > 0)
    {
        auto poseInCam = scene.camera().object().absoluteTransformationMatrix().inverted() * scene.backgroundPlanePose();

        // Scale the unit plane s.t. it has the desired dimensions
        Matrix4 scaledPoseInCam = poseInCam * Matrix4::scaling({
            scene.backgroundPlaneSize().x() / 2.0f,
            scene.backgroundPlaneSize().y() / 2.0f,
            1.0f
        });

        auto texture = scene.backgroundPlaneTexture();
        if(texture)
        {
            (*m_shaderTextured)
                .setObjectToCamMatrix(scaledPoseInCam)
                .setMeshToObjectMatrix(Matrix4{Magnum::Math::IdentityInit})
                .setNormalMatrix(poseInCam.rotation())
                .setProjectionMatrix(scene.camera().projectionMatrix())
                .setClassIndex(0)
                .setInstanceIndex(0)
                .setAmbientColor(scene.ambientLight())
                .setSpecularColor(Magnum::Color4{1.0f})
                .setShininess(80.0f)
                .setMetalness(0.04f)
                .setRoughness(0.5f)
                .setStickerRange({})
                .setLightPosition(lightPositionInCam)
                .bindDiffuseTexture(*texture)
            ;

            m_backgroundPlaneMesh.draw(*m_shaderTextured);
        }
        else
        {
            (*m_shaderUniform)
                .setObjectToCamMatrix(scaledPoseInCam)
                .setMeshToObjectMatrix(Matrix4{Magnum::Math::IdentityInit})
                .setNormalMatrix(poseInCam.rotation())
                .setProjectionMatrix(scene.camera().projectionMatrix())
                .setClassIndex(0)
                .setInstanceIndex(0)
                .setAmbientColor(scene.ambientLight())
                .setSpecularColor(Magnum::Color4{1.0f})
                .setShininess(80.0f)
                .setMetalness(0.04f)
                .setRoughness(0.5f)
                .setStickerRange({})
                .setLightPosition(lightPositionInCam)
                .setDiffuseColor({0.0f, 0.8f, 0.0f, 1.0f})
            ;

            m_backgroundPlaneMesh.draw(*m_shaderUniform);
        }
    }

    // Let the fun begin!
    for(auto& object : scene.objects())
    {
        Matrix4 objectToCam = scene.camera().object().absoluteTransformationMatrix().inverted() * object->pose();
        Matrix4 objectToCamInv = objectToCam.inverted();

        for(auto& shader : {std::ref(m_shaderTextured), std::ref(m_shaderUniform), std::ref(m_shaderVertexColors)})
        {
            (*shader.get())
                .setObjectToCamMatrix(objectToCam)
                .setClassIndex(object->mesh()->classIndex())
                .setInstanceIndex(object->instanceIndex())
                .setAmbientColor(scene.ambientLight())
                .setSpecularColor(object->specularColor())
                .setShininess(object->shininess())

                .setLightPosition(lightPositionInCam)

                .setStickerProjection(object->stickerViewProjection())
                .setStickerRange(object->stickerRange())
            ;

            if(object->stickerTexture())
                shader.get()->bindStickerTexture(*object->stickerTexture());

            if(scene.lightMap())
                shader.get()->bindLightMap(*scene.lightMap());
            else
                shader.get()->disableLightMap();
        }

        object->draw(scene.camera(), [&](const Matrix4& meshToCam, SceneGraph::Camera3D& cam, Drawable* drawable) {
            Matrix4 meshToObject = objectToCamInv * meshToCam;

            if(drawable->texture())
            {
                (*m_shaderTextured)
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .bindDiffuseTexture(*drawable->texture())
                    .setMetalness(object->metalness() * drawable->metallic())
                    .setRoughness(object->roughness() * drawable->roughness())
                ;

                drawable->mesh().draw(*m_shaderTextured);
            }
            else if(drawable->hasVertexColors())
            {
                (*m_shaderVertexColors)
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .setDiffuseColor(drawable->color())
                    .setMetalness(object->metalness() * drawable->metallic())
                    .setRoughness(object->roughness() * drawable->roughness())
                ;

                drawable->mesh().draw(*m_shaderVertexColors);
            }
            else
            {
                (*m_shaderUniform)
                    .setMeshToObjectMatrix(meshToObject)
                    .setNormalMatrix(meshToCam.rotation())
                    .setProjectionMatrix(cam.projectionMatrix())
                    .setDiffuseColor(drawable->color())
                    .setMetalness(object->metalness() * drawable->metallic())
                    .setRoughness(object->roughness() * drawable->roughness())
                ;

                drawable->mesh().draw(*m_shaderUniform);
            }
        });
    }

    GL::Renderer::setFrontFace(GL::Renderer::FrontFace::CounterClockWise);

    if(m_ssaoEnabled)
    {
        m_ssaoFramebuffer
            .attachTexture(GL::Framebuffer::ColorAttachment{0}, m_ssaoTexture)
            .mapForDraw({
                {SSAOShader::AOOutput, GL::Framebuffer::ColorAttachment{0}}
            });

        m_ssaoFramebuffer.bind();
        m_ssaoFramebuffer.clearColor(0, Color4{1.0f});

        (*m_ssaoShader)
            .setProjection(scene.camera().projectionMatrix())
            .bindCoordinates(result->camCoordinates)
            .bindNormals(result->normals)
            .bindNoise();

        m_quadMesh.draw(*m_ssaoShader);

        m_ssaoApplyFramebuffer
            .attachTexture(GL::Framebuffer::ColorAttachment{0}, result->rgb)
            .mapForDraw({
                {SSAOApplyShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
            });

        m_ssaoApplyFramebuffer.bind();
        m_ssaoApplyFramebuffer.clearColor(0, Color4{0.0f});

        (*m_ssaoApplyShader)
            .bindAO(m_ssaoTexture)
            .bindColor(m_ssaoRGBInputTexture)
            .bindCoordinates(result->camCoordinates);

        m_quadMesh.draw(*m_ssaoApplyShader);
    }

    // Map for CUDA access
    result->mapCUDA();

    return result;
}

void RenderPass::setSSAOEnabled(bool enabled)
{
    m_ssaoEnabled = enabled;
}

}
