// Complete render pass for complex scenes
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/render_pass.h>
#include <stillleben/scene.h>
#include <stillleben/object.h>
#include <stillleben/mesh.h>
#include <stillleben/cuda_interop.h>
#include <stillleben/context.h>
#include <stillleben/light_map.h>

#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Utility/DebugStl.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/DebugOutput.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Trade/AbstractImageConverter.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>

#include <Magnum/MeshTools/Compile.h>

#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/Primitives/UVSphere.h>

#include <Magnum/PixelFormat.h>

#include <Magnum/SceneGraph/Camera.h>

#include "shaders/render_shader.h"
#include "shaders/background_shader.h"
#include "shaders/background_cube_shader.h"
#include "shaders/ssao_shader.h"
#include "shaders/ssao_apply_shader.h"
#include "shaders/shadow_shader.h"
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

using FrustumCorners = Containers::StaticArray<8, Vector3>;

FrustumCorners computeFrustumCorners(Scene& scene)
{
    auto& camera = scene.camera();
    Matrix4 P = camera.projectionMatrix();
    Matrix4 Pinv = P.inverted();

    Float near = -1.0f;
    Float far = 1.0f;

    if(!scene.objects().empty())
    {
        Float nearObj = std::numeric_limits<Float>::infinity();
        Float farObj = -std::numeric_limits<Float>::infinity();

        for(auto& obj : scene.objects())
        {
            Vector3 objInCam = (camera.cameraMatrix() * obj->pose()).transformPoint(obj->mesh()->bbox().center());

            Float radius = obj->mesh()->bbox().size().length() / 2;
            Vector3 nearPoint = objInCam - Vector3::zAxis(radius);
            Vector3 farPoint  = objInCam + Vector3::zAxis(radius);

            nearPoint = camera.projectionMatrix().transformPoint(nearPoint);
            farPoint = camera.projectionMatrix().transformPoint(farPoint);

            nearObj = Math::min(nearObj, nearPoint.z());
            farObj = Math::max(farObj, farPoint.z());
        }

        near = Math::max(Math::max(-1.0f, nearObj), near);
        far = Math::min(farObj, far);
    }

    // homogeneous corner coords
    Containers::StaticArray<8, Vector4> hcorners{InPlaceInit,
        // near
        Vector4{-1,  1, near, 1},
        Vector4{ 1,  1, near, 1},
        Vector4{ 1, -1, near, 1},
        Vector4{-1, -1, near, 1},

        // far
        Vector4{-1,  1, far, 1},
        Vector4{ 1,  1, far, 1},
        Vector4{ 1, -1, far, 1},
        Vector4{-1, -1, far, 1}
    };

    FrustumCorners corners;
    for(UnsignedInt i = 0; i < 8; ++i)
    {
        Vector4 p = camera.cameraMatrix().invertedRigid() * Pinv * hcorners[i];
        corners[i] = p.xyz() / p.w();
    }

    return corners;
}

Matrix4 computeShadowMapMatrix(FrustumCorners& corners, const Vector3& lightDirection)
{
    // Z always points into the scene
    Vector3 z = lightDirection.normalized();

    Vector3 x = Math::cross(z, Vector3::zAxis()).normalized();
    Vector3 y = Math::cross(z, x).normalized();

    Matrix4 camToWorld = Matrix4::from(
        Matrix3{x,y,z}, {}
    );

    Matrix4 worldToCam = camToWorld.invertedRigid();

    Vector3 minInCam = Vector3{std::numeric_limits<Float>::infinity()};
    Vector3 maxInCam = Vector3{-std::numeric_limits<Float>::infinity()};

    for(UnsignedInt i = 0; i < 8; ++i)
    {
        Vector3 cornerInCam = worldToCam.transformPoint(corners[i]);

        minInCam = Math::min(minInCam, cornerInCam);
        maxInCam = Math::max(maxInCam, cornerInCam);
    }

    Float near = minInCam.z();
    Float far = maxInCam.z();

    Float L = minInCam.x();
    Float R = maxInCam.x();
    Float T = minInCam.y();
    Float B = maxInCam.y();

    Matrix4 P{
        {2.0f / (R-L),         0.0f,                   0.0f, 0.0f},
        {        0.0f, 2.0f / (B-T),                   0.0f, 0.0f},
        {        0.0f,         0.0f,        2.0f/(far-near), 0.0f},
        {-(R+L)/(R-L), -(B+T)/(B-T), -(far+near)/(far-near), 1.0f}
    };

    // Sanity check
    // P * (R,0,0,1) = (2.0 * R / (R-L) - (R+L)/(R-L), *, *, 1.0f)
    //   => (2*R - R - L) / (R-L) = 1

    return P * worldToCam;
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
 , m_postprocessFramebuffer{Magnum::NoCreate}
 , m_renderShader{std::make_unique<RenderShader>()}
 , m_backgroundShader{std::make_unique<BackgroundShader>()}
 , m_backgroundCubeShader{std::make_unique<BackgroundCubeShader>()}
 , m_ssaoShader{std::make_unique<SSAOShader>()}
 , m_ssaoApplyShader{std::make_unique<SSAOApplyShader>()}
 , m_toneMapShader{std::make_unique<ToneMapShader>()}
 , m_meshShader{std::make_unique<Magnum::Shaders::MeshVisualizerGL3D>(Shaders::MeshVisualizerGL3D::Flag::Wireframe)}
 , m_shadowShader{std::make_unique<ShadowShader>()}
{
    m_quadMesh = MeshTools::compile(Primitives::squareSolid());
    m_cubeMesh = MeshTools::compile(Primitives::cubeSolid());
    m_backgroundPlaneMesh = MeshTools::compile(Primitives::planeSolid(Primitives::PlaneFlag::TextureCoordinates));
    m_sphereMesh = MeshTools::compile(Primitives::uvSphereSolid(80, 80));

    Vector2i shadowResolution{2048, 2048};
    m_shadowMaps
        .setWrapping(GL::SamplerWrapping::ClampToEdge)
        .setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy())
        .setMaxLevel(0)
        .setCompareFunction(GL::SamplerCompareFunction::LessOrEqual)
        .setCompareMode(GL::SamplerCompareMode::CompareRefToTexture)
        .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Base)
        .setMagnificationFilter(GL::SamplerFilter::Linear)
        .setImage(0, GL::TextureFormat::DepthComponent, ImageView3D{GL::PixelFormat::DepthComponent, GL::PixelType::Float, {shadowResolution, sl::NumLights}})
    ;

    Containers::arrayResize(m_shadowFB, DirectInit, sl::NumLights, Range2Di::fromSize({}, shadowResolution));
    for(UnsignedInt i = 0; i < sl::NumLights; ++i)
    {
        m_shadowFB[i]
            .attachTextureLayer(GL::Framebuffer::BufferAttachment::Depth, m_shadowMaps, 0, i)
            .mapForDraw(GL::Framebuffer::DrawAttachment::None)
            .bind();

        CORRADE_INTERNAL_ASSERT(m_shadowFB[i].checkStatus(GL::FramebufferTarget::Draw) == GL::Framebuffer::Status::Complete);
    }

    Containers::arrayResize(m_shadowMatrices, sl::NumLights);

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

    GL::Renderer::disable(GL::Renderer::Feature::Blending);

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

        m_postprocessInput
            .setStorage(levels, GL::TextureFormat::RGBA32F, viewport)
            .setMinificationFilter(SamplerFilter::Linear, SamplerMipmap::Linear)
            .setMaxLevel(levels-1)
        ;
        m_postprocessFramebuffer = GL::Framebuffer{Range2Di::fromSize({}, viewport)};

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

    // Shadow mapping
    {
        Containers::ArrayView<const Vector3> lightDirections;
        Containers::ArrayView<const Color3> lightColors;

        if(scene.lightMap())
        {
            lightDirections = scene.lightMap()->lightDirections();
            lightColors = scene.lightMap()->lightColors();

            if(lightDirections.size() > NumLights)
                lightDirections = lightDirections.slice(0, NumLights);
        }
        else
        {
            lightDirections = scene.lightDirections();
            lightColors = scene.lightColors();
        }

        FrustumCorners frustum = computeFrustumCorners(scene);

        GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
        GL::Renderer::setFaceCullingMode(GL::Renderer::PolygonFacing::Front);
        for(UnsignedInt i = 0; i < lightDirections.size(); ++i)
        {
            // Skip lights with no output
            if(lightColors[i] == Color3{0.0} || lightDirections[i] == Vector3{0.0f})
                continue;

            m_shadowMatrices[i] = computeShadowMapMatrix(frustum, lightDirections[i]);

            m_shadowFB[i]
                .clear(GL::FramebufferClear::Depth)
                .bind();

            for(auto& object : scene.objects())
            {
                if(predicate && !predicate(object))
                    continue;

                object->draw(scene.camera(), [&](const Matrix4&, SceneGraph::Camera3D&, Drawable* drawable) {
                    (*m_shadowShader)
                        .setTransformation(m_shadowMatrices[i] * drawable->object().absoluteTransformationMatrix())
                        .draw(drawable->mesh())
                    ;
                });
            }
        }
        GL::Renderer::setFaceCullingMode(GL::Renderer::PolygonFacing::Back);
        GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    }

    Magnum::GL::RectangleTexture* minDepth;
    if(depthBufferResult)
        minDepth = &depthBufferResult->objectCoordinates;
    else
        minDepth = &m_zeroMinDepth;

    m_framebuffer
        .attachTexture(
            GL::Framebuffer::ColorAttachment{0},
            ssaoEnabled ? m_ssaoRGBInputTexture : m_postprocessInput,
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
        m_renderShader->setManualLighting(scene.lightDirections(), scene.lightColors(), scene.ambientLight());

    (*m_renderShader)
        .bindDepthTexture(*minDepth)
        .setProjectionMatrix(scene.camera().projectionMatrix())
        .setShadowMap(m_shadowMaps, m_shadowMatrices)
    ;

    // Do we have a background plane?
    if(scene.backgroundPlaneSize().dot() > 0)
    {
        // Scale the unit plane s.t. it has the desired dimensions
        Matrix4 scaledPoseInWorld = scene.backgroundPlanePose() * Matrix4::scaling({
            scene.backgroundPlaneSize().x() / 2.0f,
            scene.backgroundPlaneSize().y() / 2.0f,
            1.0f
        });

        auto texture = scene.backgroundPlaneTexture();

        Trade::MaterialData material = [&](){
            if(texture)
            {
                return Trade::MaterialData{Trade::MaterialType::PbrMetallicRoughness, {
                    {Trade::MaterialAttribute::BaseColor, Color4{1.0f}},
                    {Trade::MaterialAttribute::BaseColorTexture, 0u}
                }};
            }
            else
            {
                return Trade::MaterialData{Trade::MaterialType::PbrMetallicRoughness, {
                    {Trade::MaterialAttribute::BaseColor, Color4{0.0f, 0.8f, 0.0f, 1.0f}},
                }};
            }
        }();

        auto textures = Containers::array<GL::Texture2D*>({texture.get()});

        (*m_renderShader)
            .setClassIndex(0)
            .setInstanceIndex(0)
            .setStickerRange({})
            .setMaterial(material, textures, {})
            .setTransformations(Matrix4{Magnum::Math::IdentityInit}, scaledPoseInWorld, scene.camera().cameraMatrix())
            .draw(m_backgroundPlaneMesh);
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

    GL::Renderer::setFrontFace(GL::Renderer::FrontFace::CounterClockWise);

    if(ssaoEnabled)
        m_ssaoRGBInputTexture.generateMipmap();
    else
        m_postprocessInput.generateMipmap();

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
            .attachTexture(GL::Framebuffer::ColorAttachment{0}, m_postprocessInput, 0)
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

    // Postprocessing
    m_postprocessFramebuffer
        .attachTexture(GL::Framebuffer::ColorAttachment{0}, result->rgb)
        .mapForDraw({
            {ToneMapShader::ColorOutput, GL::Framebuffer::ColorAttachment{0}}
        });

    m_postprocessFramebuffer.bind();

    // HDR Tonemapping
    (*m_toneMapShader)
        .bindColor(m_postprocessInput)
        .bindObjectLuminance(ssaoEnabled ? m_ssaoRGBInputTexture : m_postprocessInput)
        .draw(m_quadMesh);

    // Draw overlays (physics / bbox debugging)
    if(m_drawPhysics || m_drawBounding != DrawBounding::Disabled)
    {
        // We re-use the original framebuffer to get access to the depth buffer
        m_framebuffer.attachTexture(GL::Framebuffer::ColorAttachment{0}, result->rgb);
        m_framebuffer.mapForDraw({
            {0, GL::Framebuffer::ColorAttachment{0}}
        });
        m_framebuffer.bind();
        GL::Renderer::setFrontFace(GL::Renderer::FrontFace::ClockWise);

        GL::Renderer::enable(GL::Renderer::Feature::Blending);
        GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
            GL::Renderer::BlendEquation::Max);
        GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
            GL::Renderer::BlendFunction::OneMinusSourceAlpha);

        GL::Renderer::setDepthFunction(GL::Renderer::DepthFunction::LessOrEqual);

        if(m_drawPhysics)
        {
            m_framebuffer.clear(GL::FramebufferClear::Depth);

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
                    Matrix4 scaling = Matrix4::scaling(1.0001f * 0.5f * object->mesh()->bbox().size());
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

        GL::Renderer::disable(GL::Renderer::Feature::Blending);
        GL::Renderer::setFrontFace(GL::Renderer::FrontFace::CounterClockWise);
        GL::Renderer::setDepthFunction(GL::Renderer::DepthFunction::Less);
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
