// Small tool to produce pre-align files for stillleben
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

// Based off the magnum-examples viewer

/*
    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>

    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or distribute
    this software, either in source code form or as a compiled binary, for any
    purpose, commercial or non-commercial, and by any means.

    In jurisdictions that recognize copyright laws, the author or authors of
    this software dedicate any and all copyright interest in the software to
    the public domain. We make this dedication for the benefit of the public
    at large and to the detriment of our heirs and successors. We intend this
    dedication to be an overt act of relinquishment in perpetuity of all
    present and future rights to this software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/ConfigurationGroup.h>
#include <Magnum/Mesh.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>
#include <Magnum/DebugTools/ResourceManager.h>
#include <Magnum/DebugTools/ObjectRenderer.h>
#include <Magnum/ImGuiIntegration/Context.hpp>

#include <experimental/filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::experimental::filesystem;

using namespace Magnum;
using namespace Math::Literals;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class AlignMesh: public Platform::Application {
    public:
        explicit AlignMesh(const Arguments& arguments);

    private:
        void drawEvent() override;
        void viewportEvent(ViewportEvent& event) override;
        void mousePressEvent(MouseEvent& event) override;
        void mouseReleaseEvent(MouseEvent& event) override;
        void mouseMoveEvent(MouseMoveEvent& event) override;
        void mouseScrollEvent(MouseScrollEvent& event) override;

        Vector3 positionOnSphere(const Vector2i& position) const;

        void addObject(Trade::AbstractImporter& importer, Containers::ArrayView<const Containers::Optional<Trade::PhongMaterialData>> materials, Object3D& parent, UnsignedInt i);

        void readPretransform(const std::string& path);
        void writePretransform(const std::string& path);

        Shaders::Phong _coloredShader,
            _texturedShader{Shaders::Phong::Flag::DiffuseTexture};
        Containers::Array<Containers::Optional<GL::Mesh>> _meshes;
        Containers::Array<Containers::Optional<GL::Texture2D>> _textures;

        DebugTools::ResourceManager manager;
        Scene3D _scene;
        Object3D _manipulator, _cameraObject, _mesh;
        SceneGraph::Camera3D* _camera;
        SceneGraph::DrawableGroup3D _drawables;
        Vector3 _previousPosition;

        ImGuiIntegration::Context _imgui{NoCreate};

        fs::path m_pretransformPath;
};

class ColoredDrawable: public SceneGraph::Drawable3D {
    public:
        explicit ColoredDrawable(Object3D& object, Shaders::Phong& shader, GL::Mesh& mesh, const Color4& color, SceneGraph::DrawableGroup3D& group): SceneGraph::Drawable3D{object, &group}, _shader(shader), _mesh(mesh), _color{color} {}

    private:
        void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) override;

        Shaders::Phong& _shader;
        GL::Mesh& _mesh;
        Color4 _color;
};

class TexturedDrawable: public SceneGraph::Drawable3D {
    public:
        explicit TexturedDrawable(Object3D& object, Shaders::Phong& shader, GL::Mesh& mesh, GL::Texture2D& texture, SceneGraph::DrawableGroup3D& group): SceneGraph::Drawable3D{object, &group}, _shader(shader), _mesh(mesh), _texture(texture) {}

    private:
        void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) override;

        Shaders::Phong& _shader;
        GL::Mesh& _mesh;
        GL::Texture2D& _texture;
};

AlignMesh::AlignMesh(const Arguments& arguments):
    Platform::Application{arguments, Configuration{}
        .setTitle("Align Mesh")
        .setWindowFlags(Configuration::WindowFlag::Resizable)}
{
    Utility::Arguments args;
    args.addArgument("file").setHelp("file", "file to load")
        .addOption("importer", "AssimpImporter").setHelp("importer", "importer plugin to use")
        .addSkippedPrefix("magnum", "engine-specific options")
        .setGlobalHelp("Displays a 3D scene file provided on command line.")
        .parse(arguments.argc, arguments.argv);

    // IMGUI setup
    {
        _imgui = ImGuiIntegration::Context(Vector2{windowSize()}/dpiScaling(),
            windowSize(), framebufferSize());

        /* Set up proper blending to be used by ImGui. There's a great chance
        you'll need this exact behavior for the rest of your scene. If not, set
        this only for the drawFrame() call. */
        GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
            GL::Renderer::BlendEquation::Add);
        GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
        GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    }

    /* Every scene needs a camera */
    _cameraObject
        .setParent(&_scene)
        .translate(Vector3::zAxis(5.0f));
    (*(_camera = new SceneGraph::Camera3D{_cameraObject}))
        .setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(Matrix4::perspectiveProjection(35.0_degf, 1.0f, 0.01f, 1000.0f))
        .setViewport(GL::defaultFramebuffer.viewport().size());

    /* Base object, parent of all (for easy manipulation) */
    _manipulator.setParent(&_scene);
    _mesh.setParent(&_manipulator);

    // Show object axes
    {
        // Create some options
        DebugTools::ResourceManager::instance().set("my",
        DebugTools::ObjectRendererOptions{}.setSize(1.0f));

        // Create debug renderer for given object, use "my" options for it
        new DebugTools::ObjectRenderer3D(_manipulator, "my", &_drawables);
    }

    /* Setup renderer and shader defaults */
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    _coloredShader
        .setAmbientColor(0x111111_rgbf)
        .setSpecularColor(0xffffff_rgbf)
        .setShininess(80.0f);
    _texturedShader
        .setAmbientColor(0x111111_rgbf)
        .setSpecularColor(0x111111_rgbf)
        .setShininess(80.0f);

    // Can we load a pretransform?
    m_pretransformPath = args.value("file") + ".pretransform";
    if(fs::exists(m_pretransformPath))
    {
        readPretransform(m_pretransformPath);
    }

    /* Load a scene importer plugin */
    PluginManager::Manager<Trade::AbstractImporter> manager;
    Containers::Pointer<Trade::AbstractImporter> importer = manager.loadAndInstantiate(args.value("importer"));
    if(!importer) std::exit(1);

    // Set up postprocess options if using AssimpImporter
    auto group = importer->configuration().group("postprocess");
    if(group)
    {
        group->setValue("JoinIdenticalVertices", true);
        group->setValue("Triangulate", true);
        group->setValue("GenSmoothNormals", true);
        group->setValue("PreTransformVertices", true);
        group->setValue("SortByPType", true);
        group->setValue("GenUVCoords", true);
        group->setValue("TransformUVCoords", true);
    }

    Debug{} << "Opening file" << args.value("file");

    /* Load file */
    if(!importer->openFile(args.value("file")))
        std::exit(4);

    /* Load all textures. Textures that fail to load will be NullOpt. */
    _textures = Containers::Array<Containers::Optional<GL::Texture2D>>{importer->textureCount()};
    for(UnsignedInt i = 0; i != importer->textureCount(); ++i) {
        Debug{} << "Importing texture" << i << importer->textureName(i);

        Containers::Optional<Trade::TextureData> textureData = importer->texture(i);
        if(!textureData || textureData->type() != Trade::TextureData::Type::Texture2D) {
            Warning{} << "Cannot load texture properties, skipping";
            continue;
        }

        Debug{} << "Importing image" << textureData->image() << importer->image2DName(textureData->image());

        Containers::Optional<Trade::ImageData2D> imageData = importer->image2D(textureData->image());
        GL::TextureFormat format;
        if(imageData && imageData->format() == PixelFormat::RGB8Unorm)
            format = GL::TextureFormat::RGB8;
        else if(imageData && imageData->format() == PixelFormat::RGBA8Unorm)
            format = GL::TextureFormat::RGBA8;
        else {
            Warning{} << "Cannot load texture image, skipping";
            continue;
        }

        /* Configure the texture */
        GL::Texture2D texture;
        texture
            .setMagnificationFilter(textureData->magnificationFilter())
            .setMinificationFilter(textureData->minificationFilter(), textureData->mipmapFilter())
            .setWrapping(textureData->wrapping().xy())
            .setStorage(Math::log2(imageData->size().max()) + 1, format, imageData->size())
            .setSubImage(0, {}, *imageData)
            .generateMipmap();

        _textures[i] = std::move(texture);
    }

    /* Load all materials. Materials that fail to load will be NullOpt. The
       data will be stored directly in objects later, so save them only
       temporarily. */
    Containers::Array<Containers::Optional<Trade::PhongMaterialData>> materials{importer->materialCount()};
    for(UnsignedInt i = 0; i != importer->materialCount(); ++i) {
        Debug{} << "Importing material" << i << importer->materialName(i);

        Containers::Pointer<Trade::AbstractMaterialData> materialData = importer->material(i);
        if(!materialData || materialData->type() != Trade::MaterialType::Phong) {
            Warning{} << "Cannot load material, skipping";
            continue;
        }

        materials[i] = std::move(static_cast<Trade::PhongMaterialData&>(*materialData));
    }

    /* Load all meshes. Meshes that fail to load will be NullOpt. */
    _meshes = Containers::Array<Containers::Optional<GL::Mesh>>{importer->mesh3DCount()};
    for(UnsignedInt i = 0; i != importer->mesh3DCount(); ++i) {
        Debug{} << "Importing mesh" << i << importer->mesh3DName(i);

        Containers::Optional<Trade::MeshData3D> meshData = importer->mesh3D(i);
        if(!meshData || !meshData->hasNormals() || meshData->primitive() != MeshPrimitive::Triangles) {
            Warning{} << "Cannot load the mesh, skipping";
            continue;
        }

        /* Compile the mesh */
        _meshes[i] = MeshTools::compile(*meshData);
    }

    /* Load the scene */
    if(importer->defaultScene() != -1) {
        Debug{} << "Adding default scene" << importer->sceneName(importer->defaultScene());

        Containers::Optional<Trade::SceneData> sceneData = importer->scene(importer->defaultScene());
        if(!sceneData) {
            Error{} << "Cannot load scene, exiting";
            return;
        }

        /* Recursively add all children */
        for(UnsignedInt objectId: sceneData->children3D())
            addObject(*importer, materials, _mesh, objectId);

    /* The format has no scene support, display just the first loaded mesh with
       a default material and be done with it */
    } else if(!_meshes.empty() && _meshes[0])
        new ColoredDrawable{_mesh, _coloredShader, *_meshes[0], 0xffffff_rgbf, _drawables};
}

void AlignMesh::readPretransform(const std::string& path)
{
    std::ifstream stream(path);

    if(!stream)
    {
        Warning{} << "Could not read pretransform file";
        return;
    }

    Matrix4 pretransform;
    for(int i = 0; i < 4; ++i)
    {
        std::string line;
        if(!std::getline(stream, line))
        {
            Error{} << "Short pretransform file";
            return;
        }

        std::stringstream ss(line);
        ss.imbue(std::locale::classic());

        for(int j = 0; j < 4; ++j)
        {
            if(!(ss >> pretransform[j][i]))
            {
                Error{} << "Could not read number from pretransform file";
                return;
            }
        }
    }

    _mesh.setTransformation(pretransform);
}

void AlignMesh::writePretransform(const std::string& path)
{
    std::ofstream stream(path);

    if(!stream)
    {
        Error{} << "Could not write to pretransform file";
        return;
    }

    Matrix4 pretransform = _mesh.transformationMatrix();

    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            stream << pretransform[j][i];
            if(j != 4)
                stream << " ";
        }
        stream << "\n";
    }
}

void AlignMesh::addObject(Trade::AbstractImporter& importer, Containers::ArrayView<const Containers::Optional<Trade::PhongMaterialData>> materials, Object3D& parent, UnsignedInt i) {
    Debug{} << "Importing object" << i << importer.object3DName(i);
    Containers::Pointer<Trade::ObjectData3D> objectData = importer.object3D(i);
    if(!objectData) {
        Error{} << "Cannot import object, skipping";
        return;
    }

    /* Add the object to the scene and set its transformation */
    auto* object = new Object3D{&parent};
    object->setTransformation(objectData->transformation());

    /* Add a drawable if the object has a mesh and the mesh is loaded */
    if(objectData->instanceType() == Trade::ObjectInstanceType3D::Mesh && objectData->instance() != -1 && _meshes[objectData->instance()]) {
        const Int materialId = static_cast<Trade::MeshObjectData3D*>(objectData.get())->material();

        /* Material not available / not loaded, use a default material */
        if(materialId == -1 || !materials[materialId]) {
            new ColoredDrawable{*object, _coloredShader, *_meshes[objectData->instance()], 0xffffff_rgbf, _drawables};

        /* Textured material. If the texture failed to load, again just use a
           default colored material. */
        } else if(materials[materialId]->flags() & Trade::PhongMaterialData::Flag::DiffuseTexture) {
            Containers::Optional<GL::Texture2D>& texture = _textures[materials[materialId]->diffuseTexture()];
            if(texture)
                new TexturedDrawable{*object, _texturedShader, *_meshes[objectData->instance()], *texture, _drawables};
            else
                new ColoredDrawable{*object, _coloredShader, *_meshes[objectData->instance()], 0xffffff_rgbf, _drawables};

        /* Color-only material */
        } else {
            new ColoredDrawable{*object, _coloredShader, *_meshes[objectData->instance()], materials[materialId]->diffuseColor(), _drawables};
        }
    }

    /* Recursively add children */
    for(std::size_t id: objectData->children())
        addObject(importer, materials, *object, id);
}

void ColoredDrawable::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) {
    _shader
        .setDiffuseColor(_color)
        .setLightPosition(camera.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}))
        .setTransformationMatrix(transformationMatrix)
        .setNormalMatrix(transformationMatrix.rotationScaling())
        .setProjectionMatrix(camera.projectionMatrix());

    _mesh.draw(_shader);
}

void TexturedDrawable::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) {
    _shader
        .setLightPosition(camera.cameraMatrix().transformPoint({-3.0f, 10.0f, 10.0f}))
        .setTransformationMatrix(transformationMatrix)
        .setNormalMatrix(transformationMatrix.rotationScaling())
        .setProjectionMatrix(camera.projectionMatrix())
        .bindDiffuseTexture(_texture);

    _mesh.draw(_shader);
}

struct Axis
{
    std::string name;
    int index;

    Vector3 axis() const
    {
        Vector3 z;
        z[index] = 1.0f;
        return z;
    }
};

void AlignMesh::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color|GL::FramebufferClear::Depth);

    _camera->draw(_drawables);

    _imgui.newFrame();

    ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Align");
    ImGui::Text("Transform");
    ImGui::Indent();

    std::vector<Axis> AXES{
        {"X", 0},
        {"Y", 1},
        {"Z", 2}
    };
    for(auto& axis : AXES)
    {
        ImGui::PushID(axis.index);

        ImGui::Text("%s", axis.name.c_str());

        ImGui::SameLine();
        if(ImGui::Button("R+"))
        {
            Matrix4 rot = Matrix4::rotation(Rad(Constants::piHalf()), axis.axis());
            _mesh.setTransformation(rot * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("r+"))
        {
            Matrix4 rot = Matrix4::rotation(Deg(5.0f), axis.axis());
            _mesh.setTransformation(rot * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("r-"))
        {
            Matrix4 rot = Matrix4::rotation(Deg(-5.0f), axis.axis());
            _mesh.setTransformation(rot * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("R-"))
        {
            Matrix4 rot = Matrix4::rotation(Rad(-Constants::piHalf()), axis.axis());
            _mesh.setTransformation(rot * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("T++"))
        {
            Matrix4 T = Matrix4::translation(0.1f * axis.axis());
            _mesh.setTransformation(T * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("T+"))
        {
            Matrix4 T = Matrix4::translation(0.01f * axis.axis());
            _mesh.setTransformation(T * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("T-"))
        {
            Matrix4 T = Matrix4::translation(-0.01f * axis.axis());
            _mesh.setTransformation(T * _mesh.transformationMatrix());
        }

        ImGui::SameLine();
        if(ImGui::Button("T--"))
        {
            Matrix4 T = Matrix4::translation(-0.1f * axis.axis());
            _mesh.setTransformation(T * _mesh.transformationMatrix());
        }

        ImGui::PopID();
    }

    ImGui::Unindent();

    ImGui::Text("Scale");
    ImGui::Indent();
    if(ImGui::Button("S+"))
    {
        Matrix4 T = Matrix4::scaling(Vector3{10.0f});
        _mesh.setTransformation(T * _mesh.transformationMatrix());
    }

    ImGui::SameLine();
    if(ImGui::Button("s+"))
    {
        Matrix4 T = Matrix4::scaling(Vector3{1.5f});
        _mesh.setTransformation(T * _mesh.transformationMatrix());
    }

    ImGui::SameLine();
    if(ImGui::Button("s-"))
    {
        Matrix4 T = Matrix4::scaling(Vector3{1.0f/1.5f});
        _mesh.setTransformation(T * _mesh.transformationMatrix());
    }

    ImGui::SameLine();
    if(ImGui::Button("S-"))
    {
        Matrix4 T = Matrix4::scaling(Vector3{0.1f});
        _mesh.setTransformation(T * _mesh.transformationMatrix());
    }
    ImGui::Unindent();

    if(ImGui::Button("Save"))
    {
        writePretransform(m_pretransformPath);
        exit();
    }

    ImGui::End();

    /* Set appropriate states. If you only draw imgui UI, it is sufficient to
       do this once in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);

    _imgui.drawFrame();

    /* Reset state. Only needed if you want to draw something else with
       different state next frame. */
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);

    swapBuffers();
    redraw();
}

void AlignMesh::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
    _camera->setViewport(event.windowSize());

    _imgui.relayout(Vector2{event.windowSize()}/event.dpiScaling(),
        event.windowSize(), event.framebufferSize());
}

void AlignMesh::mousePressEvent(MouseEvent& event)
{
    if(_imgui.handleMousePressEvent(event)) return;

    if(event.button() == MouseEvent::Button::Left)
        _previousPosition = positionOnSphere(event.position());
}

void AlignMesh::mouseReleaseEvent(MouseEvent& event)
{
    if(_imgui.handleMouseReleaseEvent(event)) return;

    if(event.button() == MouseEvent::Button::Left)
        _previousPosition = Vector3();
}

void AlignMesh::mouseScrollEvent(MouseScrollEvent& event)
{
    if(_imgui.handleMouseScrollEvent(event)) return;

    if(!event.offset().y()) return;

    /* Distance to origin */
    const Float distance = _cameraObject.transformation().translation().z();

    /* Move 15% of the distance back or forward */
    _cameraObject.translate(Vector3::zAxis(
        distance*(1.0f - (event.offset().y() > 0 ? 1/0.85f : 0.85f))));

    redraw();
}

Vector3 AlignMesh::positionOnSphere(const Vector2i& position) const {
    const Vector2 positionNormalized = Vector2{position}/Vector2{_camera->viewport()} - Vector2{0.5f};
    const Float length = positionNormalized.length();
    const Vector3 result(length > 1.0f ? Vector3(positionNormalized, 0.0f) : Vector3(positionNormalized, 1.0f - length));
    return (result*Vector3::yScale(-1.0f)).normalized();
}

void AlignMesh::mouseMoveEvent(MouseMoveEvent& event)
{
    if(_imgui.handleMouseMoveEvent(event)) return;

    if(!(event.buttons() & MouseMoveEvent::Button::Left)) return;

    const Vector3 currentPosition = positionOnSphere(event.position());
    const Vector3 axis = Math::cross(_previousPosition, currentPosition);

    if(_previousPosition.length() < 0.001f || axis.length() < 0.001f) return;

    _manipulator.rotate(Math::angle(_previousPosition, currentPosition), axis.normalized());
    _previousPosition = currentPosition;

    redraw();
}

MAGNUM_APPLICATION_MAIN(AlignMesh)
