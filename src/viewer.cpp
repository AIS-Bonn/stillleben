// Interactive scene viewer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/viewer.h>

#include <stillleben/context.h>
#include <stillleben/render_pass.h>
#include <stillleben/scene.h>

#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/System.h>
#include <Corrade/Utility/FormatStl.h>

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/TextureFormat.h>

#include <Magnum/ImGuiIntegration/Context.h>
#include <Magnum/ImGuiIntegration/Integration.h>
#include <Magnum/ImGuiIntegration/Widgets.h>
#include <imgui.h>

#include <Egl.h>

#include <EGL/egl.h>
/* undef Xlib nonsense to avoid conflicts */
#undef None
#undef Complex

/* EGL returns visual ID as int, but Xorg expects long unsigned int */
#ifdef __unix__
typedef VisualID VisualId;
#else
typedef EGLInt VisualId;
#endif

/* Mask for X events */
#define INPUT_MASK KeyPressMask|KeyReleaseMask|ButtonPressMask|ButtonReleaseMask|PointerMotionMask|StructureNotifyMask

using namespace Magnum;

namespace sl
{

class Viewer::Private
{
public:
    enum class Flag: unsigned int {
        Redraw = 1 << 0,
        Exit = 1 << 1
    };

    typedef Containers::EnumSet<Flag> Flags;
    CORRADE_ENUMSET_FRIEND_OPERATORS(Flags)


    explicit Private(const std::shared_ptr<Context>& ctx)
     : ctx{ctx}
     , renderer{std::make_unique<RenderPass>(RenderPass::Type::Phong, false)}
    {}

    std::shared_ptr<Context> ctx;
    std::unique_ptr<RenderPass> renderer;
    std::shared_ptr<Scene> scene;
    std::shared_ptr<RenderPass::Result> result;

    Display* display{};
    Window window{};
    Atom deleteWindow{};

    EGLSurface surface{};

    Flags flags{};

    Magnum::GL::Framebuffer framebuffer{Magnum::NoCreate};
    Magnum::GL::Texture2D textureRGB{Magnum::NoCreate};
    Magnum::Vector2i windowSize{};

    Magnum::ImGuiIntegration::Context imgui{Magnum::NoCreate};
};

Viewer::Viewer(const std::shared_ptr<Context>& ctx)
 : m_d{std::make_unique<Private>(ctx)}
{
}

Viewer::~Viewer()
{
    if(m_d->surface)
        eglDestroySurface(eglGetCurrentDisplay(), m_d->surface);

    if(m_d->window)
        XDestroyWindow(m_d->display, m_d->window);

    if(m_d->display)
        XCloseDisplay(m_d->display);
}

void Viewer::setScene(const std::shared_ptr<Scene>& scene)
{
    m_d->scene = scene;
}

std::shared_ptr<Scene> Viewer::scene() const
{
    return m_d->scene;
}

void Viewer::setup()
{
    if(!m_d->scene)
        throw std::logic_error{"Need to call Viewer::setScene() first"};

    m_d->windowSize = {1280, 720};

    // Get default X display
    m_d->display = XOpenDisplay(nullptr);

    VisualId visualId = m_d->ctx->visualID();

    /* Get visual info */
    XVisualInfo *visInfo, visTemplate;
    int visualCount;
    visTemplate.visualid = visualId;
    visInfo = XGetVisualInfo(m_d->display, VisualIDMask, &visTemplate, &visualCount);
    if(!visInfo)
    {
        throw std::runtime_error{
            Corrade::Utility::formatString("Viewer: cannot get X visual using visualid {}", visualId)
        };
    }

    /* Create X Window */
    Window root = RootWindow(m_d->display, DefaultScreen(m_d->display));
    XSetWindowAttributes attr;
    attr.background_pixel = 0;
    attr.border_pixel = 0;
    attr.colormap = XCreateColormap(m_d->display, root, visInfo->visual, AllocNone);
    attr.event_mask = 0;
    unsigned long mask = CWBackPixel|CWBorderPixel|CWColormap|CWEventMask;
    m_d->window = XCreateWindow(m_d->display, root, 20, 20, m_d->windowSize.x(), m_d->windowSize.y(), 0, visInfo->depth, InputOutput, visInfo->visual, mask, &attr);
    XSetStandardProperties(m_d->display, m_d->window, "stillleben", nullptr, 0, nullptr, 0, nullptr);
    XFree(visInfo);

    /* Be notified about closing the window */
    m_d->deleteWindow = XInternAtom(m_d->display, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(m_d->display, m_d->window, &m_d->deleteWindow, 1);

    // Create the EGL surface
    m_d->surface = eglCreateWindowSurface(
        eglGetCurrentDisplay(), m_d->ctx->eglConfig(), m_d->window, nullptr
    );
    if(!m_d->surface)
        throw std::runtime_error{std::string{"Cannot create window surface:"} + Platform::Implementation::eglErrorString(eglGetError())};

    // Make current
    eglMakeCurrent(eglGetCurrentDisplay(), m_d->surface, m_d->surface, eglGetCurrentContext());

    Magnum::GL::defaultFramebuffer.setViewport({{}, m_d->windowSize});

    m_d->imgui = Magnum::ImGuiIntegration::Context{m_d->windowSize};

    /* Set up proper blending to be used by ImGui. There's a great chance
       you'll need this exact behavior for the rest of your scene. If not, set
       this only for the drawFrame() call. */
    GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
        GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
        GL::Renderer::BlendFunction::OneMinusSourceAlpha);

    // Setup our intermediate framebuffer
    auto size = m_d->scene->viewport();
    m_d->framebuffer = Magnum::GL::Framebuffer{{{}, size}};
    m_d->textureRGB = Magnum::GL::Texture2D{};
    m_d->textureRGB
        .setMagnificationFilter(GL::SamplerFilter::Linear)
        .setMinificationFilter(GL::SamplerFilter::Linear, GL::SamplerMipmap::Linear)
        .setWrapping(GL::SamplerWrapping::ClampToEdge)
        .setMaxAnisotropy(GL::Sampler::maxMaxAnisotropy())
        .setStorage(Math::log2(size.max())+1, GL::TextureFormat::RGBA8, size)
    ;

    m_d->flags |= Private::Flag::Redraw;
}

void Viewer::draw()
{
    m_d->result = m_d->renderer->render(*m_d->scene, m_d->result);

    // Get the result into a Texture2D
    m_d->framebuffer.attachTexture(GL::Framebuffer::ColorAttachment{0}, m_d->result->rgb);
    m_d->framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{0});

    m_d->framebuffer.copySubImage({{}, m_d->scene->viewport()}, m_d->textureRGB, 0, {});
    m_d->textureRGB.generateMipmap();

    Magnum::GL::defaultFramebuffer.bind();
    Magnum::GL::defaultFramebuffer.mapForDraw(Magnum::GL::DefaultFramebuffer::DrawAttachment::Back);
    Magnum::GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);

    if(!m_d->scene)
        return;

    m_d->imgui.newFrame();

    Vector2 srcSize{m_d->scene->viewport()};
    Vector2 qSize = Vector2{(m_d->windowSize / 2)};

    auto fitImage = [&](Magnum::GL::Texture2D& tex){
        auto available = Vector2{ImGui::GetWindowContentRegionMax()} - Vector2{ImGui::GetWindowContentRegionMin()};

        float scale = (available / srcSize).min();
        Vector2 imgSize = scale * srcSize;

        Magnum::ImGuiIntegration::image(tex, imgSize, {{0.0, 1.0}, {1.0, 0.0}});
    };

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));

    {
        ImGui::SetNextWindowPos(ImVec2{0,0});
        ImGui::SetNextWindowSize(ImVec2{qSize});
        ImGui::Begin("RGB");
        fitImage(m_d->textureRGB);
        ImGui::End();
    }

    ImGui::PopStyleVar();

    /* Set appropriate states. If you only draw ImGui, it is sufficient to
       just enable blending and scissor test in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    m_d->imgui.drawFrame();

    /* Reset state. Only needed if you want to draw something else with
       different state after. */
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);

    eglSwapBuffers(eglGetCurrentDisplay(), m_d->surface);

    m_d->flags |= Private::Flag::Redraw;
}

void Viewer::run()
{
    setup();

    // Show window
    XMapWindow(m_d->display, m_d->window);

    while(mainLoopIteration()) {}
}

bool Viewer::mainLoopIteration()
{
    XEvent event;

    // Closed window
    if(XCheckTypedWindowEvent(m_d->display, m_d->window, ClientMessage, &event) &&
        Atom(event.xclient.data.l[0]) == m_d->deleteWindow)
    {
        return false;
    }

    while(XCheckWindowEvent(m_d->display, m_d->window, INPUT_MASK, &event)) {
        switch(event.type) {
            /* Window resizing */
            case ConfigureNotify: {
                Vector2i size(event.xconfigure.width, event.xconfigure.height);
                if(size != m_d->windowSize)
                {
                    m_d->windowSize = size;
                    Magnum::GL::defaultFramebuffer.setViewport({{}, size});
                    m_d->framebuffer.setViewport({{}, size});
                    m_d->flags |= Private::Flag::Redraw;
                }
            } break;

            /* Key/mouse events */
            case KeyPress:
            case KeyRelease: {
//                 KeyEvent e(static_cast<KeyEvent::Key>(XLookupKeysym(&event.xkey, 0)), static_cast<InputEvent::Modifier>(event.xkey.state), {event.xkey.x, event.xkey.y});
//                 event.type == KeyPress ? keyPressEvent(e) : keyReleaseEvent(e);
            } break;
            case ButtonPress:
            case ButtonRelease: {
//                 MouseEvent e(static_cast<MouseEvent::Button>(event.xbutton.button), static_cast<InputEvent::Modifier>(event.xkey.state), {event.xbutton.x, event.xbutton.y});
//                 event.type == ButtonPress ? mousePressEvent(e) : mouseReleaseEvent(e);
            } break;

            /* Mouse move events */
            case MotionNotify: {
//                 MouseMoveEvent e(static_cast<InputEvent::Modifier>(event.xmotion.state), {event.xmotion.x, event.xmotion.y});
//                 mouseMoveEvent(e);
            } break;
        }
    }

    if(m_d->flags & Private::Flag::Redraw) {
        m_d->flags &= ~Private::Flag::Redraw;
        draw();
    } else Corrade::Utility::System::sleep(5);

    return !(m_d->flags & Private::Flag::Exit);
}

}
