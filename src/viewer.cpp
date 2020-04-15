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

    Magnum::GL::Framebuffer framebuffer{Magnum::Range2Di{{}, {800,600}}};
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

    auto windowSize = m_d->scene->viewport();

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
    m_d->window = XCreateWindow(m_d->display, root, 20, 20, windowSize.x(), windowSize.y(), 0, visInfo->depth, InputOutput, visInfo->visual, mask, &attr);
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

    Debug{} << "viewport:" << windowSize;
    Magnum::GL::defaultFramebuffer.setViewport({{}, windowSize});

    m_d->flags |= Private::Flag::Redraw;
}

void Viewer::draw()
{
    if(!m_d->scene)
        return;

    eglMakeCurrent(eglGetCurrentDisplay(), m_d->surface, m_d->surface, eglGetCurrentContext());

    m_d->result = m_d->renderer->render(*m_d->scene, m_d->result);

    eglMakeCurrent(eglGetCurrentDisplay(), m_d->surface, m_d->surface, eglGetCurrentContext());

    m_d->framebuffer.attachTexture(GL::Framebuffer::ColorAttachment{0}, m_d->result->rgb);
    m_d->framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{0});

    Magnum::GL::defaultFramebuffer.bind();
    Magnum::GL::defaultFramebuffer.clear(GL::FramebufferClear::Color|
                                 GL::FramebufferClear::Depth);
    Magnum::GL::defaultFramebuffer.mapForDraw(Magnum::GL::DefaultFramebuffer::DrawAttachment::Back);

    Debug{} << "source" << m_d->framebuffer.viewport() << "dest" << GL::defaultFramebuffer.viewport();

    Magnum::GL::Framebuffer::blit(m_d->framebuffer, GL::defaultFramebuffer,
        m_d->framebuffer.viewport(),
        GL::defaultFramebuffer.viewport(),
        Magnum::GL::FramebufferBlit::Color,
        Magnum::GL::FramebufferBlitFilter::Linear
    );

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
#warning FIXME
//                 Vector2i size(event.xconfigure.width, event.xconfigure.height);
//                 if(size != m_d->windowSize) {
//                     m_d->windowSize = size;
//                     ViewportEvent e{size};
//                     viewportEvent(e);
//                     _flags |= Flag::Redraw;
//                 }
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
