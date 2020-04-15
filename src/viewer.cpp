// Interactive scene viewer
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "viewer.h"

#include <stillleben/context.h>
#include <stillleben/render_pass.h>
#include <stillleben/scene.h>

#include <Corrade/Utility/Debug.h>

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

using namespace Magnum;

namespace sl
{

class Viewer::Private
{
public:
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
};

Viewer::Viewer(const std::shared_ptr<Context>& ctx)
 : m_d{std::make_unique<Private>(ctx)}
{
}

Viewer::~Viewer()
{
    if(m_d->surface)
        eglDestroySurface(eglGetCurrentDisplay(), m_d->surface);

    XDestroyWindow(m_d->display, m_d->window);
    XCloseDisplay(m_d->display);
}

void Viewer::setScene(const std::shared_ptr<Scene>& scene)
{
    m_d->scene = scene;
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
        throw std::runtime_error{"Viewer: cannot get X visual"};

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
}

void Viewer::draw()
{
    if(!m_d->scene)
        return;

    m_d->result = m_d->renderer->render(*m_d->scene, m_d->result);

    eglSwapBuffers(eglGetCurrentDisplay(), m_d->surface);
}


}
