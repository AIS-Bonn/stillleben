// X11 event types (extracted from Magnum::Platform::AbstractXApplication)
// Extractor: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_UTILS_X11_EVENTS_H
#define SL_UTILS_X11_EVENTS_H

#include <Corrade/Containers/EnumSet.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
/* undef Xlib nonsense to avoid conflicts */
#undef Always
#undef Complex
#undef Convex
#undef None
#undef Status
#undef Success
#undef Button1
#undef Button2
#undef Button3
#undef Button4
#undef Button5

namespace sl
{
namespace utils
{

/**
@brief Viewport event

@see @ref viewportEvent()
*/
class ViewportEvent
{
    public:
        explicit ViewportEvent(const Magnum::Vector2i& size): _size{size} {}

        /** @brief Copying is not allowed */
        ViewportEvent(const ViewportEvent&) = delete;

        /** @brief Moving is not allowed */
        ViewportEvent(ViewportEvent&&) = delete;

        /** @brief Copying is not allowed */
        ViewportEvent& operator=(const ViewportEvent&) = delete;

        /** @brief Moving is not allowed */
        ViewportEvent& operator=(ViewportEvent&&) = delete;

        /**
         * @brief Window size
         *
         * Same as @ref framebufferSize().
         * @see @ref AbstractXApplication::windowSize()
         */
        Magnum::Vector2i windowSize() const { return _size; }

        /**
         * @brief Framebuffer size
         *
         * Same as @ref windowSize().
         * @see @ref AbstractXApplication::framebufferSize()
         */
        Magnum::Vector2i framebufferSize() const { return _size; }

    private:
        const Magnum::Vector2i _size;
};

/**
@brief Base for input events

@see @ref KeyEvent, @ref MouseEvent, @ref MouseMoveEvent, @ref keyPressEvent(),
    @ref keyReleaseEvent(), @ref mousePressEvent(), @ref mouseReleaseEvent(),
    @ref mouseMoveEvent()
*/
class InputEvent
{
    public:
        /**
         * @brief Modifier
         *
         * @see @ref Modifiers, @ref modifiers()
         */
        enum class Modifier: unsigned int {
            Shift = ShiftMask,          /**< Shift */
            Ctrl = ControlMask,         /**< Ctrl */
            Alt = Mod1Mask,             /**< Alt */
            AltGr = Mod5Mask,           /**< AltGr */
            CapsLock = LockMask,        /**< Caps lock */
            NumLock = Mod2Mask          /**< Num lock */
        };

        /**
         * @brief Set of modifiers
         *
         * @see @ref modifiers()
         */
        typedef Corrade::Containers::EnumSet<Modifier> Modifiers;

        /**
         * @brief Mouse button
         *
         * @see @ref Buttons, @ref buttons()
         */
        enum class Button: unsigned int {
            Left = Button1Mask,     /**< Left button */
            Middle = Button2Mask,   /**< Middle button */
            Right = Button3Mask     /**< Right button */
        };

        /**
         * @brief Set of mouse buttons
         *
         * @see @ref buttons()
         */
        typedef Corrade::Containers::EnumSet<Button> Buttons;

        /** @brief Copying is not allowed */
        InputEvent(const InputEvent&) = delete;

        /** @brief Moving is not allowed */
        InputEvent(InputEvent&&) = delete;

        /** @brief Copying is not allowed */
        InputEvent& operator=(const InputEvent&) = delete;

        /** @brief Moving is not allowed */
        InputEvent& operator=(InputEvent&&) = delete;

        /** @copydoc Sdl2Application::InputEvent::setAccepted() */
        void setAccepted(bool accepted = true) { _accepted = accepted; }

        /** @copydoc Sdl2Application::InputEvent::isAccepted() */
        bool isAccepted() const { return _accepted; }

        /** @brief Modifiers */
        Modifiers modifiers() const { return _modifiers; }

        /** @brief Mouse buttons */
        Buttons buttons() const { return Button(static_cast<unsigned int>(_modifiers)); }

    #ifndef DOXYGEN_GENERATING_OUTPUT
    protected:
        explicit InputEvent(Modifiers modifiers): _modifiers(modifiers), _accepted(false) {}

        ~InputEvent() = default;
    #endif

    private:
        Modifiers _modifiers;
        bool _accepted;
};

CORRADE_ENUMSET_OPERATORS(InputEvent::Modifiers)
CORRADE_ENUMSET_OPERATORS(InputEvent::Buttons)

/**
@brief Key event

@see @ref keyPressEvent(), @ref keyReleaseEvent()
*/
class KeyEvent : public InputEvent
{
    public:
        /**
         * @brief Key
         *
         * @see @ref key()
         */
        enum class Key: KeySym {
            Enter = XK_Return,          /**< Enter */
            Esc = XK_Escape,            /**< Escape */

            Up = XK_Up,                 /**< Up arrow */
            Down = XK_Down,             /**< Down arrow */
            Left = XK_Left,             /**< Left arrow */
            Right = XK_Right,           /**< Right arrow */
            F1 = XK_F1,                 /**< F1 */
            F2 = XK_F2,                 /**< F2 */
            F3 = XK_F3,                 /**< F3 */
            F4 = XK_F4,                 /**< F4 */
            F5 = XK_F5,                 /**< F5 */
            F6 = XK_F6,                 /**< F6 */
            F7 = XK_F7,                 /**< F7 */
            F8 = XK_F8,                 /**< F8 */
            F9 = XK_F9,                 /**< F9 */
            F10 = XK_F10,               /**< F10 */
            F11 = XK_F11,               /**< F11 */
            F12 = XK_F12,               /**< F12 */
            Home = XK_Home,             /**< Home */
            End = XK_End,               /**< End */
            PageUp = XK_Page_Up,        /**< Page up */
            PageDown = XK_Page_Down,    /**< Page down */

            Space = XK_space,           /**< Space */
            Comma = XK_comma,           /**< Comma */
            Period = XK_period,         /**< Period */
            Minus = XK_minus,           /**< Minus */
            Plus = XK_plus,             /**< Plus */
            Slash = XK_slash,           /**< Slash */
            Percent = XK_percent,       /**< Percent */
            Equal = XK_equal,           /**< Equal */

            Zero = XK_0,                /**< Zero */
            One = XK_1,                 /**< One */
            Two = XK_2,                 /**< Two */
            Three = XK_3,               /**< Three */
            Four = XK_4,                /**< Four */
            Five = XK_5,                /**< Five */
            Six = XK_6,                 /**< Six */
            Seven = XK_7,               /**< Seven */
            Eight = XK_8,               /**< Eight */
            Nine = XK_9,                /**< Nine */

            A = XK_a,                   /**< Small letter A */
            B = XK_b,                   /**< Small letter B */
            C = XK_c,                   /**< Small letter C */
            D = XK_d,                   /**< Small letter D */
            E = XK_e,                   /**< Small letter E */
            F = XK_f,                   /**< Small letter F */
            G = XK_g,                   /**< Small letter G */
            H = XK_h,                   /**< Small letter H */
            I = XK_i,                   /**< Small letter I */
            J = XK_j,                   /**< Small letter J */
            K = XK_k,                   /**< Small letter K */
            L = XK_l,                   /**< Small letter L */
            M = XK_m,                   /**< Small letter M */
            N = XK_n,                   /**< Small letter N */
            O = XK_o,                   /**< Small letter O */
            P = XK_p,                   /**< Small letter P */
            Q = XK_q,                   /**< Small letter Q */
            R = XK_r,                   /**< Small letter R */
            S = XK_s,                   /**< Small letter S */
            T = XK_t,                   /**< Small letter T */
            U = XK_u,                   /**< Small letter U */
            V = XK_v,                   /**< Small letter V */
            W = XK_w,                   /**< Small letter W */
            X = XK_x,                   /**< Small letter X */
            Y = XK_y,                   /**< Small letter Y */
            Z = XK_z                    /**< Small letter Z */
        };

        explicit KeyEvent(Key key, Modifiers modifiers, const Magnum::Vector2i& position)
         : InputEvent(modifiers), _key(key), _position(position) {}

        /** @brief Key */
        Key key() const { return _key; }

        /** @brief Position */
        Magnum::Vector2i position() const { return _position; }

    private:
        const Key _key;
        const Magnum::Vector2i _position;
};

/**
@brief Mouse event

@see @ref MouseMoveEvent, @ref mousePressEvent(), @ref mouseReleaseEvent()
*/
class MouseEvent : public InputEvent
{
    public:
        /**
         * @brief Mouse button
         *
         * @see @ref button()
         */
        enum class Button: unsigned int {
            Left      = 1 /*Button1*/,  /**< Left button */
            Middle    = 2 /*Button2*/,  /**< Middle button */
            Right     = 3 /*Button3*/,  /**< Right button */
            WheelUp   = 4 /*Button4*/,  /**< Wheel up */
            WheelDown = 5 /*Button5*/   /**< Wheel down */
        };

        explicit MouseEvent(Button button, Modifiers modifiers, const Magnum::Vector2i& position)
         : InputEvent(modifiers), _button(button), _position(position) {}

        /** @brief Button */
        Button button() const { return _button; }

        /** @brief Position */
        Magnum::Vector2i position() const { return _position; }

    private:
        const Button _button;
        const Magnum::Vector2i _position;
};

/**
@brief Mouse move event

@see @ref MouseEvent, @ref mouseMoveEvent()
*/
class MouseMoveEvent : public InputEvent
{
    public:
        explicit MouseMoveEvent(Modifiers modifiers, const Magnum::Vector2i& position): InputEvent(modifiers), _position(position) {}

        /** @brief Position */
        Magnum::Vector2i position() const { return _position; }

    private:
        const Magnum::Vector2i _position;
};

}
}

#endif
