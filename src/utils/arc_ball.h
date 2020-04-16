#ifndef SL_UTILS_ARC_BALL_H
#define SL_UTILS_ARC_BALL_H

// Imported from magnum-examples, original header follows

/*
    This file is part of Magnum.

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2020 — Nghia Truong <nghiatruong.vn@gmail.com>

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

#include <Magnum/Magnum.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Math/DualQuaternion.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>

namespace sl { namespace utils {

/* Implementation of Ken Shoemake's arcball camera with smooth navigation
   feature: https://www.talisman.org/~erlkonig/misc/shoemake92-arcball.pdf */
class ArcBall {
    public:
        ArcBall(const Magnum::Vector3& cameraPosition, const Magnum::Vector3& viewCenter,
            const Magnum::Vector3& upDir, Magnum::Deg fov, const Magnum::Vector2i& windowSize);

        /* Set the camera view parameters: eye position, view center, up
           direction */
        void setViewParameters(const Magnum::Vector3& eye, const Magnum::Vector3& viewCenter,
            const Magnum::Vector3& upDir);

        /* Reset the camera to its initial position, view center, and up dir */
        void reset();

        /* Update screen size after the window has been resized */
        void reshape(const Magnum::Vector2i& windowSize) { _windowSize = windowSize; }

        /* Update any unfinished transformation due to lagging, return true if
           the camera matrices have changed */
        bool updateTransformation();

        /* Get/set the amount of lagging such that the camera will (slowly)
           smoothly navigate. Lagging must be in [0, 1) */
        Magnum::Float lagging() const { return _lagging; }
        void setLagging(Magnum::Float lagging);

        /* Initialize the first (screen) mouse position for camera
           transformation. This should be called in mouse pressed event. */
        void initTransformation(const Magnum::Vector2i& mousePos);

        /* Rotate the camera from the previous (screen) mouse position to the
           current (screen) position */
        void rotate(const Magnum::Vector2i& mousePos);

        /* Translate the camera from the previous (screen) mouse position to
           the current (screen) mouse position */
        void translate(const Magnum::Vector2i& mousePos);

        /* Translate the camera by the delta amount of (NDC) mouse position.
           Note that NDC position must be in [-1, -1] to [1, 1]. */
        void translateDelta(const Magnum::Vector2& translationNDC);

        /* Zoom the camera (positive delta = zoom in, negative = zoom out) */
        void zoom(Magnum::Float delta);

        /* Get the camera's view transformation as a qual quaternion */
        const Magnum::DualQuaternion& view() const { return _view; }

        /* Get the camera's view transformation as a matrix */
        Magnum::Matrix4 viewMatrix() const { return _view.toMatrix(); }

        /* Get the camera's inverse view matrix (which also produces
           transformation of the camera) */
        Magnum::Matrix4 inverseViewMatrix() const { return _inverseView.toMatrix(); }

        /* Get the camera's transformation as a dual quaternion */
        const Magnum::DualQuaternion& transformation() const { return _inverseView; }

        /* Get the camera's transformation matrix */
        Magnum::Matrix4 transformationMatrix() const { return _inverseView.toMatrix(); }

        /* Return the distance from the camera position to the center view */
        Magnum::Float viewDistance() const { return Magnum::Math::abs(_targetZooming); }

    protected:
        /* Update the camera transformations */
        void updateInternalTransformations();

        /* Transform from screen coordinate to NDC - normalized device
           coordinate. The top-left of the screen corresponds to [-1, 1] NDC,
           and the bottom right is [1, -1] NDC. */
        Magnum::Vector2 screenCoordToNDC(const Magnum::Vector2i& mousePos) const;

        Magnum::Deg _fov;
        Magnum::Vector2i _windowSize;

        Magnum::Vector2 _prevMousePosNDC;
        Magnum::Float _lagging{};

        Magnum::Vector3 _targetPosition, _currentPosition, _positionT0;
        Magnum::Quaternion _targetQRotation, _currentQRotation, _qRotationT0;
        Magnum::Float _targetZooming, _currentZooming, _zoomingT0;
        Magnum::DualQuaternion _view, _inverseView;
};

}}

#endif
