// Basic Magnum types binding
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef SL_PY_MAGNUM_H
#define SL_PY_MAGNUM_H

#include <torch/extension.h>

#include <Corrade/Containers/Array.h>

#include <Magnum/ImageView.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Range.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/PixelFormat.h>

#include <stillleben/context.h>
#include <stillleben/cuda_interop.h>

#include "py_context.h"

namespace sl
{
namespace python
{
namespace magnum
{

// Magnum -> Torch
template<class T>
struct toTorch
{
    using Result = const T&;
    static const T& convert(const T& t)
    { return t; }
};

template<>
struct toTorch<void>
{
    using Result = void;
};

template<>
struct toTorch<Magnum::Matrix4>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Matrix4& mat)
    {
        auto tensor = torch::from_blob(
            const_cast<float*>(mat.data()),
            {4,4},
            at::kFloat
        );

        // NOTE: Magnum matrices are column-major
        return tensor.t().clone();
    }
};

template<>
struct toTorch<Magnum::Matrix3>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Matrix3& mat)
    {
        auto tensor = torch::from_blob(
            const_cast<float*>(mat.data()),
            {3,3},
            at::kFloat
        );

        // NOTE: Magnum matrices are column-major
        return tensor.t().clone();
    }
};

template<>
struct toTorch<Magnum::Vector2>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Vector2& vec)
    {
        return torch::from_blob(const_cast<float*>(vec.data()), {2}, at::kFloat).clone();
    }
};

template<>
struct toTorch<Magnum::Vector3>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Vector3& vec)
    {
        return torch::from_blob(const_cast<float*>(vec.data()), {3}, at::kFloat).clone();
    }
};

template<>
struct toTorch<Magnum::Vector4>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Vector4& vec)
    {
        return torch::from_blob(const_cast<float*>(vec.data()), {4}, at::kFloat).clone();
    }
};

template<>
struct toTorch<Corrade::Containers::Array<Magnum::Vector3>>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Corrade::Containers::Array<Magnum::Vector3>& vec)
    {
        return torch::from_blob(
            const_cast<float*>(reinterpret_cast<const float*>(vec.data())),
            {static_cast<long int>(vec.size()) * 3},
            at::kFloat
        ).clone();
    }
};

template<>
struct toTorch<Corrade::Containers::Array<Magnum::Color4>>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Corrade::Containers::Array<Magnum::Color4>& vec)
    {
        return torch::from_blob(
            const_cast<float*>(reinterpret_cast<const float*>(vec.data())),
            {static_cast<long int>(vec.size()) * 4},
            at::kFloat
        ).clone();
    }
};

template<>
struct toTorch<Corrade::Containers::Array<Magnum::UnsignedInt>>
{
    using Result = at::Tensor;
    static_assert(sizeof(int) == sizeof(Magnum::UnsignedInt), "Mismatch between PyTorch and Magnum int sizes");

    static at::Tensor convert(const Corrade::Containers::Array<Magnum::UnsignedInt>& vec)
    {
        return torch::from_blob(
            const_cast<int*>(reinterpret_cast<const int*>(vec.data())),
            {static_cast<int>(vec.size())},
            at::kInt
        ).clone();
    }
};

template<>
struct toTorch<Magnum::Color4>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Color4& vec)
    {
        return toTorch<Magnum::Vector4>::convert(vec);
    }
};

template<>
struct toTorch<Magnum::Color3>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Color3& vec)
    {
        return toTorch<Magnum::Vector3>::convert(vec);
    }
};

template<>
struct toTorch<Magnum::Range2D>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Range2D& range)
    {
        return toTorch<Magnum::Vector4>::convert({
            range.min().x(), range.min().y(),
            range.max().x(), range.max().y()
        });
    }
};

template<>
struct toTorch<Magnum::Quaternion>
{
    using Result = at::Tensor;
    static at::Tensor convert(const Magnum::Quaternion& q)
    {
        return toTorch<Magnum::Vector4>::convert({q.vector(), q.scalar()});
    }
};

// Torch -> Magnum
template<class T>
struct fromTorch
{
    using Type = T;
    static T convert(const T& t)
    { return t; }
};

template<>
struct fromTorch<Magnum::Matrix4>
{
    using Type = at::Tensor;
    static Magnum::Matrix4 convert(const at::Tensor& tensor)
    {
        auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous().t().contiguous();
        if(cpuTensor.dim() != 2 || cpuTensor.size(0) != 4 || cpuTensor.size(1) != 4)
            throw std::invalid_argument("A pose tensor must be 4x4");

        const float* data = cpuTensor.data_ptr<float>();

        Magnum::Matrix4 mat{Magnum::Math::NoInit};

        memcpy(mat.data(), data, 16*sizeof(float));

        return mat;
    }
};

template<>
struct fromTorch<Magnum::Matrix3>
{
    using Type = at::Tensor;
    static Magnum::Matrix3 convert(const at::Tensor& tensor)
    {
        auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous().t().contiguous();
        if(cpuTensor.dim() != 2 || cpuTensor.size(0) != 3 || cpuTensor.size(1) != 3)
            throw std::invalid_argument("An orientation tensor must be 3x3");

        const float* data = cpuTensor.data_ptr<float>();

        Magnum::Matrix3 mat{Magnum::Math::NoInit};

        memcpy(mat.data(), data, 9*sizeof(float));

        return mat;
    }
};

template<>
struct fromTorch<Magnum::Vector2>
{
    using Type = at::Tensor;
    static Magnum::Vector2 convert(const at::Tensor& tensor)
    {
        auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous();
        if(cpuTensor.dim() != 1 || cpuTensor.size(0) != 2)
            throw std::invalid_argument("A 2D vector tensor must have size 2");

        const float* data = cpuTensor.data_ptr<float>();
        Magnum::Vector2 vec{Magnum::Math::NoInit};
        memcpy(vec.data(), data, 2*sizeof(float));

        return vec;
    }
};

template<>
struct fromTorch<Magnum::Vector3>
{
    using Type = at::Tensor;
    static Magnum::Vector3 convert(const at::Tensor& tensor)
    {
        auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous();
        if(cpuTensor.dim() != 1 || cpuTensor.size(0) != 3)
            throw std::invalid_argument("A vector tensor must have size 3");

        const float* data = cpuTensor.data_ptr<float>();
        Magnum::Vector3 vec{Magnum::Math::NoInit};
        memcpy(vec.data(), data, 3*sizeof(float));

        return vec;
    }
};

template<>
struct fromTorch<Magnum::Vector4>
{
    using Type = at::Tensor;
    static Magnum::Vector4 convert(const at::Tensor& tensor)
    {
        auto cpuTensor = tensor.to(at::kFloat).cpu().contiguous();
        if(cpuTensor.dim() != 1 || cpuTensor.size(0) != 4)
            throw std::invalid_argument("A vector4 tensor must have size 4");

        const float* data = cpuTensor.data_ptr<float>();
        Magnum::Vector4 vec{Magnum::Math::NoInit};
        memcpy(vec.data(), data, 4*sizeof(float));

        return vec;
    }
};

template<>
struct fromTorch<Magnum::Color4>
{
    using Type = at::Tensor;
    static Magnum::Color4 convert(const at::Tensor& tensor)
    {
        return fromTorch<Magnum::Vector4>::convert(tensor);
    }
};

template<>
struct fromTorch<Magnum::Color3>
{
    using Type = at::Tensor;
    static Magnum::Color3 convert(const at::Tensor& tensor)
    {
        return fromTorch<Magnum::Vector3>::convert(tensor);
    }
};

template<>
struct fromTorch<Magnum::Range2D>
{
    using Type = at::Tensor;
    static Magnum::Range2D convert(const at::Tensor& tensor)
    {
        auto vec = fromTorch<Magnum::Vector4>::convert(tensor);
        return Magnum::Range2D{vec.xy(), Magnum::Vector2{vec.z(), vec.w()}};
    }
};

template<>
struct fromTorch<Magnum::Quaternion>
{
    using Type = at::Tensor;
    static Magnum::Quaternion convert(const at::Tensor& tensor)
    {
        auto vec = fromTorch<Magnum::Vector4>::convert(tensor);
        return Magnum::Quaternion{vec.xyz(), vec.w()};
    }
};

// Automatic wrapping
template<class T, class R, class ... Args>
std::function<typename toTorch<std::decay_t<R>>::Result (const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapShared(R (T::*fun)(Args...) const)
{
    using RConv = toTorch<std::decay_t<R>>;
    return [=](const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
        return RConv::convert((obj.get()->*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
    };
}
template<class T, class R, class ... Args>
std::function<typename toTorch<std::decay_t<R>>::Result (const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapShared(R (T::*fun)(Args...))
{
    using RConv = toTorch<std::decay_t<R>>;
    return [=](const std::shared_ptr<T>& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) -> typename toTorch<std::decay_t<R>>::Result {
        if constexpr(std::is_same_v<R, void>)
            (obj.get()->*fun)(fromTorch<std::decay_t<Args>>::convert(args)...);
        else
            return RConv::convert((obj.get()->*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
    };
}

template<class T, class R, class ... Args>
std::function<typename toTorch<std::decay_t<R>>::Result (T& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapRef(R (T::*fun)(Args...) const)
{
    using RConv = toTorch<std::decay_t<R>>;
    return [=](T& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
        return RConv::convert((obj.*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
    };
}
template<class T, class R, class ... Args>
std::function<typename toTorch<std::decay_t<R>>::Result (T& obj, typename fromTorch<std::decay_t<Args>>::Type...)> wrapRef(R (T::*fun)(Args...))
{
    using RConv = toTorch<std::decay_t<R>>;
    return [=](T& obj, typename fromTorch<std::decay_t<Args>>::Type ... args) {
        if constexpr(std::is_same_v<R, void>)
            (obj.*fun)(fromTorch<std::decay_t<Args>>::convert(args)...);
        else
            return RConv::convert((obj.*fun)(fromTorch<std::decay_t<Args>>::convert(args)...));
    };
}

at::Tensor extract(sl::CUDATexture& texture, Magnum::PixelFormat format, int channels, const torch::TensorOptions& opts);

void init(py::module& m);

}
}
}

#endif
