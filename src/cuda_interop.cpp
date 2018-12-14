// CUDA integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/cuda_interop.h>

#if HAVE_CUDA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#pragma GCC diagnostic pop
#endif

using namespace Magnum;

namespace sl
{

class CUDAMapper::Private
{
public:
#if HAVE_CUDA
    explicit Private(GL::RectangleTexture& texture, std::size_t bytesPerPixel)
     : texture(texture)
     , bytesPerPixel(bytesPerPixel)
    {}

    GL::RectangleTexture& texture;
    std::size_t bytesPerPixel;
    cudaGraphicsResource* cuda_resource = nullptr;
#endif
};

CUDAMapper::CUDAMapper(GL::RectangleTexture& texture, std::size_t bytesPerPixel)
#if HAVE_CUDA
 : m_d(new Private(texture, bytesPerPixel))
#endif
{
#if HAVE_CUDA
    auto err = cudaGraphicsGLRegisterImage(&m_d->cuda_resource, texture.id(), GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsReadOnly);
    if(err != cudaSuccess)
    {
        Error{} << "Could not register texture with CUDA:" << cudaGetErrorString(err);
        std::abort();
    }

    if(cudaGraphicsMapResources(1, &m_d->cuda_resource) != cudaSuccess)
    {
        Error{} << "Could not map texture for CUDA";
        std::abort();
    }
#else
    throw std::runtime_error("stillleben was compiled without CUDA interop");
#endif
}

CUDAMapper::~CUDAMapper()
{
#if HAVE_CUDA
    if(cudaGraphicsUnmapResources(1, m_d->cuda_resource) != cudaSuccess)
    {
        Error{} << "Could not map render buffers for CUDA";
        std::abort();
    }
    if(cudaGraphicsUnregisterResource(&m_d->cuda_resource) != cudaSuccess)
    {
        Error{} << "Could not unregister texture with CUDA";
        std::abort();
    }
#endif
}

void CUDAMapper::readInto(void* cudaDest) const
{
#if HAVE_CUDA
    cudaArray_t array = nullptr;
    if(cudaGraphicsSubResourceGetMappedArray(&array, m_d->cuda_resource, 0, 0) != cudaSuccess)
    {
        Error{} << "Could not get mapped array";
        std::abort();
    }

    auto size = m_d->texture.imageSize();
    auto err = cudaMemcpy2DFromArray(cudaDest, size.x()*m_d->bytesPerPixel, array, 0, 0, size.x()*m_d->bytesPerPixel, size.y(), cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess)
    {
        Error{} << "Could not cudaMemcpy:" << cudaGetErrorString(err);
        std::abort();
    }
#endif
}

}
