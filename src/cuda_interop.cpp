// CUDA integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/cuda_interop.h>

#include <vector>

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

class CUDAMap::Private
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

CUDAMap::CUDAMap(CUDAMapper& mapper, GL::RectangleTexture& texture, std::size_t bytesPerPixel)
 : m_parent{mapper}
#if HAVE_CUDA
 , m_d(new Private(texture, bytesPerPixel))
#endif
{
#if HAVE_CUDA
    auto err = cudaGraphicsGLRegisterImage(&m_d->cuda_resource, texture.id(), GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsReadOnly);
    if(err != cudaSuccess)
    {
        Error{} << "Could not register texture with CUDA:" << cudaGetErrorString(err);
        std::abort();
    }

    mapper.registerMap(*m_d);
#else
    throw std::runtime_error("stillleben was compiled without CUDA interop");
#endif
}

CUDAMap::~CUDAMap()
{
    m_parent.unregisterMap(*m_d);

#if HAVE_CUDA
    if(cudaGraphicsUnregisterResource(m_d->cuda_resource) != cudaSuccess)
    {
        Error{} << "Could not unregister texture with CUDA";
        std::abort();
    }
#endif
}

void CUDAMap::readInto(void* cudaDest) const
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


class CUDAMapper::Private
{
public:
#if HAVE_CUDA
    std::vector<cudaGraphicsResource*> resources;
    bool mapped = false;
#endif
};

CUDAMapper::CUDAMapper()
 : m_d{new CUDAMapper::Private}
{
}

CUDAMapper::~CUDAMapper()
{
}

void CUDAMapper::registerMap(CUDAMap::Private& map)
{
#if HAVE_CUDA
    m_d->resources.push_back(map.cuda_resource);
#endif
}

void CUDAMapper::unregisterMap(CUDAMap::Private& map)
{
#if HAVE_CUDA
    if(m_d->mapped)
        cudaGraphicsUnmapResources(1, &map.cuda_resource);

    auto it = std::find(m_d->resources.begin(), m_d->resources.end(), map.cuda_resource);
    if(it != m_d->resource.end())
        m_d->resources.erase(it);
#endif
}

void CUDAMapper::mapAll()
{
#if HAVE_CUDA
    if(cudaGraphicsMapResources(m_d->resources.size(), m_d->resources.data()) != cudaSuccess)
    {
        Error{} << "Could not map textures for CUDA";
        std::abort();
    }

    m_d->mapped = true;
#endif
}

void CUDAMapper::unmapAll()
{
#if HAVE_CUDA
    if(cudaGraphicsUnmapResources(m_d->resources.size(), m_d->resources.data()) != cudaSuccess)
    {
        Error{} << "Could not unmap textures from CUDA";
    }

    m_d->mapped = false;
#endif
}


}
