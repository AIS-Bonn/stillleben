// CUDA integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/cuda_interop.h>

#include <vector>
#include <algorithm>

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

class CUDATexture::Private
{
public:
#if HAVE_CUDA
    explicit Private(std::size_t bytesPerPixel)
     : bytesPerPixel(bytesPerPixel)
    {}

    ~Private()
    {
#if HAVE_CUDA
        if(cuda_resource)
        {
            if(cudaGraphicsUnregisterResource(cuda_resource) != cudaSuccess)
            {
                Error{} << "Could not unregister texture with CUDA";
                std::abort();
            }
        }
#endif
    }

    std::size_t bytesPerPixel;
    cudaGraphicsResource* cuda_resource = nullptr;
#endif
};

CUDATexture::CUDATexture(CUDAMapper& mapper, Magnum::NoCreateT)
 : RectangleTexture{Magnum::NoCreate}
 , m_parent{mapper}
{
}

CUDATexture::CUDATexture(CUDAMapper& mapper)
 : m_parent{mapper}
{
}

CUDATexture::CUDATexture(CUDATexture&& other)
 : m_parent{other.m_parent}
 , m_d{std::move(other.m_d)}
{
}

CUDATexture::~CUDATexture()
{
    if(m_d)
        m_parent.unregisterMap(*m_d);
}

CUDATexture& CUDATexture::operator=(CUDATexture&& other)
{
    std::swap(m_d, other.m_d);
    RectangleTexture::operator=(std::move(other));

    return *this;
}

void CUDATexture::setStorage(Magnum::GL::TextureFormat internalFormat, std::size_t pixelSize, const Magnum::Vector2i& size)
{
    if(m_d)
        m_parent.unregisterMap(*m_d);

    m_d.reset();

    RectangleTexture::setStorage(internalFormat, size);

#if HAVE_CUDA
    if(m_parent.active())
    {
        m_d = std::make_unique<Private>(pixelSize);

        auto err = cudaGraphicsGLRegisterImage(&m_d->cuda_resource, id(), GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsReadOnly);
        if(err != cudaSuccess)
        {
            Error{} << "Could not register texture with CUDA:" << cudaGetErrorString(err);
            std::abort();
        }

        m_parent.registerMap(*m_d);
    }
#endif
}

void CUDATexture::readIntoCUDA(void* cudaDest)
{
#if HAVE_CUDA
    if(!m_parent.active())
        throw std::logic_error("CUDATexture::readIntoCUDA(): called but CUDAMapper is not active");

    cudaArray_t array = nullptr;
    if(cudaGraphicsSubResourceGetMappedArray(&array, m_d->cuda_resource, 0, 0) != cudaSuccess)
    {
        Error{} << "Could not get mapped array";
        std::abort();
    }

    auto size = imageSize();
    auto err = cudaMemcpy2DFromArray(cudaDest, size.x()*m_d->bytesPerPixel, array, 0, 0, size.x()*m_d->bytesPerPixel, size.y(), cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess)
    {
        Error{} << "Could not cudaMemcpy:" << cudaGetErrorString(err);
        std::abort();
    }
#else
    throw std::runtime_error("CUDATexture::readIntoCUDA(): stillleben was compiled without CUDA support");
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

CUDAMapper::CUDAMapper(bool active)
 : m_active{active}
 , m_d{new CUDAMapper::Private}
{
}

CUDAMapper::~CUDAMapper()
{
}

void CUDAMapper::registerMap(CUDATexture::Private& map)
{
#if HAVE_CUDA
    if(!m_active)
        return;

    m_d->resources.push_back(map.cuda_resource);
#endif
}

void CUDAMapper::unregisterMap(CUDATexture::Private& map)
{
#if HAVE_CUDA
    if(!m_active)
        return;

    if(m_d->mapped)
        cudaGraphicsUnmapResources(1, &map.cuda_resource);

    auto it = std::find(m_d->resources.begin(), m_d->resources.end(), map.cuda_resource);
    if(it != m_d->resources.end())
        m_d->resources.erase(it);
#endif
}

void CUDAMapper::mapAll()
{
#if HAVE_CUDA
    if(!m_active)
        return;

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
    if(!m_active)
        return;

    if(cudaGraphicsUnmapResources(m_d->resources.size(), m_d->resources.data()) != cudaSuccess)
    {
        Error{} << "Could not unmap textures from CUDA";
    }

    m_d->mapped = false;
#endif
}


}
